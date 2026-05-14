"""Extract (state, BFS-optimal action) tuples for policy warm-start.

Generates problems via `ggmr.problems.round2_categories.CATEGORIES`, BFS-solves
each, and emits one tuple per intermediate state along the path. Splits the
generated problems into train (90%) and held-out (10%) sets so we can measure
policy top-1 accuracy on unseen states.

Output JSONL schema:
    problem_id: str
    category:   str
    split:      "train" | "held_out"
    step_index: int       # 0-indexed step along the BFS path
    path_length: int      # total length of the BFS path
    state_lhs_srepr: str
    state_rhs_srepr: str
    var: str
    excluded_srepr: list[str]
    rule_name: str        # the BFS-optimal action's rule_name from this state
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sympy as sp

# Register rules
import ggmr.rules.core  # noqa: F401
from ggmr.problems.round2_categories import CATEGORIES, bfs_budget_for, timeout_for
from ggmr.search.bfs import bfs
from ggmr.state import EqState
from ggmr.training.extract_pairs import _build_is_target

logger = logging.getLogger(__name__)

_DEPTH_CYCLE = (3, 5, 8, 10, 12, 15)
_HELD_OUT_FRAC = 0.10


def _seed_for(category: str, idx: int, base_seed: int) -> int:
    return (base_seed * 1_000_003 + hash(("warmstart", category, idx)) % 1_000_003)


def _generate_problem(category: str, idx: int, base_seed: int) -> dict | None:
    """Generate one (initial, target) pair via the Round 2 category function."""
    if category not in CATEGORIES:
        return None
    rng = random.Random(_seed_for(category, idx, base_seed))
    depth = _DEPTH_CYCLE[idx % len(_DEPTH_CYCLE)]
    try:
        instance = CATEGORIES[category](rng, depth)
    except Exception as e:
        logger.debug(f"gen_fail {category}/{idx}: {e}")
        return None
    return {
        "category": category,
        "idx": idx,
        "depth": depth,
        "initial": instance.eq_state,
        "target": instance.target_eq_state,
    }


def _bfs_path(problem: dict, max_nodes: int) -> list[tuple[EqState, str]] | None:
    initial: EqState = problem["initial"]
    target: EqState = problem["target"]
    is_target = _build_is_target(target)
    if is_target(initial):
        return []  # already solved; no tuples to emit
    result = bfs(
        initial,
        is_target,
        max_nodes=max_nodes,
        max_depth=30,
        check_soundness=False,
        problem_id=f"warmstart_{problem['category']}_{problem['idx']:05d}",
    )
    if not result.found:
        return None
    return [(s, a.rule_name) for s, a in result.path]


def _to_record(
    state: EqState,
    rule_name: str,
    problem: dict,
    step_index: int,
    path_length: int,
    split: str,
) -> dict:
    return {
        "problem_id": f"warmstart_{problem['category']}_{problem['idx']:05d}",
        "category": problem["category"],
        "split": split,
        "step_index": step_index,
        "path_length": path_length,
        "state_lhs_srepr": sp.srepr(state.lhs),
        "state_rhs_srepr": sp.srepr(state.rhs),
        "var": state.var.name,
        "excluded_srepr": sorted(sp.srepr(e) for e in state.excluded),
        "rule_name": rule_name,
    }


def _process_one(args: tuple[str, int, int, str]) -> list[dict]:
    """Worker: generate + BFS-solve + emit records for one problem."""
    category, idx, base_seed, split = args
    problem = _generate_problem(category, idx, base_seed)
    if problem is None:
        return []
    max_nodes = bfs_budget_for(category)
    try:
        path = _bfs_path(problem, max_nodes)
    except Exception as e:
        logger.debug(f"bfs_fail {category}/{idx}: {e}")
        return []
    if path is None or not path:
        return []
    n = len(path)
    return [
        _to_record(state, rule_name, problem, i, n, split)
        for i, (state, rule_name) in enumerate(path)
    ]


def run(
    *,
    num_problems: int,
    output: Path,
    base_seed: int = 42,
    held_out_frac: float = _HELD_OUT_FRAC,
    max_workers: int = 8,
) -> dict:
    """Generate `num_problems` problems stratified across categories, BFS-solve
    each, emit (state, rule_name) tuples to `output`.

    Returns a stats dict.
    """
    cats = list(CATEGORIES.keys())
    per_cat = max(1, num_problems // len(cats))
    jobs: list[tuple[str, int, int, str]] = []
    held_out_per_cat = max(1, int(per_cat * held_out_frac))
    for cat in cats:
        for idx in range(per_cat):
            split = "held_out" if idx < held_out_per_cat else "train"
            jobs.append((cat, idx, base_seed, split))

    output.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    n_problems_solved = 0
    n_problems_failed = 0
    by_split: dict[str, int] = {"train": 0, "held_out": 0}
    by_category: dict[str, int] = {}

    t0 = time.perf_counter()

    def _drain(iterator):
        nonlocal n_records, n_problems_solved, n_problems_failed
        for i, recs in enumerate(iterator):
            if not recs:
                n_problems_failed += 1
            else:
                n_problems_solved += 1
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
                    n_records += 1
                    by_split[r["split"]] = by_split.get(r["split"], 0) + 1
                    by_category[r["category"]] = by_category.get(r["category"], 0) + 1
            if (i + 1) % 100 == 0:
                rate = (i + 1) / max(time.perf_counter() - t0, 1e-9)
                logger.info(
                    f"progress {i + 1}/{len(jobs)} | solved={n_problems_solved} "
                    f"failed={n_problems_failed} records={n_records} rate={rate:.1f}/s"
                )

    with output.open("w", encoding="utf-8") as fh:
        if max_workers <= 1:
            _drain(_process_one(j) for j in jobs)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_process_one, j) for j in jobs]
                _drain(f.result() for f in as_completed(futures))

    elapsed = time.perf_counter() - t0
    stats = {
        "num_jobs": len(jobs),
        "num_problems_solved": n_problems_solved,
        "num_problems_failed": n_problems_failed,
        "num_records": n_records,
        "by_split": by_split,
        "by_category": by_category,
        "elapsed_s": elapsed,
        "solve_rate": n_problems_solved / len(jobs) if jobs else 0.0,
    }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=900,
                        help="Approximate number of problems to generate (rounded by category count)")
    parser.add_argument("--output", type=Path, default=Path("policy_warmstart.jsonl"))
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--held-out-frac", type=float, default=_HELD_OUT_FRAC)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stats = run(
        num_problems=args.num_problems,
        output=args.output,
        base_seed=args.base_seed,
        held_out_frac=args.held_out_frac,
        max_workers=args.max_workers,
    )
    print(json.dumps(stats, indent=2))

    if stats["solve_rate"] < 0.5:
        print(f"WARNING: low solve rate ({stats['solve_rate']:.2%})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
