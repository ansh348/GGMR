"""Phase 3 (ExIt) evaluation: A* + MCTS dual metric on hard_v2/phase0/external.

Compared to `ggmr.training.evaluate`, this script:
  - Loads a value checkpoint AND a policy checkpoint
  - Runs A* with the new LearnedHeuristic (hand-vs-learned compression)
  - Runs MCTS at inference (simulations-to-solve) using value+policy advisors
  - Optionally runs A*-with-policy-ordering (LearnedHeuristic + policy reorders rule enumeration)
  - Reports geomean compression, geomean MCTS sims, regression counts, family breakdown.

    python -m ggmr.training.evaluate_exit \
        --value-ckpt checkpoints/exit_v1/value_iter_02.pt \
        --policy-ckpt checkpoints/exit_v1/policy_iter_02.pt \
        --problems all \
        --output exit_v1_results.md \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import ggmr.rules.core  # noqa: F401  (register rules)
from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
from ggmr.heuristics.learned import LearnedHeuristic
from ggmr.problems.loader import Problem, load_hard_evaluation_set, load_phase0_problems
from ggmr.rules.registry import Registry, default_registry
from ggmr.search.astar import astar
from ggmr.search.mcts import mcts_search
from ggmr.state import EqState
from ggmr.training.metrics import geomean_ratio
from ggmr.training.policy_heuristic import PolicyAdvisor, ValueAdvisor

logger = logging.getLogger(__name__)


class PolicyOrderedRegistry:
    """Wrapper around `Registry` that reorders `enumerate_actions` by descending
    policy logit. A* doesn't care about enumeration order for correctness — only
    for tie-breaking via the monotonic counter — but a well-ordered enumeration
    causes A* to expand the "right" child first when f-scores are close, which
    cuts node-expansion counts on problems where the value heuristic alone
    spreads probability across many similar-cost actions.

    Used in eval mode "learned+policy-ordering" to isolate the policy's
    contribution from the value heuristic's.
    """

    def __init__(self, base: Registry, advisor: PolicyAdvisor):
        self._base = base
        self._advisor = advisor

    def enumerate_actions(self, state: EqState, *, training_only: bool = False):
        pairs = list(self._base.enumerate_actions(state, training_only=training_only))
        # Higher logit first; ties preserve canonical order (stable sort).
        pairs.sort(key=lambda ra: -self._advisor.action_ordering_key(state, ra[1]))
        return iter(pairs)


def _run_astar(
    problem: Problem, heuristic, *, max_nodes: int, max_depth: int,
    rules: Registry | PolicyOrderedRegistry | None = None,
) -> dict:
    t0 = time.perf_counter()
    try:
        result = astar(
            problem.initial, problem.is_target,
            heuristic=heuristic, max_nodes=max_nodes, max_depth=max_depth,
            check_soundness=True, problem_id=problem.id,
            rules=rules if rules is not None else default_registry,
        )
        return {
            "found": result.found,
            "nodes_expanded": result.stats.nodes_expanded,
            "path_length": result.num_steps if result.found else 0,
            "time_ms": (time.perf_counter() - t0) * 1000,
        }
    except Exception as e:
        logger.warning(f"A* crashed on {problem.id}: {type(e).__name__}: {e}")
        return {"found": False, "nodes_expanded": max_nodes, "path_length": 0,
                "time_ms": (time.perf_counter() - t0) * 1000, "error": str(e)}


def _run_mcts(
    problem: Problem,
    value_advisor: ValueAdvisor, policy_advisor: PolicyAdvisor,
    *, num_simulations: int, max_moves: int, c_puct: float,
) -> dict:
    t0 = time.perf_counter()
    try:
        result = mcts_search(
            problem.initial, problem.is_target,
            value_fn=value_advisor.value_fn, policy_fn=policy_advisor.policy_fn,
            num_simulations=num_simulations, max_moves=max_moves, c_puct=c_puct,
        )
        return {
            "found": result.found,
            "total_simulations": result.stats.total_simulations,
            "nodes_expanded": result.stats.nodes_expanded,
            "path_length": result.num_steps,
            "moves_taken": result.stats.moves_taken,
            "time_ms": (time.perf_counter() - t0) * 1000,
        }
    except Exception as e:
        logger.warning(f"MCTS crashed on {problem.id}: {type(e).__name__}: {e}")
        return {"found": False, "total_simulations": num_simulations * max_moves,
                "nodes_expanded": 0, "path_length": 0, "moves_taken": 0,
                "time_ms": (time.perf_counter() - t0) * 1000, "error": str(e)}


def evaluate_dual(
    problems: list[Problem],
    *,
    value_ckpt: Path,
    policy_ckpt: Path | None,
    device: str,
    astar_max_nodes: int,
    astar_max_depth: int,
    mcts_simulations: int,
    mcts_max_moves: int,
    c_puct: float,
    run_mcts: bool,
    run_policy_ordering: bool = True,
) -> list[dict]:
    hand = WeightedSumCompositeHeuristic()
    learned = LearnedHeuristic(value_ckpt, device=device)
    va = ValueAdvisor(value_ckpt, device=device)
    pa = PolicyAdvisor(policy_ckpt, device=device) if policy_ckpt else PolicyAdvisor(None, device=device)
    policy_registry = (
        PolicyOrderedRegistry(default_registry, pa)
        if run_policy_ordering and policy_ckpt is not None
        else None
    )
    rows: list[dict] = []
    for i, prob in enumerate(problems):
        hand_res = _run_astar(prob, hand, max_nodes=astar_max_nodes, max_depth=astar_max_depth)
        learned_res = _run_astar(prob, learned, max_nodes=astar_max_nodes, max_depth=astar_max_depth)
        pol_res = (
            _run_astar(prob, learned, max_nodes=astar_max_nodes,
                       max_depth=astar_max_depth, rules=policy_registry)
            if policy_registry is not None else None
        )
        mcts_res = (
            _run_mcts(prob, va, pa,
                      num_simulations=mcts_simulations, max_moves=mcts_max_moves, c_puct=c_puct)
            if run_mcts else None
        )
        row = {
            "id": prob.id, "family": prob.family, "source": prob.source,
            "hand_found": hand_res["found"], "hand_nodes": hand_res["nodes_expanded"],
            "learned_found": learned_res["found"], "learned_nodes": learned_res["nodes_expanded"],
        }
        if pol_res is not None:
            row["pol_found"] = pol_res["found"]
            row["pol_nodes"] = pol_res["nodes_expanded"]
        if mcts_res is not None:
            row.update({
                "mcts_found": mcts_res["found"],
                "mcts_sims": mcts_res["total_simulations"],
                "mcts_path_length": mcts_res["path_length"],
                "mcts_moves": mcts_res["moves_taken"],
            })
        rows.append(row)
        cmp_astar = (f"{hand_res['nodes_expanded'] / max(learned_res['nodes_expanded'], 1):.2f}x"
                     if hand_res["found"] and learned_res["found"] else "-")
        cmp_pol = (f"pol={pol_res['nodes_expanded']}({hand_res['nodes_expanded']/max(pol_res['nodes_expanded'],1):.2f}x)"
                   if pol_res and pol_res["found"] and hand_res["found"] else
                   ("pol-fail" if pol_res else ""))
        cmp_mcts = (f"sims={mcts_res['total_simulations']}" if mcts_res and mcts_res["found"]
                    else ("MCTS-fail" if mcts_res else ""))
        logger.info(
            f"[{i+1}/{len(problems)}] {prob.id}: "
            f"hand={'Y' if hand_res['found'] else 'N'}/{hand_res['nodes_expanded']} "
            f"learned={'Y' if learned_res['found'] else 'N'}/{learned_res['nodes_expanded']}  "
            f"astar_cmp={cmp_astar}  {cmp_pol}  {cmp_mcts}"
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    joint_astar = [r for r in rows if r["hand_found"] and r["learned_found"]]
    hand_only = [r for r in rows if r["hand_found"] and not r["learned_found"]]
    learned_only = [r for r in rows if not r["hand_found"] and r["learned_found"]]
    both_fail = [r for r in rows if not r["hand_found"] and not r["learned_found"]]

    astar_geomean = geomean_ratio(
        [r["hand_nodes"] for r in joint_astar],
        [r["learned_nodes"] for r in joint_astar],
    )
    ratios = [r["hand_nodes"] / max(r["learned_nodes"], 1) for r in joint_astar]
    astar_median = float(np.median(ratios)) if ratios else 0.0

    by_family = defaultdict(list)
    for r in joint_astar:
        by_family[r["family"]].append(r["hand_nodes"] / max(r["learned_nodes"], 1))
    family_astar = {f: float(np.exp(np.mean([np.log(x) for x in v if x > 0])))
                    if v else 0.0 for f, v in by_family.items()}

    out: dict = {
        "total": len(rows),
        "astar_joint_solved": len(joint_astar),
        "astar_hand_only": len(hand_only),
        "astar_learned_only": len(learned_only),
        "astar_both_failed": len(both_fail),
        "astar_geomean": astar_geomean,
        "astar_median": astar_median,
        "family_astar_geomean": family_astar,
        "hand_only_ids": [r["id"] for r in hand_only],
        "learned_only_ids": [r["id"] for r in learned_only],
    }

    # Per-source breakdown (hard_v2 vs phase0) for clean comparison vs Round 2 numbers.
    by_source: dict[str, dict] = {}
    for src in {r.get("source", "?") for r in rows}:
        src_rows = [r for r in rows if r.get("source") == src]
        src_joint = [r for r in src_rows if r["hand_found"] and r["learned_found"]]
        if src_joint:
            by_source[src] = {
                "total": len(src_rows),
                "joint_solved": len(src_joint),
                "astar_geomean": geomean_ratio(
                    [r["hand_nodes"] for r in src_joint],
                    [r["learned_nodes"] for r in src_joint],
                ),
                "learned_found": sum(1 for r in src_rows if r["learned_found"]),
                "hand_found": sum(1 for r in src_rows if r["hand_found"]),
            }
        else:
            by_source[src] = {"total": len(src_rows), "joint_solved": 0}
    out["by_source"] = by_source

    if any("pol_found" in r for r in rows):
        pol_joint = [r for r in rows
                     if r["hand_found"] and r.get("pol_found")]
        pol_geomean = geomean_ratio(
            [r["hand_nodes"] for r in pol_joint],
            [r["pol_nodes"] for r in pol_joint],
        )
        pol_ratios = [r["hand_nodes"] / max(r["pol_nodes"], 1) for r in pol_joint]
        out["pol_joint_solved"] = len(pol_joint)
        out["pol_geomean"] = pol_geomean
        out["pol_median"] = float(np.median(pol_ratios)) if pol_ratios else 0.0
        # per-source breakdown
        for src in {r.get("source", "?") for r in rows}:
            src_pol_joint = [r for r in rows
                             if r.get("source") == src
                             and r["hand_found"] and r.get("pol_found")]
            if src_pol_joint:
                by_source[src]["pol_geomean"] = geomean_ratio(
                    [r["hand_nodes"] for r in src_pol_joint],
                    [r["pol_nodes"] for r in src_pol_joint],
                )
                by_source[src]["pol_joint_solved"] = len(src_pol_joint)

    if any("mcts_found" in r for r in rows):
        mcts_solved = [r for r in rows if r.get("mcts_found")]
        mcts_unsolved = [r for r in rows if "mcts_found" in r and not r["mcts_found"]]
        sims = [r["mcts_sims"] for r in mcts_solved]
        out["mcts_solved"] = len(mcts_solved)
        out["mcts_unsolved"] = len(mcts_unsolved)
        out["mcts_median_sims"] = float(np.median(sims)) if sims else 0.0
        out["mcts_mean_sims"] = float(np.mean(sims)) if sims else 0.0

        # Super-BFS check: MCTS path length < A* path length for joint-solved problems
        joint = [r for r in rows if r.get("mcts_found") and r.get("learned_found")]
        # We don't have learned-path-length directly (learned A* returns nodes_expanded,
        # not path length). For now: compare MCTS path_length to hand A* path? Best signal
        # is that mcts_path_length < some baseline. Just report the lengths.
        out["mcts_avg_path_length"] = float(np.mean([r["mcts_path_length"] for r in mcts_solved])) \
            if mcts_solved else 0.0

    return out


def write_markdown(rows: list[dict], agg: dict, out: Path, value_ckpt: str, policy_ckpt: str | None) -> None:
    lines: list[str] = []
    lines.append("# Phase 3 ExIt Evaluation Results\n")
    lines.append(f"- Value checkpoint: `{value_ckpt}`")
    lines.append(f"- Policy checkpoint: `{policy_ckpt}`\n")

    lines.append("## A* metric (hand vs learned)\n")
    lines.append(f"- Total problems: {agg['total']}")
    lines.append(f"- Joint solved: {agg['astar_joint_solved']}")
    lines.append(f"- Hand-only: {agg['astar_hand_only']} (regression detector)")
    lines.append(f"- Learned-only: {agg['astar_learned_only']} (new wins)")
    lines.append(f"- Both failed: {agg['astar_both_failed']}")
    lines.append(f"- **A* geomean compression**: {agg['astar_geomean']:.3f}x")
    lines.append(f"- A* median compression: {agg['astar_median']:.3f}x")
    if agg["hand_only_ids"]:
        lines.append(f"- Regression IDs: {', '.join(agg['hand_only_ids'])}")
    lines.append("")

    if "pol_geomean" in agg:
        lines.append("## A*-with-policy-ordering metric (hand vs learned+policy)\n")
        lines.append(f"- Joint solved: {agg['pol_joint_solved']}")
        lines.append(f"- **A*+policy geomean compression**: {agg['pol_geomean']:.3f}x")
        lines.append(f"- A*+policy median compression: {agg['pol_median']:.3f}x")
        lines.append("")

    lines.append("## Per-source breakdown\n")
    lines.append("| source | total | hand_found | learned_found | joint | A* geomean | A*+pol geomean |")
    lines.append("|---|---|---|---|---|---|---|")
    for src, s in sorted(agg.get("by_source", {}).items()):
        lines.append(
            f"| {src} | {s.get('total', '-')} | {s.get('hand_found', '-')} | "
            f"{s.get('learned_found', '-')} | {s.get('joint_solved', '-')} | "
            f"{s.get('astar_geomean', 0):.3f}x | "
            f"{s.get('pol_geomean', 0):.3f}x |"
        )
    lines.append("")

    if "mcts_solved" in agg:
        lines.append("## MCTS metric (simulations to solve)\n")
        lines.append(f"- MCTS solved: {agg['mcts_solved']}")
        lines.append(f"- MCTS unsolved: {agg['mcts_unsolved']}")
        lines.append(f"- **MCTS median sims**: {agg['mcts_median_sims']:.0f}")
        lines.append(f"- MCTS mean sims: {agg['mcts_mean_sims']:.0f}")
        lines.append(f"- MCTS avg path length: {agg['mcts_avg_path_length']:.1f}")
        lines.append("")

    lines.append("## Per-family A* geomean\n")
    lines.append("| family | geomean |")
    lines.append("|---|---|")
    for fam, g in sorted(agg["family_astar_geomean"].items()):
        lines.append(f"| {fam} | {g:.3f}x |")
    lines.append("")

    lines.append("## Per-problem detail\n")
    headers = ["id", "family", "hand", "hand_nodes", "learned", "learned_nodes",
               "astar_cmp", "mcts_found", "mcts_sims", "mcts_path_len"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for r in rows:
        astar_cmp = (f"{r['hand_nodes'] / max(r['learned_nodes'], 1):.2f}x"
                     if r["hand_found"] and r["learned_found"] else "-")
        lines.append("| " + " | ".join([
            r["id"], r["family"],
            "Y" if r["hand_found"] else "N", str(r["hand_nodes"]),
            "Y" if r["learned_found"] else "N", str(r["learned_nodes"]),
            astar_cmp,
            ("Y" if r.get("mcts_found") else ("N" if "mcts_found" in r else "-")),
            str(r.get("mcts_sims", "-")),
            str(r.get("mcts_path_length", "-")),
        ]) + " |")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--value-ckpt", type=Path, required=True)
    p.add_argument("--policy-ckpt", type=Path, default=None)
    p.add_argument("--problems", choices=["hard_v2", "phase0", "all"], default="all")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=Path("PHASE3_RESULTS.md"))
    p.add_argument("--astar-max-nodes", type=int, default=50_000)
    p.add_argument("--astar-max-depth", type=int, default=25)
    p.add_argument("--mcts-simulations", type=int, default=400)
    p.add_argument("--mcts-max-moves", type=int, default=20)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--no-mcts", action="store_true", help="skip MCTS eval (A* only)")
    p.add_argument("--no-policy-ordering", action="store_true",
                   help="skip A*-with-policy-ordering eval")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    problems: list[Problem] = []
    if args.problems in ("hard_v2", "all"):
        h = load_hard_evaluation_set()
        problems.extend(h)
        logger.info(f"loaded {len(h)} hard_v2 problems")
    if args.problems in ("phase0", "all"):
        p0 = load_phase0_problems()
        problems.extend(p0)
        logger.info(f"loaded {len(p0)} phase0 problems")
    if args.limit:
        problems = problems[: args.limit]

    rows = evaluate_dual(
        problems,
        value_ckpt=args.value_ckpt, policy_ckpt=args.policy_ckpt, device=args.device,
        astar_max_nodes=args.astar_max_nodes, astar_max_depth=args.astar_max_depth,
        mcts_simulations=args.mcts_simulations, mcts_max_moves=args.mcts_max_moves,
        c_puct=args.c_puct, run_mcts=not args.no_mcts,
        run_policy_ordering=not args.no_policy_ordering,
    )
    agg = aggregate(rows)
    write_markdown(rows, agg, args.output, str(args.value_ckpt), str(args.policy_ckpt) if args.policy_ckpt else None)

    csv_path = args.output.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            w.writeheader()
            w.writerows(rows)

    json_path = args.output.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print("=" * 60)
    print("Phase 3 evaluation summary")
    print("=" * 60)
    print(f"  A* geomean: {agg['astar_geomean']:.3f}x")
    print(f"  A* hand-only (regressions): {agg['astar_hand_only']}")
    print(f"  A* learned-only (new wins): {agg['astar_learned_only']}")
    if "pol_geomean" in agg:
        print(f"  A*+policy geomean: {agg['pol_geomean']:.3f}x")
    if "mcts_median_sims" in agg:
        print(f"  MCTS solved: {agg['mcts_solved']}/{agg['total']}")
        print(f"  MCTS median sims: {agg['mcts_median_sims']:.0f}")
    for src, s in sorted(agg.get("by_source", {}).items()):
        msg = (f"  [{src}] A*={s.get('astar_geomean', 0):.3f}x "
               f"joint={s.get('joint_solved', '-')}/{s.get('total', '-')}")
        if "pol_geomean" in s:
            msg += f"  A*+pol={s['pol_geomean']:.3f}x"
        print(msg)
    print(f"  markdown: {args.output}")
    print(f"  csv: {csv_path}")
    print(f"  json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
