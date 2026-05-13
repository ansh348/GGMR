"""Phase 2 evaluation: hand-vs-learned A* on hard_v2 and/or phase0.

    python -m ggmr.training.evaluate --ckpt <path> --problems {hard_v2,phase0,all} \
        --device cuda --output ggmr/training/PHASE2_RESULTS.md

For each problem:
  1. A* with WeightedSumCompositeHeuristic, max_nodes=50_000, max_depth=25
  2. A* with LearnedHeuristic(ckpt), same budget
Reports three primary metrics + per-family breakdown + per-problem CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
from ggmr.heuristics.learned import LearnedHeuristic
from ggmr.problems.loader import (
    Problem,
    load_hard_evaluation_set,
    load_phase0_problems,
)
from ggmr.search.astar import astar
from ggmr.training.metrics import geomean_ratio

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to best.pt checkpoint")
    p.add_argument("--problems", choices=["hard_v2", "phase0", "all"], default="all")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="ggmr/training/PHASE2_RESULTS.md")
    p.add_argument("--max-nodes", type=int, default=50_000)
    p.add_argument("--max-depth", type=int, default=25)
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N problems")
    p.add_argument("--offset", type=int, default=0, help="skip the first N problems (for resume)")
    return p.parse_args()


def _run_one(problem: Problem, heuristic, *, max_nodes: int, max_depth: int) -> dict:
    t0 = time.perf_counter()
    try:
        result = astar(
            problem.initial,
            problem.is_target,
            heuristic=heuristic,
            max_nodes=max_nodes,
            max_depth=max_depth,
            check_soundness=True,
            problem_id=problem.id,
        )
        elapsed = time.perf_counter() - t0
        return {
            "found": result.found,
            "nodes_expanded": result.stats.nodes_expanded,
            "nodes_generated": result.stats.nodes_generated,
            "max_depth": result.stats.max_depth_reached,
            "time_ms": elapsed * 1000,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.warning(f"A* crashed on {problem.id}: {type(e).__name__}: {e}")
        return {
            "found": False,
            "nodes_expanded": max_nodes,
            "nodes_generated": 0,
            "max_depth": 0,
            "time_ms": elapsed * 1000,
            "error": f"{type(e).__name__}: {e}",
        }


def evaluate_problems(
    problems: list[Problem], ckpt_path: str, device: str, max_nodes: int, max_depth: int
) -> list[dict]:
    hand = WeightedSumCompositeHeuristic()
    learned = LearnedHeuristic(ckpt_path, device=device)
    rows: list[dict] = []
    for i, prob in enumerate(problems):
        hand_res = _run_one(prob, hand, max_nodes=max_nodes, max_depth=max_depth)
        learned_res = _run_one(prob, learned, max_nodes=max_nodes, max_depth=max_depth)
        rows.append({
            "id": prob.id,
            "family": prob.family,
            "source": prob.source,
            "hand_found": hand_res["found"],
            "hand_nodes": hand_res["nodes_expanded"],
            "hand_time_ms": hand_res["time_ms"],
            "learned_found": learned_res["found"],
            "learned_nodes": learned_res["nodes_expanded"],
            "learned_time_ms": learned_res["time_ms"],
        })
        cmp_str = ""
        if hand_res["found"] and learned_res["found"]:
            ratio = hand_res["nodes_expanded"] / max(learned_res["nodes_expanded"], 1)
            cmp_str = f"  cmp={ratio:.2f}x"
        logger.info(
            f"[{i+1}/{len(problems)}] {prob.id}: "
            f"hand={'Y' if hand_res['found'] else 'N'}/{hand_res['nodes_expanded']:>5}  "
            f"learned={'Y' if learned_res['found'] else 'N'}/{learned_res['nodes_expanded']:>5}"
            f"{cmp_str}"
        )
    return rows


def _aggregate(rows: list[dict]) -> dict:
    """Three primary metrics + per-family breakdown."""
    joint = [r for r in rows if r["hand_found"] and r["learned_found"]]
    hand_only = [r for r in rows if r["hand_found"] and not r["learned_found"]]
    learned_only = [r for r in rows if not r["hand_found"] and r["learned_found"]]
    both_fail = [r for r in rows if not r["hand_found"] and not r["learned_found"]]

    geomean = geomean_ratio(
        [r["hand_nodes"] for r in joint],
        [r["learned_nodes"] for r in joint],
    )
    ratios = [r["hand_nodes"] / max(r["learned_nodes"], 1) for r in joint]
    median = float(np.median(ratios)) if ratios else 0.0

    by_family: dict[str, list[float]] = defaultdict(list)
    for r in joint:
        by_family[r["family"]].append(r["hand_nodes"] / max(r["learned_nodes"], 1))
    family_geomean: dict[str, float] = {}
    for fam, rats in by_family.items():
        logs = [np.log(x) for x in rats if x > 0]
        family_geomean[fam] = float(np.exp(np.mean(logs))) if logs else 0.0

    return {
        "total": len(rows),
        "joint_solved": len(joint),
        "hand_only_solved": len(hand_only),
        "learned_only_solved": len(learned_only),
        "both_failed": len(both_fail),
        "geomean_compression": geomean,
        "median_compression": median,
        "family_geomean": family_geomean,
        "hand_only_ids": [r["id"] for r in hand_only],
        "learned_only_ids": [r["id"] for r in learned_only],
    }


def _write_markdown(rows: list[dict], agg: dict, out_path: Path, ckpt: str) -> None:
    lines: list[str] = []
    lines.append(f"# Phase 2 Evaluation Results\n")
    lines.append(f"Checkpoint: `{ckpt}`\n\n")
    lines.append("## Aggregate\n")
    lines.append(f"- Total problems: {agg['total']}")
    lines.append(f"- Joint solved (hand AND learned): {agg['joint_solved']}")
    lines.append(f"- Hand-only solved: {agg['hand_only_solved']}  (regression detector)")
    lines.append(f"- Learned-only solved: {agg['learned_only_solved']}  (new problems opened)")
    lines.append(f"- Both failed: {agg['both_failed']}")
    lines.append(f"- **Geomean compression (joint)**: {agg['geomean_compression']:.3f}x")
    lines.append(f"- Median compression (joint): {agg['median_compression']:.3f}x")
    if agg["hand_only_ids"]:
        lines.append(f"- Regression IDs: {', '.join(agg['hand_only_ids'])}")
    lines.append("")
    lines.append("## Per-family geomean (joint-solved subset)\n")
    lines.append("| family | geomean |")
    lines.append("|---|---|")
    for fam, g in sorted(agg["family_geomean"].items()):
        lines.append(f"| {fam} | {g:.3f}x |")
    lines.append("")
    lines.append("## Per-problem\n")
    lines.append("| id | family | hand | hand_nodes | learned | learned_nodes | ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        ratio = (
            f"{r['hand_nodes'] / max(r['learned_nodes'], 1):.2f}x"
            if r["hand_found"] and r["learned_found"]
            else "-"
        )
        lines.append(
            f"| {r['id']} | {r['family']} | "
            f"{'Y' if r['hand_found'] else 'N'} | {r['hand_nodes']} | "
            f"{'Y' if r['learned_found'] else 'N'} | {r['learned_nodes']} | {ratio} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    problems: list[Problem] = []
    if args.problems in ("hard_v2", "all"):
        hard = load_hard_evaluation_set()
        problems.extend(hard)
        logger.info(f"loaded {len(hard)} hard_v2 problems")
    if args.problems in ("phase0", "all"):
        p0 = load_phase0_problems()
        problems.extend(p0)
        logger.info(f"loaded {len(p0)} phase0 problems")
    if args.offset:
        problems = problems[args.offset:]
        logger.info(f"--offset {args.offset}: starting at problem index {args.offset}")
    if args.limit:
        problems = problems[: args.limit]

    rows = evaluate_problems(
        problems, args.ckpt, args.device, args.max_nodes, args.max_depth
    )
    agg = _aggregate(rows)

    out_path = Path(args.output)
    _write_markdown(rows, agg, out_path, args.ckpt)

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    print("=" * 60)
    print("evaluation summary")
    print("=" * 60)
    print(f"  total: {agg['total']}")
    print(f"  joint solved: {agg['joint_solved']}")
    print(f"  hand-only: {agg['hand_only_solved']}  (regressions)")
    print(f"  learned-only: {agg['learned_only_solved']}  (new wins)")
    print(f"  GEOMEAN compression: {agg['geomean_compression']:.3f}x")
    print(f"  median compression: {agg['median_compression']:.3f}x")
    print(f"  markdown: {out_path}")
    print(f"  csv:      {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
