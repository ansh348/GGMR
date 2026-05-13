"""Run learned heuristic on the L2 and R3 problems that BOTH BFS and A* timed out on
during validation (`motif_candidates_v2_timeouts_validation_report.json`).

Baseline: hand A* hit 1800s wall-clock without solving.
Question: does the learned heuristic crack them?
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sympy as sp
import yaml
from dataclasses import replace

from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
from ggmr.heuristics.learned import LearnedHeuristic
from ggmr.search.astar import astar
from ggmr.state import EqState
from ggmr.training.extract_pairs import _build_is_target
from ggmr.training.srepr_parse import parse_srepr

YAML_PATH = Path("ggmr/problems/motif_candidates_v2_timeouts.yaml")
TARGET_IDS = [
    "motif_v2_001_L2_cross_reciprocal",
    "motif_v2_013_R3_cross_reciprocal_fractional",
]
CKPT = "checkpoints/full/best.pt"
MAX_NODES = 50_000
MAX_DEPTH = 30


def _short(state: EqState, lim: int = 90) -> str:
    s = f"{state.lhs}  =  {state.rhs}"
    return s if len(s) <= lim else s[: lim - 3] + "..."


def main() -> int:
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_id = {e["id"]: e for e in entries}

    print(f"\nLoading LearnedHeuristic from {CKPT}...")
    learned = LearnedHeuristic(CKPT, device="cuda")
    hand = WeightedSumCompositeHeuristic()

    print("\nValidation baseline (motif_candidates_v2_timeouts_validation_report.json):")
    print("  hand BFS:  TIMEOUT @ 1800s (30 min)")
    print("  hand A*:   TIMEOUT @ 1800s (30 min)")
    print(f"\nLearned heuristic budget: max_nodes={MAX_NODES}, max_depth={MAX_DEPTH}")
    print(f"GPU: RTX 3050 (cu124)")
    print("=" * 72)

    for tid in TARGET_IDS:
        e = by_id.get(tid)
        if e is None:
            print(f"\n*** {tid} not found in YAML ***")
            continue
        print(f"\n--- {tid} ({e['category']}) ---")
        var_name = e["variable"]
        initial = EqState.from_strings(e["initial"]["lhs"], e["initial"]["rhs"], var_name=var_name)
        excluded_srepr = e.get("excluded_srepr") or []
        if excluded_srepr:
            excluded = [parse_srepr(s) for s in excluded_srepr]
            initial = initial.with_excluded(*excluded)
        target = EqState.from_strings(
            e["canonical_target"]["lhs"], str(e["canonical_target"]["rhs"]),
            var_name=var_name,
        )
        is_target = _build_is_target(target)

        print(f"  Initial equation:")
        print(f"    LHS: {e['initial']['lhs']}")
        print(f"    RHS: {e['initial']['rhs']}")
        print(f"  Canonical target: {target.lhs} = {target.rhs}")
        if initial.excluded:
            print(f"  Excluded: {set(initial.excluded)}")

        print(f"\n  Initial state hand-h:    {hand.evaluate(initial):.2f}")
        print(f"  Initial state learned-h: {learned.evaluate(initial):.3f}")

        print(f"\n  >>> Running A* with LearnedHeuristic ...")
        t0 = time.perf_counter()
        try:
            result = astar(
                initial, is_target, heuristic=learned,
                max_nodes=MAX_NODES, max_depth=MAX_DEPTH,
                problem_id=tid,
            )
            elapsed = time.perf_counter() - t0
        except Exception as ex:
            elapsed = time.perf_counter() - t0
            print(f"  *** A* CRASHED: {type(ex).__name__}: {ex} ***")
            print(f"  elapsed: {elapsed:.1f}s")
            continue

        print(f"  Wall time: {elapsed:.1f}s")
        print(f"  Solved:           {result.found}")
        print(f"  Nodes expanded:   {result.stats.nodes_expanded}")
        print(f"  Nodes generated:  {result.stats.nodes_generated}")
        print(f"  Max depth:        {result.stats.max_depth_reached}")
        print(f"  Path length:      {len(result.path)}")

        if result.found and result.path:
            print(f"\n  Solution path:")
            print(f"    [start] {_short(initial)}")
            for i, (state, action) in enumerate(result.path):
                rule = getattr(action, "rule_name", type(action).__name__)
                print(f"    [step {i+1}] {rule}")
                print(f"             -> {_short(state)}")
            print(f"    [end]   {_short(result.final_state) if result.final_state else 'N/A'}")
        elif not result.found:
            print(f"  *** UNSOLVED within budget ***")

    print("\n" + "=" * 72)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
