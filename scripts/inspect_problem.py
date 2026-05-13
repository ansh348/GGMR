"""Deep-dive on a single hard_v2 problem: equation, target, and A* traces
for both hand and learned heuristics.

Usage:
    python scripts/inspect_problem.py --id hard_motif_L1_004 --ckpt checkpoints/full/best.pt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sympy as sp

from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
from ggmr.heuristics.learned import LearnedHeuristic
from ggmr.problems.loader import load_hard_evaluation_set
from ggmr.search.astar import astar
from ggmr.state import EqState


@dataclass
class _CallRecord:
    order: int
    h: float
    state: EqState


class _TracingHeuristic:
    """Wraps any Heuristic, records every evaluate(state) -> h call."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[_CallRecord] = []

    def evaluate(self, state: EqState) -> float:
        h = float(self._inner.evaluate(state))
        self.calls.append(_CallRecord(order=len(self.calls), h=h, state=state))
        return h


def _short_eq(state: EqState, max_len: int = 100) -> str:
    s = f"{state.lhs}  =  {state.rhs}"
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-nodes", type=int, default=50_000)
    p.add_argument("--max-depth", type=int, default=25)
    args = p.parse_args()

    problems = load_hard_evaluation_set()
    target = next((pr for pr in problems if pr.id == args.id), None)
    if target is None:
        print(f"No problem with id {args.id!r}.")
        return 1

    print("=" * 72)
    print(f"Problem: {target.id}   family={target.family}")
    print("=" * 72)
    print("\nInitial equation (raw):")
    print(f"  LHS: {target.initial.lhs}")
    print(f"  RHS: {target.initial.rhs}")
    print("\nInitial equation (pretty):")
    print(sp.pretty(sp.Eq(target.initial.lhs, target.initial.rhs, evaluate=False)))
    print("\nInitial equation (LaTeX):")
    print(f"  $$ {sp.latex(sp.Eq(target.initial.lhs, target.initial.rhs, evaluate=False))} $$")
    print("\nCanonical target:")
    print(f"  {target.target.lhs}  =  {target.target.rhs}")
    if target.initial.excluded:
        print(f"\nExcluded values: {set(target.initial.excluded)}")

    # ------- Hand heuristic run --------
    print("\n" + "=" * 72)
    print("Hand heuristic (WeightedSumCompositeHeuristic): tracing A*")
    print("=" * 72)
    hand = _TracingHeuristic(WeightedSumCompositeHeuristic())
    r_hand = astar(
        target.initial, target.is_target, heuristic=hand,
        max_nodes=args.max_nodes, max_depth=args.max_depth,
        problem_id=target.id,
    )
    print(f"\nsolved: {r_hand.found}")
    print(f"nodes_expanded:  {r_hand.stats.nodes_expanded}")
    print(f"nodes_generated: {r_hand.stats.nodes_generated}")
    print(f"path length:     {len(r_hand.path)}")
    print(f"heuristic calls: {len(hand.calls)}")

    # Show what hand finds attractive (the 10 lowest-h states it evaluated)
    print("\nHand's 10 most attractive states (lowest h, in order of discovery):")
    by_h = sorted(hand.calls, key=lambda c: (c.h, c.order))[:10]
    for c in by_h:
        print(f"  h={c.h:7.2f}  (eval #{c.order:4d})  {_short_eq(c.state, 90)}")

    # Show the optimal path A* eventually returned
    print("\nFinal solution path (after expanding 1000s of nodes):")
    print(f"  [start] {_short_eq(target.initial, 90)}")
    for i, (state, action) in enumerate(r_hand.path):
        rule_name = action.rule_name if hasattr(action, 'rule_name') else type(action).__name__
        print(f"  [step {i+1}] apply {rule_name}")
        print(f"           -> {_short_eq(state, 90)}")
    print(f"  [end]   {_short_eq(r_hand.final_state, 90) if r_hand.final_state else 'N/A'}")

    # H-values along the optimal path
    print("\nHand-h values along the optimal solution path:")
    path_states = [target.initial] + [s for s, _ in r_hand.path]
    raw_hand = WeightedSumCompositeHeuristic()
    for i, s in enumerate(path_states):
        h = raw_hand.evaluate(s)
        print(f"  step {i:2d}: hand-h={h:7.2f}   {_short_eq(s, 70)}")

    # ------- Learned heuristic run --------
    print("\n" + "=" * 72)
    print(f"Learned heuristic (GIN ckpt {args.ckpt}): tracing A*")
    print("=" * 72)
    learned_raw = LearnedHeuristic(args.ckpt, device=args.device)
    learned = _TracingHeuristic(learned_raw)
    r_learned = astar(
        target.initial, target.is_target, heuristic=learned,
        max_nodes=args.max_nodes, max_depth=args.max_depth,
        problem_id=target.id,
    )
    print(f"\nsolved: {r_learned.found}")
    print(f"nodes_expanded:  {r_learned.stats.nodes_expanded}")
    print(f"nodes_generated: {r_learned.stats.nodes_generated}")
    print(f"path length:     {len(r_learned.path)}")
    print(f"heuristic calls: {len(learned.calls)}")

    # H-values along learned's solution path
    print("\nLearned-h values along the optimal solution path (initial + each step):")
    path_states_l = [target.initial] + [s for s, _ in r_learned.path]
    for i, s in enumerate(path_states_l):
        h_l = learned_raw.evaluate(s)
        h_h = raw_hand.evaluate(s)
        print(f"  step {i:2d}: learned-h={h_l:6.3f}  hand-h={h_h:7.2f}   {_short_eq(s, 65)}")

    # Show the 10 lowest-h states learned saw (likely just the path + a few siblings)
    print("\nLearned's lowest-h states (first 10):")
    by_h_l = sorted(learned.calls, key=lambda c: (c.h, c.order))[:10]
    for c in by_h_l:
        print(f"  h={c.h:6.3f}  (eval #{c.order:4d})  {_short_eq(c.state, 90)}")

    print("\nLearned heuristic's solution path:")
    print(f"  [start] {_short_eq(target.initial, 90)}")
    for i, (state, action) in enumerate(r_learned.path):
        rule_name = action.rule_name if hasattr(action, 'rule_name') else type(action).__name__
        print(f"  [step {i+1}] apply {rule_name}")
        print(f"           -> {_short_eq(state, 90)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
