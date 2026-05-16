"""Hand-coded trigonometric structural heuristic (Phase 1.2).

Foil baseline for trigonometric problems. The trained GIN must beat this on
compression ratios — the Marcus claim is that real strategy learning beats
target-blind structural counts.

Score components (smaller = closer to solved):
    - Number of distinct trig function calls in the subtree
    - Number of distinct trig arguments (separate calls with the same argument count once)
    - Number of Add terms on lhs/rhs
    - Tree depth
    - Count of `Pow(trig_fn, n)` with n != 1 (non-trivial trig powers)

Weights chosen so canonical targets (`x = c`, `0 = 0`) hit score 0 and
typical intermediate states score 3-15. Calibration is rough — the heuristic
is meant as a baseline, not an optimized solver.
"""

from __future__ import annotations

import sympy as sp
from sympy import Add, Expr, Pow
from sympy.functions.elementary.trigonometric import TrigonometricFunction

from ..expr.tree import canonical_repr, tree_depth
from ..heuristics.composite import Heuristic
from ..state import EqState


def _count_trig_atoms(expr: Expr) -> int:
    """Number of trig-function applications anywhere in the subtree."""
    return len(expr.atoms(TrigonometricFunction))


def _count_distinct_trig_args(expr: Expr) -> int:
    """Number of distinct argument expressions among trig function calls."""
    args = set()
    for atom in expr.atoms(TrigonometricFunction):
        if atom.args:
            args.add(canonical_repr(atom.args[0]))
    return len(args)


def _count_add_terms(expr: Expr) -> int:
    return len(expr.args) if isinstance(expr, Add) else 1


def _count_nontrivial_trig_powers(expr: Expr) -> int:
    """Count Pow(trig_fn, n) for n != 1 (typically sin², cos², etc.)."""
    cnt = 0
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, Pow) and isinstance(sub.args[0], TrigonometricFunction):
            if sub.args[1] != 1:
                cnt += 1
    return cnt


class TrigHandHeuristic:
    """Trig structural complexity baseline. Smaller = closer to a solved
    canonical state.

    Compatible with the `Heuristic` protocol used by A* / beam search.
    """

    name = "TRIG_HAND_V1"

    def __init__(
        self,
        w_atoms: float = 1.0,
        w_args: float = 0.5,
        w_terms: float = 0.3,
        w_depth: float = 0.2,
        w_powers: float = 0.8,
    ):
        self.w_atoms = w_atoms
        self.w_args = w_args
        self.w_terms = w_terms
        self.w_depth = w_depth
        self.w_powers = w_powers

    def evaluate(self, state: EqState) -> float:
        lhs, rhs = state.lhs, state.rhs
        atoms = _count_trig_atoms(lhs) + _count_trig_atoms(rhs)
        args = _count_distinct_trig_args(lhs) + _count_distinct_trig_args(rhs)
        terms = _count_add_terms(lhs) + _count_add_terms(rhs)
        depth = max(tree_depth(lhs), tree_depth(rhs))
        powers = _count_nontrivial_trig_powers(lhs) + _count_nontrivial_trig_powers(rhs)
        return (
            self.w_atoms * atoms
            + self.w_args * args
            + self.w_terms * terms
            + self.w_depth * depth
            + self.w_powers * powers
        )
