"""Structural complexity features for Phase 0 monotonicity analysis.

Per PHASE0_PREREG.md §4, features are computed on `lhs - rhs` of each equation.
Composite is the z-scored sum of all four features across the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from sympy import Eq, Expr, Symbol


def tree_depth(expr: Expr) -> int:
    """Maximum AST depth: 1 for atoms, 1 + max(child depths) for internal nodes."""
    if not expr.args:
        return 1
    return 1 + max(tree_depth(a) for a in expr.args)


def op_count(expr: Expr) -> int:
    """Count of internal (non-atomic) AST nodes."""
    if not expr.args:
        return 0
    return 1 + sum(op_count(a) for a in expr.args)


def leaf_count(expr: Expr) -> int:
    """Count of atomic nodes (Symbol, Number)."""
    if not expr.args:
        return 1
    return sum(leaf_count(a) for a in expr.args)


def var_isolation_score(eq: Eq, var: Symbol) -> int:
    """Variable isolation: 0 means isolated (Symbol = constant); higher = less isolated.

    A perfectly-isolated equation is `var = constant` (or symmetric `constant = var`)
    where `var` does NOT appear on the other side. Otherwise, return the count of
    `var` occurrences across both sides.
    """
    lhs_only = (eq.lhs == var) and (var not in eq.rhs.free_symbols)
    rhs_only = (eq.rhs == var) and (var not in eq.lhs.free_symbols)
    if lhs_only or rhs_only:
        return 0
    count = 0
    for sub in sp.preorder_traversal(eq.lhs):
        if sub == var:
            count += 1
    for sub in sp.preorder_traversal(eq.rhs):
        if sub == var:
            count += 1
    return count


@dataclass(frozen=True)
class FeatureRow:
    depth: int
    ops: int
    leaves: int
    isolation: int

    def as_dict(self) -> dict:
        return {
            "depth": self.depth,
            "ops": self.ops,
            "leaves": self.leaves,
            "isolation": self.isolation,
        }


def features(eq: Eq, var: Symbol) -> FeatureRow:
    """Extract all 4 features from an equation. Composite is computed separately."""
    diff = eq.lhs - eq.rhs
    return FeatureRow(
        depth=tree_depth(diff),
        ops=op_count(diff),
        leaves=leaf_count(diff),
        isolation=var_isolation_score(eq, var),
    )


def composite_z(rows: list[FeatureRow]) -> list[float]:
    """Z-score each feature across the corpus, then sum to a single composite per row.

    Returns a list of composite scalars in the same order as `rows`.
    """
    if not rows:
        return []
    arr = np.array(
        [[r.depth, r.ops, r.leaves, r.isolation] for r in rows],
        dtype=float,
    )
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0)
    sd_safe = np.where(sd < 1e-9, 1.0, sd)
    z = (arr - mu) / sd_safe
    composite = z.sum(axis=1)
    return composite.tolist()
