"""Equation state: immutable, hashable, with excluded values and side conditions.

Per `ggmr_v10.pdf` §3.1, search operates over typed expression-tree states.
Hashing is by AC-canonical structural fingerprint (`canonical_repr`), so
two states differing only by Add/Mul argument order dedup correctly in BFS.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import sympy as sp
from sympy import Eq, Expr, Symbol

from .expr.tree import canonical_repr
from .soundness import solution_set as compute_solution_set
from .targets import is_canonical_target


@dataclass(frozen=True)
class EqState:
    """An equation `lhs = rhs` in variable `var`, with guard-tracked side state.

    `excluded`:        values of `var` forbidden by a prior cancellation/division
    `side_conditions`: SymPy boolean predicates that must hold (e.g., x > 0)
    """

    lhs: Expr
    rhs: Expr
    var: Symbol = field(default_factory=lambda: sp.Symbol("x"))
    excluded: frozenset = frozenset()
    side_conditions: frozenset = frozenset()

    @classmethod
    def from_strings(cls, lhs: str, rhs: str, var_name: str = "x") -> "EqState":
        """Construct from `lhs`, `rhs` strings. Convenience for tests."""
        from sympy.parsing.sympy_parser import parse_expr

        var = sp.Symbol(var_name)
        local = {var_name: var}
        l = parse_expr(lhs, local_dict=local, evaluate=False)
        r = parse_expr(rhs, local_dict=local, evaluate=False)
        return cls(lhs=l, rhs=r, var=var)

    def to_eq(self) -> Eq:
        return Eq(self.lhs, self.rhs, evaluate=False)

    def is_canonical_target(self) -> bool:
        return is_canonical_target(self.lhs, self.rhs, self.var)

    def solution_set(self) -> frozenset:
        """Effective solution set: raw solve output minus values in `excluded`."""
        raw = compute_solution_set(self.lhs, self.rhs, self.var)
        if not self.excluded:
            return raw
        excluded_simplified = frozenset(sp.simplify(e) for e in self.excluded)
        return frozenset(s for s in raw if s not in excluded_simplified)

    def with_excluded(self, *values: Expr) -> "EqState":
        return replace(self, excluded=self.excluded | frozenset(values))

    def with_side_conditions(self, *conds) -> "EqState":
        return replace(self, side_conditions=self.side_conditions | frozenset(conds))

    def with_lhs_rhs(self, lhs: Expr, rhs: Expr) -> "EqState":
        return replace(self, lhs=lhs, rhs=rhs)

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(
            (
                canonical_repr(self.lhs),
                canonical_repr(self.rhs),
                self.var.name,
                frozenset(canonical_repr(e) for e in self.excluded),
                frozenset(map(str, self.side_conditions)),
            )
        )

    def __eq__(self, other) -> bool:  # type: ignore[override]
        if not isinstance(other, EqState):
            return NotImplemented
        return (
            self.var.name == other.var.name
            and canonical_repr(self.lhs) == canonical_repr(other.lhs)
            and canonical_repr(self.rhs) == canonical_repr(other.rhs)
            and frozenset(canonical_repr(e) for e in self.excluded)
            == frozenset(canonical_repr(e) for e in other.excluded)
            and frozenset(map(str, self.side_conditions))
            == frozenset(map(str, other.side_conditions))
        )

    def __repr__(self) -> str:
        return f"EqState({self.lhs} = {self.rhs}, var={self.var.name}, excluded={set(self.excluded)})"
