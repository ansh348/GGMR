"""Exponent rules: laws of exponents on Pow subtrees.

These rules don't fire on the Phase 0 problem set (no exponent-manipulation
problems beyond the integer powers handled by `EXPAND_POWER`) but round the
rule library toward textbook completeness and demonstrate the architecture
handles exponent manipulation.

- `POW_PRODUCT_AT(path)`: `a^m * a^n` → `a^(m+n)` when both factors share a base.
- `POW_QUOTIENT_AT(path)`: `a^m * a^(-n)` → `a^(m-n)` (requires `a ≠ 0`).
- `POW_OF_POW_AT(path)`: `(a^m)^n` → `a^(m*n)`.
"""

from __future__ import annotations

from typing import Iterator

import sympy as sp
from sympy import Add, Expr, Integer, Mul, Pow

from ...expr.tree import canonical_repr
from ...state import EqState
from ..base import Action, GuardResult
from ..registry import default_registry
from .algebra import DistributeOverSubtree, _replace_in_state, _walk_with_side


def _split_pow(expr: Expr) -> tuple[Expr, Expr] | None:
    """If expr is Pow(b, e), return (b, e). Else if expr is an atom, return (expr, 1)."""
    if isinstance(expr, Pow):
        return expr.args[0], expr.args[1]
    if not expr.args:
        return expr, Integer(1)
    return None


# ---------------------------------------------------------------------------
# 43. POW_PRODUCT_AT(path) — Mul of two Pow factors with matching bases
#     a^m * a^n → a^(m+n)
# ---------------------------------------------------------------------------


class PowProductAt:
    name = "POW_PRODUCT_AT"
    arity = 1  # path

    @staticmethod
    def _find_matching_pair(args: tuple[Expr, ...]) -> tuple[int, int, Expr] | None:
        """Find indices (i, j) in args of two factors with the same base, both
        Pow with non-negative exponent (excludes inverse-factor case which is
        handled by POW_QUOTIENT_AT). Returns (i, j, base).
        """
        bases: list[tuple[int, Expr, Expr]] = []
        for idx, a in enumerate(args):
            sp_pow = _split_pow(a)
            if sp_pow is None:
                continue
            base, exp = sp_pow
            # Skip negative-exponent: that's POW_QUOTIENT's domain
            if isinstance(exp, Integer) and int(exp) < 0:
                continue
            bases.append((idx, base, exp))
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                if canonical_repr(bases[i][1]) == canonical_repr(bases[j][1]):
                    return bases[i][0], bases[j][0], bases[i][1]
        return None

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            if self._find_matching_pair(sub.args) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Mul):
            return state
        match = self._find_matching_pair(sub.args)
        if match is None:
            return state
        i, j, base = match
        # Get exponents
        ei = _split_pow(sub.args[i])[1]
        ej = _split_pow(sub.args[j])[1]
        new_exp = Add(ei, ej, evaluate=False)
        new_pow = Pow(base, new_exp, evaluate=False)
        new_args = [a for k, a in enumerate(sub.args) if k != i and k != j]
        new_args.append(new_pow)
        if len(new_args) == 1:
            new_sub: Expr = new_args[0]
        else:
            new_sub = Mul(*new_args, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(PowProductAt())


# ---------------------------------------------------------------------------
# 44. POW_QUOTIENT_AT(path) — a^m * a^(-n) → a^(m-n). Requires a != 0.
# ---------------------------------------------------------------------------


class PowQuotientAt:
    name = "POW_QUOTIENT_AT"
    arity = 1  # path

    @staticmethod
    def _find_quotient_pair(args: tuple[Expr, ...]) -> tuple[int, int, Expr] | None:
        """Find (i, j, base) where args[i] is a^m, args[j] is a^(-n) for some n>0."""
        candidates: list[tuple[int, Expr, Expr]] = []
        for idx, a in enumerate(args):
            sp_pow = _split_pow(a)
            if sp_pow is None:
                continue
            base, exp = sp_pow
            candidates.append((idx, base, exp))
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i == j:
                    continue
                bi, ei = candidates[i][1], candidates[i][2]
                bj, ej = candidates[j][1], candidates[j][2]
                if canonical_repr(bi) != canonical_repr(bj):
                    continue
                # i is positive-exp, j is negative-exp
                if not (isinstance(ej, Integer) and int(ej) < 0):
                    continue
                return candidates[i][0], candidates[j][0], bi
        return None

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            if self._find_quotient_pair(sub.args) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Mul):
            return GuardResult.failing("not a Mul")
        match = self._find_quotient_pair(sub.args)
        if match is None:
            return GuardResult.failing("no matching quotient pair")
        _, _, base = match
        # Base must be non-zero
        if sp.simplify(base) == 0:
            return GuardResult.failing("base simplifies to zero")
        new_excluded: list[Expr] = []
        if state.var in base.free_symbols:
            new_excluded.extend(sp.solve(base, state.var))
        return GuardResult.passing(new_excluded=new_excluded)

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Mul):
            return state
        match = self._find_quotient_pair(sub.args)
        if match is None:
            return state
        i, j, base = match
        ei = _split_pow(sub.args[i])[1]
        ej = _split_pow(sub.args[j])[1]
        new_exp = Add(ei, ej, evaluate=False)
        new_pow = Pow(base, new_exp, evaluate=False)
        new_args = [a for k, a in enumerate(sub.args) if k != i and k != j]
        new_args.append(new_pow)
        if len(new_args) == 1:
            new_sub: Expr = new_args[0]
        else:
            new_sub = Mul(*new_args, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(PowQuotientAt())


# ---------------------------------------------------------------------------
# 45. POW_OF_POW_AT(path) — (a^m)^n → a^(m*n)
# ---------------------------------------------------------------------------


class PowOfPowAt:
    name = "POW_OF_POW_AT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Pow):
                continue
            base, _ = sub.args
            if not isinstance(base, Pow):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Pow):
            return state
        outer_base, outer_exp = sub.args
        if not isinstance(outer_base, Pow):
            return state
        inner_base, inner_exp = outer_base.args
        new_exp = Mul(inner_exp, outer_exp, evaluate=False)
        new_sub = Pow(inner_base, new_exp, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(PowOfPowAt())
