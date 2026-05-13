"""Polynomial-advanced rules: difference/sum of cubes.

Subtree-scoped (path) rules: fire on any subtree shaped a^3 +/- b^3.

FACTOR_POLYNOMIAL (polynomial.py) already factors top-level cubics like
`x^3 - 8 -> (x-2)(x^2+2x+4)`. These rules add two distinct values:

  1. Subtree firing on cubic sub-expressions in larger contexts like
     `(x^3-8)*(x+1) = 0` where FACTOR_POLYNOMIAL skips because the side
     is not a univariate polynomial in `var` of degree <= 4.
  2. Explicit identity label that the learned heuristic (Phase 2 GIN) can
     pattern-match directly.
"""

from __future__ import annotations

from typing import Iterator

import sympy as sp
from sympy import Add, Expr, Integer, Mul, Pow

from ...state import EqState
from ..base import Action, GuardResult
from ..registry import default_registry
from .algebra import DistributeOverSubtree, _replace_in_state, _walk_with_side


def _is_positive_cube(t: Expr) -> Expr | None:
    """Pow(a, 3) -> a, or positive Integer perfect cube k**3 -> Integer(k). Else None."""
    if isinstance(t, Pow) and t.args[1] == Integer(3):
        return t.args[0]
    if t.is_Integer and t > 0:
        v = int(t)
        r = round(v ** (1 / 3))
        for c in (r - 1, r, r + 1):
            if c > 0 and c ** 3 == v:
                return Integer(c)
    return None


def _is_negative_cube(t: Expr) -> Expr | None:
    """Return positive base if `t` is a structural negation of a positive cube.

    Recognizes: Mul(-1, X) where X is any positive cube (Pow(b,3) or k**3 Integer),
    OR a direct negative Integer that is a perfect cube (e.g., Integer(-8) -> Integer(2)).
    """
    # Case 1: Mul(-1, X) with X any positive cube. Covers both `-1 * b**3` and `-1 * 8`.
    if isinstance(t, Mul) and len(t.args) == 2:
        if any(f == Integer(-1) for f in t.args):
            other = next(f for f in t.args if f != Integer(-1))
            pos = _is_positive_cube(other)
            if pos is not None:
                return pos
    # Case 2: post-eval negative Integer
    if t.is_Integer and t < 0:
        v = -int(t)
        r = round(v ** (1 / 3))
        for c in (r - 1, r, r + 1):
            if c > 0 and c ** 3 == v:
                return Integer(c)
    return None


def _detect_diff_of_cubes(sub: Expr) -> tuple[Expr, Expr] | None:
    """If sub is `a^3 - b^3` (Add of 2 args, one positive cube + one negative cube),
    return (a, b). Else None."""
    if not isinstance(sub, Add) or len(sub.args) != 2:
        return None
    t0, t1 = sub.args
    # Try both orderings (Add is commutative; SymPy may yield either)
    for pos_t, neg_t in ((t0, t1), (t1, t0)):
        a = _is_positive_cube(pos_t)
        b = _is_negative_cube(neg_t)
        if a is not None and b is not None:
            return a, b
    return None


def _detect_sum_of_cubes(sub: Expr) -> tuple[Expr, Expr] | None:
    """If sub is `a^3 + b^3` (Add of 2 args, both positive cubes), return (a, b). Else None."""
    if not isinstance(sub, Add) or len(sub.args) != 2:
        return None
    bases = []
    for t in sub.args:
        r = _is_positive_cube(t)
        if r is None:
            return None
        bases.append(r)
    return bases[0], bases[1]


# ---------------------------------------------------------------------------
# 46. FACTOR_DIFFERENCE_OF_CUBES(path) — `a^3 - b^3` -> `(a-b)(a^2+ab+b^2)`
# ---------------------------------------------------------------------------


class FactorDifferenceOfCubes:
    name = "FACTOR_DIFFERENCE_OF_CUBES"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if _detect_diff_of_cubes(sub) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side, path = action.target_side, action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        result = _detect_diff_of_cubes(sub)
        if result is None:
            return state
        a, b = result
        # (a - b) * (a^2 + a*b + b^2)
        linear = Add(a, Mul(Integer(-1), b, evaluate=False), evaluate=False)
        a2 = Pow(a, Integer(2), evaluate=False)
        b2 = Pow(b, Integer(2), evaluate=False)
        ab = Mul(a, b, evaluate=False)
        trinom = Add(a2, ab, b2, evaluate=False)
        new_sub = Mul(linear, trinom, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(FactorDifferenceOfCubes())


# ---------------------------------------------------------------------------
# 47. FACTOR_SUM_OF_CUBES(path) — `a^3 + b^3` -> `(a+b)(a^2-ab+b^2)`
# ---------------------------------------------------------------------------


class FactorSumOfCubes:
    name = "FACTOR_SUM_OF_CUBES"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if _detect_sum_of_cubes(sub) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side, path = action.target_side, action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        result = _detect_sum_of_cubes(sub)
        if result is None:
            return state
        a, b = result
        # (a + b) * (a^2 - a*b + b^2)
        linear = Add(a, b, evaluate=False)
        a2 = Pow(a, Integer(2), evaluate=False)
        b2 = Pow(b, Integer(2), evaluate=False)
        neg_ab = Mul(Integer(-1), a, b, evaluate=False)
        trinom = Add(a2, neg_ab, b2, evaluate=False)
        new_sub = Mul(linear, trinom, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(FactorSumOfCubes())
