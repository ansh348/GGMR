"""Quadratic rules: complete the square, sqrt both sides, simplify-numeric-at.

`COMPLETE_THE_SQUARE` rewrites `ax² + bx + c` to `a(x + b/2a)² + (c − b²/4a)`,
applied to lhs only when lhs is a quadratic in `var` (univariate).

`SQRT_BOTH_SIDES` applies when one side is a perfect-square form `(linear)^2`
and the other side is non-negative; produces `linear = ±sqrt(other_side)`.

`SIMPLIFY_NUMERIC_AT` collapses pure-numeric arithmetic in a subtree (e.g.,
`Add(-6, 4) → -2`). Used as the 15th rule to clean up after operations like
`COMBINE_LIKE_TERMS_AT` for symbolic simplification.
"""

from __future__ import annotations

from typing import Iterator

import sympy as sp
from sympy import Add, Expr, Integer, Mul, Pow

from ...expr.tree import canonical_repr, iter_subtrees, normalize
from ...state import EqState
from ..base import Action, GuardResult
from ..registry import default_registry
from .algebra import DistributeOverSubtree, _replace_in_state, _walk_with_side


# ---------------------------------------------------------------------------
# 12. COMPLETE_THE_SQUARE — apply to lhs (or rhs) when it is `a*var^2 + b*var + c`
# ---------------------------------------------------------------------------


def _quadratic_coeffs(expr: Expr, var: sp.Symbol) -> tuple[Expr, Expr, Expr] | None:
    """Return (a, b, c) if `expr` is a quadratic polynomial in `var`, else None."""
    try:
        poly = sp.Poly(sp.expand(expr), var)
    except sp.PolynomialError:
        return None
    if poly.degree() != 2:
        return None
    a = poly.coeff_monomial(var**2)
    b = poly.coeff_monomial(var)
    c = poly.coeff_monomial(1)
    return a, b, c


class CompleteTheSquare:
    name = "COMPLETE_THE_SQUARE"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side in ("lhs", "rhs"):
            expr = state.lhs if side == "lhs" else state.rhs
            coeffs = _quadratic_coeffs(expr, state.var)
            if coeffs is None:
                continue
            yield Action(self.name, params=(), target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        expr = state.lhs if side == "lhs" else state.rhs
        coeffs = _quadratic_coeffs(expr, state.var)
        if coeffs is None:
            return state
        a, b, c = coeffs
        h = b / (2 * a)  # canonical (will be evaluated by sympy)
        k = c - b**2 / (4 * a)
        # Build a*(x + h)^2 + k structurally
        inner = Add(state.var, h, evaluate=False)
        sq = Pow(inner, Integer(2), evaluate=False)
        scaled = Mul(a, sq, evaluate=False) if a != Integer(1) else sq
        new_expr = Add(scaled, k, evaluate=False) if k != Integer(0) else scaled
        if side == "lhs":
            return state.with_lhs_rhs(new_expr, state.rhs)
        return state.with_lhs_rhs(state.lhs, new_expr)


default_registry.register(CompleteTheSquare())


# ---------------------------------------------------------------------------
# 13. SQRT_BOTH_SIDES — when one side is `(linear)^2`, take sqrt with ± choice.
#
# Phase 1a only emits the `+` (principal) branch to avoid multiplying state
# count. The `-` branch is a Phase 1b extension; for the Phase 0 test set, qua04
# stops at `(x-2)^2 = 3` which is already a recognized canonical-target shape
# and does NOT need to go further.
# ---------------------------------------------------------------------------


class SqrtBothSides:
    name = "SQRT_BOTH_SIDES"
    arity = 1  # sign choice: +1 or -1 (Phase 1a emits +1 only)

    def enumerate(self, state: EqState) -> Iterator[Action]:
        # Only enumerate when one side is a Pow(_, 2) and the other is a constant
        # whose sign we can verify non-negative.
        for side in ("lhs", "rhs"):
            sq_side = state.lhs if side == "lhs" else state.rhs
            other_side = state.rhs if side == "lhs" else state.lhs
            if not (isinstance(sq_side, Pow) and sq_side.args[1] == Integer(2)):
                continue
            try:
                if other_side.is_real is True and other_side.is_nonnegative is True:
                    pass
                else:
                    continue
            except Exception:
                continue
            yield Action(self.name, params=(Integer(1),), target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        (sign,) = action.params
        if side == "lhs":
            base = state.lhs.args[0]
            new_rhs = Mul(sign, sp.sqrt(state.rhs), evaluate=False)
            return state.with_lhs_rhs(base, new_rhs)
        base = state.rhs.args[0]
        new_lhs = Mul(sign, sp.sqrt(state.lhs), evaluate=False)
        return state.with_lhs_rhs(new_lhs, base)


default_registry.register(SqrtBothSides())


# ---------------------------------------------------------------------------
# 14. SIMPLIFY_NUMERIC_AT(path) — collapse pure-numeric arithmetic in a subtree.
# ---------------------------------------------------------------------------


def _is_pure_numeric(expr: Expr) -> bool:
    return all(getattr(a, "is_number", False) for a in expr.args) and bool(expr.args)


class SimplifyNumericAt:
    name = "SIMPLIFY_NUMERIC_AT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not sub.args:
                continue
            if not _is_pure_numeric(sub):
                continue
            simplified = normalize(sub)
            # Use sp.srepr (not canonical_repr — that would re-normalize the input)
            if sp.srepr(simplified) == sp.srepr(sub):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        return _replace_in_state(state, side, path, normalize(sub))


default_registry.register(SimplifyNumericAt())


# ===========================================================================
# Phase 1b additions (rules 34-37)
# ===========================================================================


# ---------------------------------------------------------------------------
# 34. QUADRATIC_FORMULA — principal (+) branch only.
#
# When lhs is `a*var² + b*var + c` and rhs == 0, emit `var = (-b + √(b²-4ac))/(2a)`.
# The (-) branch is deferred to Phase 1c (would require multi-successor support).
# ---------------------------------------------------------------------------


class QuadraticFormula:
    name = "QUADRATIC_FORMULA"
    arity = 0  # principal branch only

    def enumerate(self, state: EqState) -> Iterator[Action]:
        # Fire only when one side is quadratic in var and the other side is zero.
        for side in ("lhs", "rhs"):
            quad = state.lhs if side == "lhs" else state.rhs
            other = state.rhs if side == "lhs" else state.lhs
            if other != Integer(0):
                continue
            coeffs = _quadratic_coeffs(quad, state.var)
            if coeffs is None:
                continue
            yield Action(self.name, params=(), target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        side = action.target_side
        quad = state.lhs if side == "lhs" else state.rhs
        coeffs = _quadratic_coeffs(quad, state.var)
        if coeffs is None:
            return GuardResult.failing("not a quadratic in var")
        a, _, _ = coeffs
        if sp.simplify(a) == 0:
            return GuardResult.failing("leading coefficient is zero")
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        quad = state.lhs if side == "lhs" else state.rhs
        coeffs = _quadratic_coeffs(quad, state.var)
        if coeffs is None:
            return state
        a, b, c = coeffs
        disc = b**2 - 4 * a * c
        # Principal (+) branch: root = (-b + sqrt(disc)) / (2a)
        root_plus = (-b + sp.sqrt(disc)) / (2 * a)
        return state.with_lhs_rhs(state.var, root_plus)


default_registry.register(QuadraticFormula())


# ---------------------------------------------------------------------------
# 35. FACTOR_BY_GROUPING — for quadratic `a*x² + b*x + c` with integer coeffs,
#     find m,n such that m+n = b and m*n = a*c, then split bx into mx+nx and group.
# ---------------------------------------------------------------------------


def _find_grouping_pair(a: int, b: int, c: int) -> tuple[int, int] | None:
    """Find integers m, n with m + n = b and m * n = a * c. Returns first match."""
    target = a * c
    for m in range(-abs(target) - 1, abs(target) + 2):
        if m == 0:
            continue
        if target % m != 0:
            continue
        n = target // m
        if m + n == b:
            return m, n
    return None


class FactorByGrouping:
    name = "FACTOR_BY_GROUPING"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side in ("lhs", "rhs"):
            quad = state.lhs if side == "lhs" else state.rhs
            coeffs = _quadratic_coeffs(quad, state.var)
            if coeffs is None:
                continue
            a, b, c = coeffs
            if not (a.is_Integer and b.is_Integer and c.is_Integer):
                continue
            ai, bi, ci = int(a), int(b), int(c)
            if _find_grouping_pair(ai, bi, ci) is None:
                continue
            yield Action(self.name, params=(), target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        side = action.target_side
        quad = state.lhs if side == "lhs" else state.rhs
        coeffs = _quadratic_coeffs(quad, state.var)
        if coeffs is None:
            return GuardResult.failing("not a quadratic in var")
        a, b, c = coeffs
        if not (a.is_Integer and b.is_Integer and c.is_Integer):
            return GuardResult.failing("non-integer coefficients")
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        quad = state.lhs if side == "lhs" else state.rhs
        coeffs = _quadratic_coeffs(quad, state.var)
        if coeffs is None:
            return state
        a, b, c = coeffs
        ai, bi, ci = int(a), int(b), int(c)
        pair = _find_grouping_pair(ai, bi, ci)
        if pair is None:
            return state
        # SymPy.factor produces the canonical factored form; we use it here for
        # the result (simplifies the apply contract — the structure is what matters).
        new_quad = sp.factor(sp.Integer(ai) * state.var**2 + sp.Integer(bi) * state.var + sp.Integer(ci))
        if side == "lhs":
            return state.with_lhs_rhs(new_quad, state.rhs)
        return state.with_lhs_rhs(state.lhs, new_quad)


default_registry.register(FactorByGrouping())


# ---------------------------------------------------------------------------
# 36. FACTOR_DIFFERENCE_OF_SQUARES_AT(path) — `a² - b²` → `(a-b)*(a+b)`
# ---------------------------------------------------------------------------


def _detect_diff_of_squares(sub: Expr) -> tuple[Expr, Expr] | None:
    """If sub is structurally `Add(Pow(a, 2), -Pow(b, 2))`, return (a, b). Else None."""
    if not isinstance(sub, Add) or len(sub.args) != 2:
        return None
    pos_term, neg_term = None, None
    for t in sub.args:
        if isinstance(t, Pow) and t.args[1] == Integer(2):
            pos_term = t.args[0]
            continue
        # neg_term: Mul(-1, Pow(b, 2)) or Mul(-1, b**2 expanded)
        if isinstance(t, Mul) and len(t.args) == 2:
            neg_one, other = None, None
            for f in t.args:
                if f == Integer(-1):
                    neg_one = f
                else:
                    other = f
            if neg_one is not None and isinstance(other, Pow) and other.args[1] == Integer(2):
                neg_term = other.args[0]
                continue
        # Also accept: Integer < 0 directly (e.g., -4)
        if t.is_number and t.is_negative:
            sqrt_val = sp.sqrt(-t)
            if sqrt_val.is_number and sqrt_val.is_Integer:
                neg_term = sqrt_val
                continue
    if pos_term is not None and neg_term is not None:
        return pos_term, neg_term
    return None


class FactorDifferenceOfSquaresAt:
    name = "FACTOR_DIFFERENCE_OF_SQUARES_AT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if _detect_diff_of_squares(sub) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        result = _detect_diff_of_squares(sub)
        if result is None:
            return state
        a, b = result
        # (a - b) * (a + b)
        diff = Add(a, Mul(Integer(-1), b, evaluate=False), evaluate=False)
        summ = Add(a, b, evaluate=False)
        new_sub = Mul(diff, summ, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(FactorDifferenceOfSquaresAt())


# ---------------------------------------------------------------------------
# 37. PERFECT_SQUARE_TRINOMIAL_AT(path) — a² + 2ab + b² → (a+b)²; a² - 2ab + b² → (a-b)²
#
# Detection: cubic over the symbolic-polynomial fingerprint of the Add. Use
# sp.factor and check whether the result is Pow(_, 2).
# ---------------------------------------------------------------------------


class PerfectSquareTrinomialAt:
    name = "PERFECT_SQUARE_TRINOMIAL_AT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add) or len(sub.args) != 3:
                continue
            try:
                factored = sp.factor(sub)
            except Exception:
                continue
            # Must be (something)**2
            if not (isinstance(factored, Pow) and factored.args[1] == Integer(2)):
                continue
            if canonical_repr(factored) == canonical_repr(sub):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        try:
            factored = sp.factor(sub)
        except Exception:
            return state
        if not (isinstance(factored, Pow) and factored.args[1] == Integer(2)):
            return state
        return _replace_in_state(state, side, path, factored)


default_registry.register(PerfectSquareTrinomialAt())
