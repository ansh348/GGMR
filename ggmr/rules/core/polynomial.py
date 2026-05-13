"""Polynomial rules: factor a univariate polynomial subtree (degree ≤ 4 in Phase 1a).

`FACTOR_POLYNOMIAL` is the workhorse for the Phase 0 quadratic and polynomial
problems: SymPy's `factor()` reaches `(x-1)*(x-2)*(x-3)` from `x^3 - 6x^2 + 11x - 6`
in one shot.
"""

from __future__ import annotations

from typing import Iterator

import sympy as sp
from sympy import Add, Expr, Integer, Mul, Pow

from ...expr.tree import canonical_repr
from ...soundness import safe_solve
from ...state import EqState
from ..base import Action, GuardResult
from ..registry import default_registry
from .algebra import DistributeOverSubtree, _replace_in_state, _walk_with_side


def _is_polynomial_in_var(expr: Expr, var: sp.Symbol) -> tuple[bool, int]:
    """Return (is_poly, degree). Returns (False, -1) if not a univariate polynomial in `var`."""
    try:
        poly = sp.Poly(sp.expand(expr), var)
    except sp.PolynomialError:
        return False, -1
    return True, poly.degree()


# ---------------------------------------------------------------------------
# 15. FACTOR_POLYNOMIAL(path) — factor a univariate polynomial subtree (degree 2..4)
# ---------------------------------------------------------------------------


class FactorPolynomial:
    name = "FACTOR_POLYNOMIAL"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if state.var not in sub.free_symbols:
                continue
            is_poly, deg = _is_polynomial_in_var(sub, state.var)
            if not is_poly or deg < 2 or deg > 4:
                continue
            # Only factor non-constant, non-trivially-factored polynomials
            factored = sp.factor(sub)
            if canonical_repr(factored) == canonical_repr(sub):
                continue
            # Skip trivial cases where factoring just rearranges (no new structure)
            if not isinstance(factored, (Mul, Pow)):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        factored = sp.factor(sub)
        return _replace_in_state(state, side, path, factored)


default_registry.register(FactorPolynomial())


# ===========================================================================
# Phase 1b additions (rules 38-42)
# ===========================================================================


# ---------------------------------------------------------------------------
# 38. POLYNOMIAL_LONG_DIVISION(divisor) — divide lhs (or rhs) polynomial by
#     divisor expression. Action's params is the divisor.
# ---------------------------------------------------------------------------


class PolynomialLongDivision:
    name = "POLYNOMIAL_LONG_DIVISION"
    arity = 1  # divisor

    def enumerate(self, state: EqState) -> Iterator[Action]:
        seen: set[tuple[str, str]] = set()
        for side in ("lhs", "rhs"):
            expr = state.lhs if side == "lhs" else state.rhs
            is_poly, deg = _is_polynomial_in_var(expr, state.var)
            if not is_poly or deg < 2:
                continue
            # Candidate divisors: linear factors of the form (x - r) for small r
            for r in range(-3, 4):
                divisor = state.var - sp.Integer(r)
                key = (side, canonical_repr(divisor))
                if key in seen:
                    continue
                seen.add(key)
                # Skip if divisor doesn't actually divide cleanly (long division
                # is still valid but yields a remainder; we keep it bounded)
                yield Action(self.name, params=(divisor,), target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        (divisor,) = action.params
        if sp.simplify(divisor) == 0:
            return GuardResult.failing("divisor simplifies to zero")
        # Roots of divisor become excluded if cancellation happens
        new_excluded: list[Expr] = []
        if state.var in divisor.free_symbols:
            new_excluded.extend(safe_solve(divisor, state.var))
        return GuardResult.passing(new_excluded=new_excluded)

    def apply(self, state: EqState, action: Action) -> EqState:
        (divisor,) = action.params
        side = action.target_side
        expr = state.lhs if side == "lhs" else state.rhs
        try:
            quot, rem = sp.div(expr, divisor, state.var)
        except Exception:
            return state
        # Rewrite as quot * divisor + rem (structural; verifier checks equivalence)
        new_expr = Add(Mul(quot, divisor, evaluate=False), rem, evaluate=False)
        if side == "lhs":
            return state.with_lhs_rhs(new_expr, state.rhs)
        return state.with_lhs_rhs(state.lhs, new_expr)


default_registry.register(PolynomialLongDivision())


# ---------------------------------------------------------------------------
# 39. SYNTHETIC_DIVISION(root) — special case of long division by (x - root).
#     Only fires when (x - root) cleanly divides the polynomial.
# ---------------------------------------------------------------------------


class SyntheticDivision:
    name = "SYNTHETIC_DIVISION"
    arity = 1  # root

    def enumerate(self, state: EqState) -> Iterator[Action]:
        seen: set[tuple[str, str]] = set()
        for side in ("lhs", "rhs"):
            expr = state.lhs if side == "lhs" else state.rhs
            is_poly, deg = _is_polynomial_in_var(expr, state.var)
            if not is_poly or deg < 2:
                continue
            # Try small integer roots
            for r in range(-5, 6):
                root_val = sp.Integer(r)
                # Check (x - r) divides expr cleanly
                try:
                    quot, rem = sp.div(expr, state.var - root_val, state.var)
                except Exception:
                    continue
                if sp.simplify(rem) != 0:
                    continue
                key = (side, canonical_repr(root_val))
                if key in seen:
                    continue
                seen.add(key)
                yield Action(self.name, params=(root_val,), target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        (root_val,) = action.params
        side = action.target_side
        expr = state.lhs if side == "lhs" else state.rhs
        try:
            quot, _ = sp.div(expr, state.var - root_val, state.var)
        except Exception:
            return state
        # Result: (x - root) * quot
        factored = Mul(
            Add(state.var, Mul(Integer(-1), root_val, evaluate=False), evaluate=False),
            quot,
            evaluate=False,
        )
        if side == "lhs":
            return state.with_lhs_rhs(factored, state.rhs)
        return state.with_lhs_rhs(state.lhs, factored)


default_registry.register(SyntheticDivision())


# ---------------------------------------------------------------------------
# 40. RATIONAL_ROOT_THEOREM — find a rational root of the polynomial side via
#     sp.solve, then factor it out (synthetic-division-like result).
# ---------------------------------------------------------------------------


class RationalRootTheorem:
    name = "RATIONAL_ROOT_THEOREM"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side in ("lhs", "rhs"):
            expr = state.lhs if side == "lhs" else state.rhs
            is_poly, deg = _is_polynomial_in_var(expr, state.var)
            if not is_poly or deg < 2:
                continue
            # Check if polynomial has at least one rational root
            try:
                roots = safe_solve(expr, state.var, rational=True)
            except Exception:
                continue
            rational_roots = [r for r in roots if r.is_rational]
            if not rational_roots:
                continue
            yield Action(self.name, params=(), target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        expr = state.lhs if side == "lhs" else state.rhs
        try:
            roots = safe_solve(expr, state.var, rational=True)
        except Exception:
            return state
        rational_roots = [r for r in roots if r.is_rational]
        if not rational_roots:
            return state
        # Pick the smallest absolute root for canonical determinism
        root_val = min(rational_roots, key=lambda r: (abs(r), r))
        try:
            quot, _ = sp.div(expr, state.var - root_val, state.var)
        except Exception:
            return state
        factored = Mul(
            Add(state.var, Mul(Integer(-1), root_val, evaluate=False), evaluate=False),
            quot,
            evaluate=False,
        )
        if side == "lhs":
            return state.with_lhs_rhs(factored, state.rhs)
        return state.with_lhs_rhs(state.lhs, factored)


default_registry.register(RationalRootTheorem())


# ---------------------------------------------------------------------------
# 41. VIETAS_FORMULAS — when polynomial is fully factored as product of linear
#     factors, record sum and product of roots as side conditions. The equation
#     itself is structurally unchanged (the rule's apply contract preserves
#     lhs/rhs); side_conditions documents the derived relationships.
# ---------------------------------------------------------------------------


def _is_fully_factored_linear_product(expr: Expr, var: sp.Symbol) -> list[Expr] | None:
    """Return list of roots if expr is product of (var - r_i). None otherwise."""
    if not isinstance(expr, Mul):
        return None
    roots: list[Expr] = []
    for f in expr.args:
        # Each factor must be (var - root) — an Add with [var, -root]
        if isinstance(f, Add) and len(f.args) == 2:
            if var in f.args[0].free_symbols and var not in f.args[1].free_symbols:
                # f = var + b → root = -b
                roots.append(-f.args[1])
                continue
            if var in f.args[1].free_symbols and var not in f.args[0].free_symbols:
                roots.append(-f.args[0])
                continue
        # Constant factor
        if var not in f.free_symbols:
            continue
        return None
    return roots if roots else None


class VietasFormulas:
    name = "VIETAS_FORMULAS"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side in ("lhs", "rhs"):
            expr = state.lhs if side == "lhs" else state.rhs
            roots = _is_fully_factored_linear_product(expr, state.var)
            if roots is None or len(roots) < 2:
                continue
            yield Action(self.name, params=(), target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        expr = state.lhs if side == "lhs" else state.rhs
        roots = _is_fully_factored_linear_product(expr, state.var)
        if roots is None:
            return state
        # Add side conditions: sum_of_roots = ..., prod_of_roots = ...
        sum_roots = sum(roots)
        prod_roots = sp.Integer(1)
        for r in roots:
            prod_roots = prod_roots * r
        cond_sum = sp.Eq(sp.Symbol("vieta_sum"), sum_roots)
        cond_prod = sp.Eq(sp.Symbol("vieta_prod"), prod_roots)
        return state.with_side_conditions(cond_sum, cond_prod)


default_registry.register(VietasFormulas())


# ---------------------------------------------------------------------------
# 42. POLY_TO_MONIC — divide both sides by leading coefficient of a polynomial
#     side. Specialization of DIVIDE_BOTH_SIDES_BY for the leading-coeff case.
# ---------------------------------------------------------------------------


class PolyToMonic:
    name = "POLY_TO_MONIC"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        # Fire when one side is a polynomial in var with leading coeff != 1
        for side in ("lhs", "rhs"):
            expr = state.lhs if side == "lhs" else state.rhs
            is_poly, deg = _is_polynomial_in_var(expr, state.var)
            if not is_poly or deg < 1:
                continue
            try:
                poly = sp.Poly(sp.expand(expr), state.var)
            except sp.PolynomialError:
                continue
            lead = poly.LC()
            if lead == 1 or lead == 0:
                continue
            yield Action(self.name, params=(), target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        side = action.target_side
        expr = state.lhs if side == "lhs" else state.rhs
        try:
            poly = sp.Poly(sp.expand(expr), state.var)
        except sp.PolynomialError:
            return GuardResult.failing("not a polynomial in var")
        lead = poly.LC()
        if lead == 0:
            return GuardResult.failing("leading coefficient is zero")
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        expr = state.lhs if side == "lhs" else state.rhs
        poly = sp.Poly(sp.expand(expr), state.var)
        lead = poly.LC()
        inv = Pow(lead, Integer(-1), evaluate=False)
        new_lhs = Mul(state.lhs, inv, evaluate=False)
        new_rhs = Mul(state.rhs, inv, evaluate=False)
        return state.with_lhs_rhs(new_lhs, new_rhs)


default_registry.register(PolyToMonic())
