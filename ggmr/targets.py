"""Canonical-target detection per `ggmr_v10.pdf` §3.4.

Phase 1a covers the single-equation families:
  - Linear: x = k (constant)
  - Polynomial root set: x = k_1 OR x = k_2 OR ... (encoded as factored zero form)
  - Quadratic factored zero-form: (x - r_1)(x - r_2) = 0
  - Polynomial fully-factored zero-form

System conjunction (multi-equation) is out of scope for Phase 1a.
"""

from __future__ import annotations

import sympy as sp
from sympy import Add, Eq, Expr, Mul, Pow, Symbol


def _is_isolated_constant(side: Expr, var: Symbol) -> bool:
    """True iff `side` is a numeric/symbolic constant in `var` (var does not appear)."""
    return var not in side.free_symbols


def is_linear_target(lhs: Expr, rhs: Expr, var: Symbol) -> bool:
    """`x = k` (or symmetric `k = x`) where k is var-free."""
    if lhs == var and _is_isolated_constant(rhs, var):
        return True
    if rhs == var and _is_isolated_constant(lhs, var):
        return True
    return False


def _is_factored_zero_form(side: Expr, var: Symbol) -> bool:
    """A product of `(var - constant)` factors (Mul of `Add(var, -k)` or `Pow` of such).

    Coefficients in front (e.g., 2*(x-1)*(x-2)) are allowed.
    """
    if isinstance(side, Add) or isinstance(side, sp.Symbol):
        # A single `(x - k)` shows as Add; bare symbol `x` represents `(x - 0)`.
        return _is_linear_factor(side, var)
    if isinstance(side, Mul):
        for f in side.args:
            if not _is_linear_factor_or_pow_or_const(f, var):
                return False
        # require at least one factor that contains the variable
        return any(var in f.free_symbols for f in side.args)
    if isinstance(side, Pow):
        return _is_linear_factor_or_pow_or_const(side, var)
    return False


def _is_linear_factor(expr: Expr, var: Symbol) -> bool:
    """`(x - k)` style with leading coefficient exactly 1: bare `var`, or an
    Add whose only var-bearing arg is `var` itself, with constant remainder.

    `2x - 4` is NOT a linear factor here (leading coefficient 2). A scaled
    linear form should be expressed as a Mul of a constant and a `(x - k)`
    factor, which is recognized by the surrounding `_is_factored_zero_form`.
    """
    if expr == var:
        return True
    if not isinstance(expr, Add):
        return False
    var_terms = [a for a in expr.args if var in a.free_symbols]
    const_terms = [a for a in expr.args if var not in a.free_symbols]
    if len(var_terms) != 1:
        return False
    if var_terms[0] != var:
        return False
    return all(var not in c.free_symbols for c in const_terms)


def _is_linear_factor_or_pow_or_const(expr: Expr, var: Symbol) -> bool:
    """A linear factor `(x-k)`, a power of one `(x-k)^n`, or a var-free constant."""
    if var not in expr.free_symbols:
        return True
    if isinstance(expr, Pow):
        base, exp = expr.args
        if var not in exp.free_symbols and _is_linear_factor(base, var):
            return True
        return False
    return _is_linear_factor(expr, var)


def is_factored_zero_target(lhs: Expr, rhs: Expr, var: Symbol) -> bool:
    """`<factored polynomial> = 0` (or symmetric)."""
    if rhs == sp.Integer(0) and _is_factored_zero_form(lhs, var):
        return True
    if lhs == sp.Integer(0) and _is_factored_zero_form(rhs, var):
        return True
    return False


def is_root_set_target(lhs: Expr, rhs: Expr, var: Symbol) -> bool:
    """`x = constant` is a single-root target. Multi-root targets must be expressed
    as factored zero form per §3.4 — this function returns True only for the
    single-root case here, since Phase 1a's BFS produces single-equation states.
    """
    return is_linear_target(lhs, rhs, var)


def is_canonical_target(lhs: Expr, rhs: Expr, var: Symbol) -> bool:
    """True iff (lhs, rhs) is a canonical end state for `var`.

    Phase 1a recognizes the linear/single-root form `x = k` and the factored
    zero form `factor1 * factor2 * ... = 0` for explicit root enumeration.
    """
    if is_linear_target(lhs, rhs, var):
        return True
    if is_factored_zero_target(lhs, rhs, var):
        return True
    return False
