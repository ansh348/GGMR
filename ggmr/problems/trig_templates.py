"""Canonical-target seeds for `TrigReverseGenerator` (Phase 1.2b, Marcus).

Each seed represents a known trig identity in `lhs = rhs` form. The generator
applies inverse trig rules to the LHS (RHS fixed) to manufacture problems
whose forward BFS path simplifies the disguised LHS back to the canonical
RHS form.

v1: identity-verification only. `solve_equation` and `simplify` modes deferred.
"""

from __future__ import annotations

import random

import sympy as sp
from sympy import Add, Integer, Mul, Pow, Symbol, sin, cos, tan, csc, sec, cot

from ..state import EqState


def _x_y(var_name: str = "x") -> tuple[Symbol, Symbol]:
    return sp.Symbol(var_name), sp.Symbol("y")


def pyth_seed(rng: random.Random) -> EqState:
    """sin²(x) + cos²(x) = 1."""
    x, _ = _x_y()
    lhs = Add(Pow(sin(x), Integer(2)), Pow(cos(x), Integer(2)), evaluate=False)
    return EqState(lhs=lhs, rhs=Integer(1), var=x)


def quotient_seed(rng: random.Random) -> EqState:
    """tan(x) = sin(x) / cos(x)."""
    x, _ = _x_y()
    rhs = Mul(sin(x), Pow(cos(x), Integer(-1), evaluate=False), evaluate=False)
    return EqState(lhs=tan(x), rhs=rhs, var=x)


def double_sin_seed(rng: random.Random) -> EqState:
    """sin(2x) = 2 sin(x) cos(x)."""
    x, _ = _x_y()
    lhs = sin(Mul(Integer(2), x, evaluate=False))
    rhs = Mul(Integer(2), sin(x), cos(x), evaluate=False)
    return EqState(lhs=lhs, rhs=rhs, var=x)


def double_cos_seed(rng: random.Random) -> EqState:
    """cos(2x) = cos²(x) - sin²(x)."""
    x, _ = _x_y()
    lhs = cos(Mul(Integer(2), x, evaluate=False))
    rhs = Add(
        Pow(cos(x), Integer(2)),
        Mul(Integer(-1), Pow(sin(x), Integer(2)), evaluate=False),
        evaluate=False,
    )
    return EqState(lhs=lhs, rhs=rhs, var=x)


def sum_sin_seed(rng: random.Random) -> EqState:
    """sin(x + y) = sin(x)cos(y) + cos(x)sin(y)."""
    x, y = _x_y()
    lhs = sin(Add(x, y, evaluate=False))
    rhs = Add(Mul(sin(x), cos(y), evaluate=False), Mul(cos(x), sin(y), evaluate=False), evaluate=False)
    return EqState(lhs=lhs, rhs=rhs, var=x)


def sum_cos_seed(rng: random.Random) -> EqState:
    """cos(x + y) = cos(x)cos(y) - sin(x)sin(y)."""
    x, y = _x_y()
    lhs = cos(Add(x, y, evaluate=False))
    rhs = Add(
        Mul(cos(x), cos(y), evaluate=False),
        Mul(Integer(-1), sin(x), sin(y), evaluate=False),
        evaluate=False,
    )
    return EqState(lhs=lhs, rhs=rhs, var=x)


def reciprocal_csc_seed(rng: random.Random) -> EqState:
    """1/sin(x) = csc(x)."""
    x, _ = _x_y()
    lhs = Pow(sin(x), Integer(-1), evaluate=False)
    return EqState(lhs=lhs, rhs=csc(x), var=x)


def reciprocal_sec_seed(rng: random.Random) -> EqState:
    """1/cos(x) = sec(x)."""
    x, _ = _x_y()
    lhs = Pow(cos(x), Integer(-1), evaluate=False)
    return EqState(lhs=lhs, rhs=sec(x), var=x)


def parity_sin_seed(rng: random.Random) -> EqState:
    """sin(-x) = -sin(x). SymPy folds sin(-x) -> -sin(x) at parse time; build
    with evaluate=False to preserve the disguised form for inverse expansion."""
    x, _ = _x_y()
    neg_x = Mul(Integer(-1), x, evaluate=False)
    lhs = sin(neg_x, evaluate=False)
    rhs = Mul(Integer(-1), sin(x), evaluate=False)
    return EqState(lhs=lhs, rhs=rhs, var=x)


def parity_cos_seed(rng: random.Random) -> EqState:
    """cos(-x) = cos(x)."""
    x, _ = _x_y()
    neg_x = Mul(Integer(-1), x, evaluate=False)
    lhs = cos(neg_x, evaluate=False)
    return EqState(lhs=lhs, rhs=cos(x), var=x)


TRIG_TEMPLATES: dict[str, callable] = {
    "pyth":        pyth_seed,
    "quotient":    quotient_seed,
    "double_sin":  double_sin_seed,
    "double_cos":  double_cos_seed,
    "sum_sin":     sum_sin_seed,
    "sum_cos":     sum_cos_seed,
    "recip_csc":   reciprocal_csc_seed,
    "recip_sec":   reciprocal_sec_seed,
    "parity_sin":  parity_sin_seed,
    "parity_cos":  parity_cos_seed,
}


def trig_mixed_seed(rng: random.Random) -> EqState:
    """Random choice over all canonical trig seeds."""
    name = rng.choice(list(TRIG_TEMPLATES.keys()))
    return TRIG_TEMPLATES[name](rng)


TRIG_TEMPLATES["mixed"] = trig_mixed_seed
