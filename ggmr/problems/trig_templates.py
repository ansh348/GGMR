"""Canonical-target seeds for `TrigReverseGenerator` (Phase 1.2b, Marcus, v2).

v2 fixes the trig training-data duplication problem identified by external
review (95.6% dedup rate from v1):
  - Parameterized angle space via `random_angle(rng)`: every seed accepts a
    `u` argument sampled from 12 distinct compound-angle expressions
    (x, y, 2x, 3x, 4x, x/2, x+y, x-y, 2x+y, 2x-y, x+2y, 3x-2y).
  - Expanded from 10 to 42 seed families across 7 categories.
  - All structural construction uses `evaluate=False` so SymPy doesn't
    auto-canonicalize the seed back to a collapsed form before the BFS
    has a chance to traverse the disguised expression.

Each seed represents a known trig identity in `lhs = rhs` form. The
generator applies inverse trig rules to one side (typically LHS) to
manufacture problems whose forward BFS path simplifies the disguised
side back to the canonical form. Forward BFS termination predicate:
`canonical_repr(lhs) == canonical_repr(rhs)` (identity verification mode).
"""

from __future__ import annotations

import random
from typing import Callable

import sympy as sp
from sympy import Add, Integer, Mul, Pow, Rational, Symbol, cos, cot, csc, pi, sec, sin, tan

from ..state import EqState


# ---------------------------------------------------------------------------
# Construction helpers — keep evaluate=False to defeat SymPy auto-canon
# ---------------------------------------------------------------------------


def _I(n: int) -> Integer:
    return Integer(n)


def _add(*args):
    return Add(*args, evaluate=False)


def _mul(*args):
    return Mul(*args, evaluate=False)


def _neg(e):
    return _mul(_I(-1), e)


def _pow2(e):
    """e**2 with evaluate=False."""
    return Pow(e, _I(2), evaluate=False)


def _pow4(e):
    return Pow(e, _I(4), evaluate=False)


def _half():
    """The Rational 1/2 (sp folds it but the result is unambiguous)."""
    return Rational(1, 2)


def _div(a, b):
    """a / b as Mul(a, Pow(b, -1)) with evaluate=False to preserve structure."""
    return _mul(a, Pow(b, _I(-1), evaluate=False))


def _div_by_int(a, n: int):
    """a / n via Mul(a, Rational(1, n))."""
    return _mul(a, Rational(1, n))


# ---------------------------------------------------------------------------
# Angle grammar — the diversity engine
# ---------------------------------------------------------------------------


_X = Symbol("x")
_Y = Symbol("y")


def random_angle(rng: random.Random):
    """Sample one compound-angle expression. 12 distinct shapes.

    Returns a SymPy expression suitable as an argument to a trig function.
    All compound forms use `evaluate=False` so SymPy doesn't fold them
    before they reach the rule layer.
    """
    choices = [
        _X,
        _Y,
        _mul(_I(2), _X),
        _mul(_I(3), _X),
        _mul(_I(4), _X),
        _mul(_X, _half()),  # x/2
        _add(_X, _Y),
        _add(_X, _neg(_Y)),  # x - y
        _add(_mul(_I(2), _X), _Y),
        _add(_mul(_I(2), _X), _neg(_Y)),
        _add(_X, _mul(_I(2), _Y)),
        _add(_mul(_I(3), _X), _neg(_mul(_I(2), _Y))),
    ]
    return rng.choice(choices)


def _two_distinct_angles(rng: random.Random):
    """Sample u, v such that srepr(u) != srepr(v). Bounded retries; falls
    back to (x, y) if exhaustion."""
    u = random_angle(rng)
    for _ in range(8):
        v = random_angle(rng)
        if sp.srepr(v) != sp.srepr(u):
            return u, v
    return _X, _Y


# ---------------------------------------------------------------------------
# Pythagorean family (5)
# ---------------------------------------------------------------------------


def pyth_sin_cos_seed(rng):
    u = random_angle(rng)
    lhs = _add(_pow2(sin(u)), _pow2(cos(u)))
    return EqState(lhs=lhs, rhs=_I(1), var=_X)


def pyth_tan_sec_seed(rng):
    u = random_angle(rng)
    lhs = _add(_I(1), _pow2(tan(u)))
    rhs = _pow2(sec(u))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def pyth_sec_tan_seed(rng):
    u = random_angle(rng)
    lhs = _add(_pow2(sec(u)), _neg(_pow2(tan(u))))
    return EqState(lhs=lhs, rhs=_I(1), var=_X)


def pyth_cot_csc_seed(rng):
    u = random_angle(rng)
    lhs = _add(_I(1), _pow2(cot(u)))
    rhs = _pow2(csc(u))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def pyth_csc_cot_seed(rng):
    u = random_angle(rng)
    lhs = _add(_pow2(csc(u)), _neg(_pow2(cot(u))))
    return EqState(lhs=lhs, rhs=_I(1), var=_X)


# ---------------------------------------------------------------------------
# Quotient / reciprocal family (6)
# ---------------------------------------------------------------------------


def quot_tan_seed(rng):
    u = random_angle(rng)
    rhs = _div(sin(u), cos(u))
    return EqState(lhs=tan(u), rhs=rhs, var=_X)


def quot_cot_seed(rng):
    u = random_angle(rng)
    rhs = _div(cos(u), sin(u))
    return EqState(lhs=cot(u), rhs=rhs, var=_X)


def recip_sec_seed(rng):
    u = random_angle(rng)
    rhs = Pow(cos(u), _I(-1), evaluate=False)
    return EqState(lhs=sec(u), rhs=rhs, var=_X)


def recip_csc_seed(rng):
    u = random_angle(rng)
    rhs = Pow(sin(u), _I(-1), evaluate=False)
    return EqState(lhs=csc(u), rhs=rhs, var=_X)


def quot_tan_cos_seed(rng):
    """tan(u) * cos(u) = sin(u)."""
    u = random_angle(rng)
    lhs = _mul(tan(u), cos(u))
    return EqState(lhs=lhs, rhs=sin(u), var=_X)


def quot_cot_sin_seed(rng):
    """cot(u) * sin(u) = cos(u)."""
    u = random_angle(rng)
    lhs = _mul(cot(u), sin(u))
    return EqState(lhs=lhs, rhs=cos(u), var=_X)


# ---------------------------------------------------------------------------
# Double / half / power family (7)
# ---------------------------------------------------------------------------


def double_sin_seed(rng):
    """sin(2u) = 2 sin(u) cos(u)."""
    u = random_angle(rng)
    lhs = sin(_mul(_I(2), u))
    rhs = _mul(_I(2), sin(u), cos(u))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def double_cos_v1_seed(rng):
    """cos(2u) = cos²(u) - sin²(u)."""
    u = random_angle(rng)
    lhs = cos(_mul(_I(2), u))
    rhs = _add(_pow2(cos(u)), _neg(_pow2(sin(u))))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def double_cos_v2_seed(rng):
    """cos(2u) = 2 cos²(u) - 1."""
    u = random_angle(rng)
    lhs = cos(_mul(_I(2), u))
    rhs = _add(_mul(_I(2), _pow2(cos(u))), _I(-1))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def double_cos_v3_seed(rng):
    """cos(2u) = 1 - 2 sin²(u)."""
    u = random_angle(rng)
    lhs = cos(_mul(_I(2), u))
    rhs = _add(_I(1), _neg(_mul(_I(2), _pow2(sin(u)))))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def power_sin_seed(rng):
    """sin²(u) = (1 - cos(2u)) / 2."""
    u = random_angle(rng)
    lhs = _pow2(sin(u))
    rhs = _div_by_int(_add(_I(1), _neg(cos(_mul(_I(2), u)))), 2)
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def power_cos_seed(rng):
    """cos²(u) = (1 + cos(2u)) / 2."""
    u = random_angle(rng)
    lhs = _pow2(cos(u))
    rhs = _div_by_int(_add(_I(1), cos(_mul(_I(2), u))), 2)
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def prod_sin_cos_seed(rng):
    """sin(u) cos(u) = sin(2u) / 2."""
    u = random_angle(rng)
    lhs = _mul(sin(u), cos(u))
    rhs = _div_by_int(sin(_mul(_I(2), u)), 2)
    return EqState(lhs=lhs, rhs=rhs, var=_X)


# ---------------------------------------------------------------------------
# Sum / difference family (6)
# ---------------------------------------------------------------------------


def sum_sin_seed(rng):
    """sin(u+v) = sin(u)cos(v) + cos(u)sin(v)."""
    u, v = _two_distinct_angles(rng)
    lhs = sin(_add(u, v))
    rhs = _add(_mul(sin(u), cos(v)), _mul(cos(u), sin(v)))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def diff_sin_seed(rng):
    """sin(u-v) = sin(u)cos(v) - cos(u)sin(v)."""
    u, v = _two_distinct_angles(rng)
    lhs = sin(_add(u, _neg(v)))
    rhs = _add(_mul(sin(u), cos(v)), _neg(_mul(cos(u), sin(v))))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def sum_cos_seed(rng):
    """cos(u+v) = cos(u)cos(v) - sin(u)sin(v)."""
    u, v = _two_distinct_angles(rng)
    lhs = cos(_add(u, v))
    rhs = _add(_mul(cos(u), cos(v)), _neg(_mul(sin(u), sin(v))))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def diff_cos_seed(rng):
    """cos(u-v) = cos(u)cos(v) + sin(u)sin(v)."""
    u, v = _two_distinct_angles(rng)
    lhs = cos(_add(u, _neg(v)))
    rhs = _add(_mul(cos(u), cos(v)), _mul(sin(u), sin(v)))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def sum_tan_seed(rng):
    """tan(u+v) = (tan(u) + tan(v)) / (1 - tan(u)tan(v))."""
    u, v = _two_distinct_angles(rng)
    lhs = tan(_add(u, v))
    num = _add(tan(u), tan(v))
    den = _add(_I(1), _neg(_mul(tan(u), tan(v))))
    return EqState(lhs=lhs, rhs=_div(num, den), var=_X)


def diff_tan_seed(rng):
    """tan(u-v) = (tan(u) - tan(v)) / (1 + tan(u)tan(v))."""
    u, v = _two_distinct_angles(rng)
    lhs = tan(_add(u, _neg(v)))
    num = _add(tan(u), _neg(tan(v)))
    den = _add(_I(1), _mul(tan(u), tan(v)))
    return EqState(lhs=lhs, rhs=_div(num, den), var=_X)


# ---------------------------------------------------------------------------
# Product-to-sum / sum-to-product family (5)
# ---------------------------------------------------------------------------


def prod_sc_seed(rng):
    """2 sin(u) cos(v) = sin(u+v) + sin(u-v)."""
    u, v = _two_distinct_angles(rng)
    lhs = _mul(_I(2), sin(u), cos(v))
    rhs = _add(sin(_add(u, v)), sin(_add(u, _neg(v))))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def prod_cc_seed(rng):
    """2 cos(u) cos(v) = cos(u+v) + cos(u-v)."""
    u, v = _two_distinct_angles(rng)
    lhs = _mul(_I(2), cos(u), cos(v))
    rhs = _add(cos(_add(u, v)), cos(_add(u, _neg(v))))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def prod_ss_seed(rng):
    """2 sin(u) sin(v) = cos(u-v) - cos(u+v)."""
    u, v = _two_distinct_angles(rng)
    lhs = _mul(_I(2), sin(u), sin(v))
    rhs = _add(cos(_add(u, _neg(v))), _neg(cos(_add(u, v))))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def sum_to_prod_sin_seed(rng):
    """sin(u) + sin(v) = 2 sin((u+v)/2) cos((u-v)/2)."""
    u, v = _two_distinct_angles(rng)
    lhs = _add(sin(u), sin(v))
    half_sum = _div_by_int(_add(u, v), 2)
    half_diff = _div_by_int(_add(u, _neg(v)), 2)
    rhs = _mul(_I(2), sin(half_sum), cos(half_diff))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def sum_to_prod_cos_seed(rng):
    """cos(u) + cos(v) = 2 cos((u+v)/2) cos((u-v)/2)."""
    u, v = _two_distinct_angles(rng)
    lhs = _add(cos(u), cos(v))
    half_sum = _div_by_int(_add(u, v), 2)
    half_diff = _div_by_int(_add(u, _neg(v)), 2)
    rhs = _mul(_I(2), cos(half_sum), cos(half_diff))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


# ---------------------------------------------------------------------------
# Parity / cofunction family (5)
# ---------------------------------------------------------------------------


def parity_sin_seed(rng):
    """sin(-u) = -sin(u). SymPy folds sin(-x) at parse time; build with
    evaluate=False to preserve the disguised form for inverse expansion."""
    u = random_angle(rng)
    neg_u = _neg(u)
    lhs = sin(neg_u, evaluate=False)
    rhs = _neg(sin(u))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def parity_cos_seed(rng):
    """cos(-u) = cos(u)."""
    u = random_angle(rng)
    neg_u = _neg(u)
    lhs = cos(neg_u, evaluate=False)
    return EqState(lhs=lhs, rhs=cos(u), var=_X)


def parity_tan_seed(rng):
    """tan(-u) = -tan(u)."""
    u = random_angle(rng)
    neg_u = _neg(u)
    lhs = tan(neg_u, evaluate=False)
    rhs = _neg(tan(u))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def cof_sin_cos_seed(rng):
    """sin(pi/2 - u) = cos(u)."""
    u = random_angle(rng)
    arg = _add(_mul(_half(), pi), _neg(u))
    lhs = sin(arg, evaluate=False)
    return EqState(lhs=lhs, rhs=cos(u), var=_X)


def cof_cos_sin_seed(rng):
    """cos(pi/2 - u) = sin(u)."""
    u = random_angle(rng)
    arg = _add(_mul(_half(), pi), _neg(u))
    lhs = cos(arg, evaluate=False)
    return EqState(lhs=lhs, rhs=sin(u), var=_X)


# ---------------------------------------------------------------------------
# Hard textbook motifs (8)
# ---------------------------------------------------------------------------


def diff_squares_seed(rng):
    """cos⁴(u) - sin⁴(u) = cos(2u). (Difference of squares + Pyth.)"""
    u = random_angle(rng)
    lhs = _add(_pow4(cos(u)), _neg(_pow4(sin(u))))
    rhs = cos(_mul(_I(2), u))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def prod_factor_pyth_seed(rng):
    """sin(u+v) sin(u-v) = sin²(u) - sin²(v)."""
    u, v = _two_distinct_angles(rng)
    lhs = _mul(sin(_add(u, v)), sin(_add(u, _neg(v))))
    rhs = _add(_pow2(sin(u)), _neg(_pow2(sin(v))))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def half_angle_tan_seed(rng):
    """(1 - cos(2u)) / sin(2u) = tan(u)."""
    u = random_angle(rng)
    num = _add(_I(1), _neg(cos(_mul(_I(2), u))))
    den = sin(_mul(_I(2), u))
    lhs = _div(num, den)
    return EqState(lhs=lhs, rhs=tan(u), var=_X)


def pyth_ratio_seed(rng):
    """(1 + tan²(u)) / (1 + cot²(u)) = tan²(u)."""
    u = random_angle(rng)
    num = _add(_I(1), _pow2(tan(u)))
    den = _add(_I(1), _pow2(cot(u)))
    lhs = _div(num, den)
    return EqState(lhs=lhs, rhs=_pow2(tan(u)), var=_X)


def tan_cot_sum_seed(rng):
    """tan(u) + cot(u) = sec(u) csc(u)."""
    u = random_angle(rng)
    lhs = _add(tan(u), cot(u))
    rhs = _mul(sec(u), csc(u))
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def sec_minus_cos_seed(rng):
    """(sec(u) - cos(u)) / sin(u) = tan(u)."""
    u = random_angle(rng)
    num = _add(sec(u), _neg(cos(u)))
    lhs = _div(num, sin(u))
    return EqState(lhs=lhs, rhs=tan(u), var=_X)


def power_4_seed(rng):
    """sin²(u) cos²(u) = (1 - cos(4u)) / 8."""
    u = random_angle(rng)
    lhs = _mul(_pow2(sin(u)), _pow2(cos(u)))
    rhs = _div_by_int(_add(_I(1), _neg(cos(_mul(_I(4), u)))), 8)
    return EqState(lhs=lhs, rhs=rhs, var=_X)


def nested_pyth_seed(rng):
    """sin²(2u) + cos²(2u) = 1."""
    u = random_angle(rng)
    arg = _mul(_I(2), u)
    lhs = _add(_pow2(sin(arg)), _pow2(cos(arg)))
    return EqState(lhs=lhs, rhs=_I(1), var=_X)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


TRIG_TEMPLATES: dict[str, Callable[[random.Random], EqState]] = {
    # Pythagorean (5)
    "pyth_sin_cos": pyth_sin_cos_seed,
    "pyth_tan_sec": pyth_tan_sec_seed,
    "pyth_sec_tan": pyth_sec_tan_seed,
    "pyth_cot_csc": pyth_cot_csc_seed,
    "pyth_csc_cot": pyth_csc_cot_seed,
    # Quotient / reciprocal (6)
    "quot_tan":     quot_tan_seed,
    "quot_cot":     quot_cot_seed,
    "recip_sec":    recip_sec_seed,
    "recip_csc":    recip_csc_seed,
    "quot_tan_cos": quot_tan_cos_seed,
    "quot_cot_sin": quot_cot_sin_seed,
    # Double / half / power (7)
    "double_sin":   double_sin_seed,
    "double_cos_v1": double_cos_v1_seed,
    "double_cos_v2": double_cos_v2_seed,
    "double_cos_v3": double_cos_v3_seed,
    "power_sin":    power_sin_seed,
    "power_cos":    power_cos_seed,
    "prod_sin_cos": prod_sin_cos_seed,
    # Sum / difference (6)
    "sum_sin":      sum_sin_seed,
    "diff_sin":     diff_sin_seed,
    "sum_cos":      sum_cos_seed,
    "diff_cos":     diff_cos_seed,
    "sum_tan":      sum_tan_seed,
    "diff_tan":     diff_tan_seed,
    # Product-to-sum / sum-to-product (5)
    "prod_sc":      prod_sc_seed,
    "prod_cc":      prod_cc_seed,
    "prod_ss":      prod_ss_seed,
    "sum_to_prod_sin": sum_to_prod_sin_seed,
    "sum_to_prod_cos": sum_to_prod_cos_seed,
    # Parity / cofunction (5)
    "parity_sin":   parity_sin_seed,
    "parity_cos":   parity_cos_seed,
    "parity_tan":   parity_tan_seed,
    "cof_sin_cos":  cof_sin_cos_seed,
    "cof_cos_sin":  cof_cos_sin_seed,
    # Hard textbook motifs (8)
    "diff_squares":     diff_squares_seed,
    "prod_factor_pyth": prod_factor_pyth_seed,
    "half_angle_tan":   half_angle_tan_seed,
    "pyth_ratio":       pyth_ratio_seed,
    "tan_cot_sum":      tan_cot_sum_seed,
    "sec_minus_cos":    sec_minus_cos_seed,
    "power_4":          power_4_seed,
    "nested_pyth":      nested_pyth_seed,
}


def trig_mixed_seed(rng: random.Random) -> EqState:
    """Random choice over all concrete trig seeds (excluding 'mixed' itself)."""
    names = [k for k in TRIG_TEMPLATES.keys() if k != "mixed"]
    name = rng.choice(names)
    return TRIG_TEMPLATES[name](rng)


TRIG_TEMPLATES["mixed"] = trig_mixed_seed


# Backward-compatibility aliases for the original 10 v1 keys
TRIG_TEMPLATES["pyth"]       = pyth_sin_cos_seed
TRIG_TEMPLATES["quotient"]   = quot_tan_seed
TRIG_TEMPLATES["double_cos"] = double_cos_v1_seed
