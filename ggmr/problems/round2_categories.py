"""Round 2 training data: 35 category generators for broad algebra coverage.

Each generator returns a `MotifInstance` (eq_state, target_eq_state, ...) which
the worker BFS-solves to extract (state, remaining_steps) pairs along the path.

Tiers:
  Trivial   (1-4):  direct sample; 0-2 step paths
  Easy      (5-13): direct construction from solution; 1-5 steps
  Medium    (14-23, 31-35): require factoring/expansion machinery; 3-8 steps
  Adversarial (24-30): reuse motif_templates via sample_motif_instance; 4-15 steps

Why broad: Round 1 had narrow coverage (70% reverse-easy + 30% hard motifs) and
heavily skewed remaining_steps (no states with 3+ steps to target). The GIN
learned "rationals → heavy machinery" and over-engineered easy problems
(0.755x OOD geomean). Round 2 spans the full difficulty spectrum with
contrastive pairs (e.g., cat 8 vs cat 30, cat 4 vs cat 14).
"""
from __future__ import annotations

import random
from typing import Callable

import sympy as sp
from sympy import Symbol, Integer, Rational, Add, Mul, Pow

from ggmr.problems.motif_templates import MotifInstance, A, M, P, I
from ggmr.state import EqState


VAR = sp.Symbol("x")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signed(rng: random.Random, max_abs: int) -> int:
    """Random integer in [-max_abs, max_abs] excluding 0."""
    while True:
        v = rng.randint(-max_abs, max_abs)
        if v != 0:
            return v


def _distinct_signed(rng: random.Random, n: int, max_abs: int) -> list[int]:
    """n distinct nonzero integers in [-max_abs, max_abs]."""
    pool = [v for v in range(-max_abs, max_abs + 1) if v != 0]
    if len(pool) < n:
        raise ValueError("pool too small")
    return rng.sample(pool, n)


def _linear_factor(x: Symbol, r: int):
    """(x - r) in evaluate=False form."""
    if r == 0:
        return x
    return A(x, I(-r))


def _factored_zero(x: Symbol, roots: list[int]) -> Mul:
    """Build M((x-r1), (x-r2), ...) for the canonical factored-zero target."""
    factors = [_linear_factor(x, r) for r in roots]
    if len(factors) == 1:
        return factors[0]
    return M(*factors)


def _linear_target(x: Symbol, k) -> EqState:
    """Canonical `x = k` target."""
    return EqState(lhs=x, rhs=sp.sympify(k), var=x)


def _multi_root_target(x: Symbol, roots: list[int]) -> EqState:
    """Canonical factored-zero target for 2+ distinct integer roots."""
    return EqState(lhs=_factored_zero(x, sorted(roots)), rhs=I(0), var=x)


# ===========================================================================
# Category 1: Canonical / near-canonical target recognition (trivial, 0-2 steps)
# ===========================================================================

def gen_cat_01(rng: random.Random, depth: int = 0) -> MotifInstance:
    """x = k, k = x, x - k = 0, c*(x-k) = 0."""
    x = VAR
    k = _signed(rng, 20)
    variant = rng.choice([0, 1, 2, 3])
    if variant == 0:
        initial = EqState(lhs=x, rhs=I(k), var=x)
    elif variant == 1:
        initial = EqState(lhs=I(k), rhs=x, var=x)
    elif variant == 2:
        initial = EqState(lhs=A(x, I(-k)), rhs=I(0), var=x)
    else:
        c = _signed(rng, 5)
        initial = EqState(lhs=M(I(c), A(x, I(-k))), rhs=I(0), var=x)
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, k),
        category="linear", motif_family="cat_01_canonical_target",
        params={"k": k, "variant": variant},
    )


# ===========================================================================
# Category 2: Direct affine isolation `a*x + b = c` (trivial, 1-3 steps)
# ===========================================================================

def gen_cat_02(rng: random.Random, depth: int = 0) -> MotifInstance:
    """a*x + b = c, target = (c-b)/a (integer)."""
    x = VAR
    a = _signed(rng, 10)
    target_val = _signed(rng, 12)
    b = rng.randint(-15, 15)
    c = a * target_val + b
    if b == 0:
        lhs = M(I(a), x)
    else:
        lhs = A(M(I(a), x), I(b))
    initial = EqState(lhs=lhs, rhs=I(c), var=x)
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="linear", motif_family="cat_02_affine_isolation",
        params={"a": a, "b": b, "c": c, "target": target_val},
    )


# ===========================================================================
# Category 3: Already-factored / difference of squares (trivial, 0-3 steps)
# ===========================================================================

def gen_cat_03(rng: random.Random, depth: int = 0) -> MotifInstance:
    """(x-r1)(x-r2) = 0, x²-k²=0, or (x+a)² - k² = 0."""
    x = VAR
    variant = rng.choice([0, 1, 2])
    if variant == 0:
        r1, r2 = _distinct_signed(rng, 2, 8)
        lhs = M(_linear_factor(x, r1), _linear_factor(x, r2))
        initial = EqState(lhs=lhs, rhs=I(0), var=x)
        target = _multi_root_target(x, [r1, r2])
        params = {"r1": r1, "r2": r2}
    elif variant == 1:
        k = rng.randint(1, 8)
        lhs = A(P(x, I(2)), I(-(k * k)))
        initial = EqState(lhs=lhs, rhs=I(0), var=x)
        target = _multi_root_target(x, [-k, k])
        params = {"k": k}
    else:
        a = _signed(rng, 5)
        k = rng.randint(1, 5)
        lhs = A(P(A(x, I(a)), I(2)), I(-(k * k)))
        initial = EqState(lhs=lhs, rhs=I(0), var=x)
        r1, r2 = -a + k, -a - k
        target = _multi_root_target(x, [r1, r2])
        params = {"a": a, "k": k}
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_03_already_factored",
        params=params,
    )


# ===========================================================================
# Category 4: Direct square-root / perfect-square = constant (1-3 steps)
# ===========================================================================

def gen_cat_04(rng: random.Random, depth: int = 0) -> MotifInstance:
    """(a*x + b)² = c² with c >= 0; target = positive root."""
    x = VAR
    a = rng.choice([1, 1, 1, 2, 2, 3])
    b = rng.randint(-8, 8)
    c = rng.randint(1, 9)
    # solutions: a*x + b = ±c → x = (-b ± c) / a
    # pick the integer root
    candidates = []
    for sign in (1, -1):
        num = sign * c - b
        if num % a == 0:
            candidates.append(num // a)
    if not candidates:
        # retry with a=1 to guarantee integer roots
        a = 1
        candidates = [c - b, -c - b]
    target_val = rng.choice(candidates)
    inner = A(M(I(a), x), I(b)) if b != 0 else M(I(a), x)
    if a == 1:
        inner = A(x, I(b)) if b != 0 else x
    lhs = P(inner, I(2))
    initial = EqState(lhs=lhs, rhs=I(c * c), var=x)
    # Two roots: target is multi-root if both are integer, else linear
    other = -target_val - 2 * b // a if a == 1 else None  # only valid for a=1
    if a == 1 and other is not None and other != target_val:
        target = _multi_root_target(x, [target_val, other])
    else:
        target = _linear_target(x, target_val)
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_04_direct_square_root",
        params={"a": a, "b": b, "c": c, "target": target_val},
    )


# ===========================================================================
# Category 5: Multi-step linear collection on both sides (1-4 steps)
# ===========================================================================

def gen_cat_05(rng: random.Random, depth: int = 0) -> MotifInstance:
    """a1*x + b1 = a2*x + b2 with a1 != a2 and integer solution."""
    x = VAR
    target_val = _signed(rng, 10)
    while True:
        a1 = _signed(rng, 8)
        a2 = _signed(rng, 8)
        if a1 != a2:
            break
    b1 = rng.randint(-12, 12)
    # b2 = (a1 - a2)*target_val + b1
    b2 = (a1 - a2) * target_val + b1
    lhs = A(M(I(a1), x), I(b1)) if b1 != 0 else M(I(a1), x)
    rhs = A(M(I(a2), x), I(b2)) if b2 != 0 else M(I(a2), x)
    initial = EqState(lhs=lhs, rhs=rhs, var=x)
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="linear", motif_family="cat_05_multi_step_linear",
        params={"a1": a1, "b1": b1, "a2": a2, "b2": b2, "target": target_val},
    )


# ===========================================================================
# Category 6: Distributed / nested linear (2-5 steps)
# ===========================================================================

def gen_cat_06(rng: random.Random, depth: int = 0) -> MotifInstance:
    """a*(x+b) = c or a1*(x-b1) + c1 = c2."""
    x = VAR
    variant = rng.choice([0, 1, 2])
    if variant == 0:
        # a*(x + b) = c
        a = _signed(rng, 6)
        target_val = _signed(rng, 10)
        b = rng.randint(-8, 8)
        c = a * (target_val + b)
        lhs = M(I(a), A(x, I(b))) if b != 0 else M(I(a), x)
        initial = EqState(lhs=lhs, rhs=I(c), var=x)
        params = {"a": a, "b": b, "c": c, "target": target_val}
    elif variant == 1:
        # a1*(x - b1) + c1 = c2
        a1 = _signed(rng, 5)
        b1 = rng.randint(-8, 8)
        c1 = rng.randint(-10, 10)
        target_val = _signed(rng, 8)
        c2 = a1 * (target_val - b1) + c1
        inner = A(x, I(-b1)) if b1 != 0 else x
        lhs = A(M(I(a1), inner), I(c1)) if c1 != 0 else M(I(a1), inner)
        initial = EqState(lhs=lhs, rhs=I(c2), var=x)
        params = {"a1": a1, "b1": b1, "c1": c1, "c2": c2, "target": target_val}
    else:
        # 3*(2x - 5) - 2*(x + 1) = c
        a1 = _signed(rng, 4)
        a2 = _signed(rng, 4)
        coef1 = _signed(rng, 4)
        coef2 = _signed(rng, 4)
        b1 = rng.randint(-6, 6)
        b2 = rng.randint(-6, 6)
        target_val = _signed(rng, 8)
        # LHS: a1*(coef1*x + b1) + a2*(coef2*x + b2) — note: but want != on coefs to be non-degenerate
        c = a1 * (coef1 * target_val + b1) + a2 * (coef2 * target_val + b2)
        if a1 * coef1 + a2 * coef2 == 0:
            return gen_cat_06(rng, depth)  # retry
        lhs = A(
            M(I(a1), A(M(I(coef1), x), I(b1))),
            M(I(a2), A(M(I(coef2), x), I(b2))),
        )
        initial = EqState(lhs=lhs, rhs=I(c), var=x)
        params = {"a1": a1, "coef1": coef1, "b1": b1, "a2": a2, "coef2": coef2, "b2": b2}
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="linear", motif_family="cat_06_distributed_linear",
        params=params,
    )


# ===========================================================================
# Category 7: Linear with numeric fractional coefficients (2-4 steps)
# ===========================================================================

def gen_cat_07(rng: random.Random, depth: int = 0) -> MotifInstance:
    """(p/q)*x + b = c or (x-b)/q = c or (x+b)/q = c."""
    x = VAR
    variant = rng.choice([0, 1, 2])
    q = rng.choice([2, 3, 4, 5])
    if variant == 0:
        # (1/q)*x + b = c, target = q*(c-b)
        b = rng.randint(-10, 10)
        target_val = _signed(rng, 6) * q  # ensure integer-ish target
        target_val = target_val if target_val != 0 else q
        rhs_val = sp.Rational(target_val, q) + b
        lhs = A(M(Rational(1, q), x), I(b)) if b != 0 else M(Rational(1, q), x)
        initial = EqState(lhs=lhs, rhs=sp.sympify(rhs_val), var=x)
        params = {"q": q, "b": b, "target": target_val}
    elif variant == 1:
        # (x - b)/q = c, target = c*q + b
        b = rng.randint(-8, 8)
        c = _signed(rng, 6)
        target_val = c * q + b
        lhs = M(A(x, I(-b)), P(I(q), I(-1))) if b != 0 else M(x, P(I(q), I(-1)))
        initial = EqState(lhs=lhs, rhs=I(c), var=x)
        params = {"q": q, "b": b, "c": c, "target": target_val}
    else:
        # (p*x + b)/q = c
        p = _signed(rng, 5)
        b = rng.randint(-8, 8)
        target_val = _signed(rng, 5)
        c = sp.Rational(p * target_val + b, q)
        num = A(M(I(p), x), I(b)) if b != 0 else M(I(p), x)
        lhs = M(num, P(I(q), I(-1)))
        initial = EqState(lhs=lhs, rhs=sp.sympify(c), var=x)
        params = {"p": p, "q": q, "b": b, "target": target_val}
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="linear", motif_family="cat_07_fractional_coefficient",
        params=params,
    )


# ===========================================================================
# Category 8: Single-rational cross-multiply (2-5 steps) — math_01 fix
# ===========================================================================

def gen_cat_08(rng: random.Random, depth: int = 0) -> MotifInstance:
    """(a*x + b)/(c*x + d) = k, integer target, denom != 0 at target.

    Derivation: a*target + b = k*(c*target + d), so b = k*c*target + k*d - a*target.
    """
    x = VAR
    for _ in range(50):
        a = _signed(rng, 5)
        c = _signed(rng, 5)
        k = _signed(rng, 5)
        if a == k * c:
            continue
        target_val = _signed(rng, 8)
        d = rng.randint(-6, 6)
        b = k * c * target_val + k * d - a * target_val
        if c * target_val + d == 0:
            continue
        excluded_val = sp.Rational(-d, c)
        if excluded_val == target_val:
            continue
        break
    else:
        a, b, c, d, k, target_val = 1, -7, 1, -1, 3, 2
        excluded_val = sp.Integer(1)
    num = A(M(I(a), x), I(b)) if b != 0 else M(I(a), x)
    den = A(M(I(c), x), I(d)) if d != 0 else M(I(c), x)
    lhs = M(num, P(den, I(-1)))
    initial = EqState(lhs=lhs, rhs=I(k), var=x).with_excluded(excluded_val)
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="rational", motif_family="cat_08_cross_multiply",
        params={"a": a, "b": b, "c": c, "d": d, "k": k, "target": target_val},
    )


# ===========================================================================
# Category 9: Rational cancellation / same-denominator combine (2-4 steps)
# ===========================================================================

def gen_cat_09(rng: random.Random, depth: int = 0) -> MotifInstance:
    """(x² - k²)/(x - k) = c — cancel to (x + k) = c → x = c - k."""
    x = VAR
    k = _signed(rng, 6)
    c = _signed(rng, 8)
    target_val = c - k
    if target_val == k:  # would make denominator zero at target
        return gen_cat_09(rng, depth)
    num = A(P(x, I(2)), I(-(k * k)))
    den = A(x, I(-k))
    lhs = M(num, P(den, I(-1)))
    initial = EqState(lhs=lhs, rhs=I(c), var=x).with_excluded(I(k))
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="rational", motif_family="cat_09_rational_cancellation",
        params={"k": k, "c": c, "target": target_val},
    )


# ===========================================================================
# Category 10: Simple rational difference, two denoms (3-6 steps) — math_05 fix
# ===========================================================================

def gen_cat_10(rng: random.Random, depth: int = 0) -> MotifInstance:
    """1/(x - a) - 1/(x - b) = p/q where the resulting equation solves for an integer x."""
    x = VAR
    for _ in range(50):
        a = _signed(rng, 4)
        b = _signed(rng, 4)
        if a == b:
            continue
        target_val = _signed(rng, 8)
        if target_val == a or target_val == b:
            continue
        lhs_val = sp.Rational(1, target_val - a) - sp.Rational(1, target_val - b)
        # We want non-trivial rhs
        if lhs_val == 0:
            continue
        rhs_val = lhs_val
        break
    else:
        a, b, target_val = -1, 1, 2
        rhs_val = sp.Rational(1, 3) - sp.Rational(-1, 3)  # ... not used; fallback path
    da = A(x, I(-a)) if a != 0 else x
    db = A(x, I(-b)) if b != 0 else x
    lhs = A(P(da, I(-1)), M(I(-1), P(db, I(-1))))
    initial = EqState(lhs=lhs, rhs=sp.sympify(rhs_val), var=x).with_excluded(I(a), I(b))
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="rational", motif_family="cat_10_rational_difference",
        params={"a": a, "b": b, "target": target_val, "rhs": rhs_val},
    )


# ===========================================================================
# Category 11: Reciprocal becoming quadratic (3-5 steps) — amc_04 fix
# ===========================================================================

def gen_cat_11(rng: random.Random, depth: int = 0) -> MotifInstance:
    """x + k/x = c. Multiply by x → x² - c*x + k = 0. Pick params so it factors with integer roots."""
    x = VAR
    for _ in range(50):
        r1, r2 = _distinct_signed(rng, 2, 6)
        # x² - (r1+r2)*x + r1*r2 = 0 → x + (r1*r2)/x = r1 + r2
        c = r1 + r2
        k = r1 * r2
        if c == 0 or k == 0:
            continue
        # x cannot be zero (would div by zero)
        break
    else:
        r1, r2, c, k = 1, 4, 5, 4
    lhs = A(x, M(I(k), P(x, I(-1))))
    initial = EqState(lhs=lhs, rhs=I(c), var=x).with_excluded(I(0))
    target = _multi_root_target(x, [r1, r2])
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="rational", motif_family="cat_11_reciprocal_quadratic",
        params={"k": k, "c": c, "r1": r1, "r2": r2},
    )


# ===========================================================================
# Category 12: Direct factorable quadratics (1-3 steps)
# ===========================================================================

def gen_cat_12(rng: random.Random, depth: int = 0) -> MotifInstance:
    """x² + p*x + q = 0 with integer roots r1, r2."""
    x = VAR
    r1, r2 = _distinct_signed(rng, 2, 8)
    p = -(r1 + r2)
    q = r1 * r2
    terms = [P(x, I(2))]
    if p != 0:
        terms.append(M(I(p), x))
    if q != 0:
        terms.append(I(q))
    lhs = A(*terms) if len(terms) > 1 else terms[0]
    initial = EqState(lhs=lhs, rhs=I(0), var=x)
    target = _multi_root_target(x, [r1, r2])
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_12_direct_factorable_quad",
        params={"r1": r1, "r2": r2, "p": p, "q": q},
    )


# ===========================================================================
# Category 13: Quadratics needing rearrangement (3-5 steps)
# ===========================================================================

def gen_cat_13(rng: random.Random, depth: int = 0) -> MotifInstance:
    """x² + p*x = q form (rearrange first), with integer roots."""
    x = VAR
    r1, r2 = _distinct_signed(rng, 2, 6)
    p_full = -(r1 + r2)
    q_full = r1 * r2
    # Move q to rhs: x² + p*x = -q
    p = p_full
    rhs_val = -q_full
    if p == 0:
        return gen_cat_13(rng, depth)
    lhs = A(P(x, I(2)), M(I(p), x))
    initial = EqState(lhs=lhs, rhs=I(rhs_val), var=x)
    target = _multi_root_target(x, [r1, r2])
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_13_quad_rearrange",
        params={"r1": r1, "r2": r2, "p": p, "rhs": rhs_val},
    )


# ===========================================================================
# Category 14: Perfect-square trinomials (3-5 steps) — text_04, amc_05 fix
# ===========================================================================

def gen_cat_14(rng: random.Random, depth: int = 0) -> MotifInstance:
    """x² + 2*a*x + a² = c² (= (x+a)²=c²). One integer root."""
    x = VAR
    a = _signed(rng, 6)
    c = rng.randint(1, 7)
    # Roots: x = -a ± c. Pick positive (target = c - a).
    target_val = c - a
    other_val = -c - a
    lhs = A(P(x, I(2)), M(I(2 * a), x), I(a * a))
    initial = EqState(lhs=lhs, rhs=I(c * c), var=x)
    if target_val != other_val:
        target = _multi_root_target(x, sorted([target_val, other_val]))
    else:
        target = _linear_target(x, target_val)
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_14_perfect_square_trinom",
        params={"a": a, "c": c, "target": target_val},
    )


# ===========================================================================
# Category 15: Irreducible quadratics (formula or completion) (3-7 steps)
# ===========================================================================

def gen_cat_15(rng: random.Random, depth: int = 0) -> MotifInstance:
    """Non-monic factorable quadratic: a*x² + b*x + c = 0 with integer roots.

    Distinct from cat 12 (which is monic). Exercises FACTOR_POLYNOMIAL producing
    a constant-prefactored form `a*(x-r1)*(x-r2) = 0` (canonical, leading-coef-1
    linear factors with a constant prefactor).
    """
    x = VAR
    a = rng.choice([2, 3, -2, -3])
    r1, r2 = _distinct_signed(rng, 2, 6)
    coef_x2 = a
    coef_x = -a * (r1 + r2)
    coef_c = a * r1 * r2
    terms = [M(I(coef_x2), P(x, I(2)))]
    if coef_x != 0:
        terms.append(M(I(coef_x), x))
    if coef_c != 0:
        terms.append(I(coef_c))
    lhs = A(*terms) if len(terms) > 1 else terms[0]
    initial = EqState(lhs=lhs, rhs=I(0), var=x)
    target = _multi_root_target(x, sorted([r1, r2]))
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_15_irreducible_quad",
        params={"a": a, "r1": r1, "r2": r2},
    )


# ===========================================================================
# Category 16: Product expansion → linear cancellation (3-5 steps)
# ===========================================================================

def gen_cat_16(rng: random.Random, depth: int = 0) -> MotifInstance:
    """(x+a)(x+b) - x² = c — x² cancels, leaves linear (a+b)*x + a*b = c."""
    x = VAR
    for _ in range(20):
        a, b = _distinct_signed(rng, 2, 8)
        if a + b == 0:
            continue
        target_val = _signed(rng, 10)
        c = (a + b) * target_val + a * b
        break
    else:
        a, b, target_val, c = 1, 2, 5, 17  # check: (1+2)*5 + 1*2 = 17 ✓
    lhs = A(M(A(x, I(a)), A(x, I(b))), M(I(-1), P(x, I(2))))
    initial = EqState(lhs=lhs, rhs=I(c), var=x)
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="polynomial", motif_family="cat_16_expansion_cancel",
        params={"a": a, "b": b, "c": c, "target": target_val},
    )


# ===========================================================================
# Category 17: Product expansion → quadratic solve (3-6 steps)
# ===========================================================================

def gen_cat_17(rng: random.Random, depth: int = 0) -> MotifInstance:
    """(x+a)(x+b) = c, picks (a,b,c) so resulting quadratic factors over Z."""
    x = VAR
    for _ in range(30):
        a, b = _distinct_signed(rng, 2, 5)
        r1, r2 = _distinct_signed(rng, 2, 6)
        if {r1, r2} == {-a, -b}:
            continue  # trivial (c = 0)
        # (x+a)(x+b) = c ↔ x²+(a+b)x+ab - c = 0
        # We want this to factor as (x-r1)(x-r2)=0, so:
        # x² + (a+b)x + ab - c = x² - (r1+r2)x + r1*r2
        # → a+b = -(r1+r2), ab - c = r1*r2 → c = ab - r1*r2
        if a + b != -(r1 + r2):
            continue
        c = a * b - r1 * r2
        break
    else:
        # fallback: (x+1)(x+2) = 12 → roots 2 and -5
        a, b, c, r1, r2 = 1, 2, 12, 2, -5
    lhs = M(A(x, I(a)), A(x, I(b)))
    initial = EqState(lhs=lhs, rhs=I(c), var=x)
    target = _multi_root_target(x, sorted([r1, r2]))
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_17_expansion_quad",
        params={"a": a, "b": b, "c": c, "r1": r1, "r2": r2},
    )


# ===========================================================================
# Category 18: True multi-denominator LCD → quadratic (4-7 steps)
# ===========================================================================

def gen_cat_18(rng: random.Random, depth: int = 0) -> MotifInstance:
    """1/(x-a) + 1/(x-b) = k. Multiply through → quadratic."""
    x = VAR
    for _ in range(50):
        a, b = _distinct_signed(rng, 2, 4)
        target_val = _signed(rng, 6)
        if target_val in (a, b):
            continue
        # Pair value: 1/(t-a) + 1/(t-b) = k
        k = sp.Rational(1, target_val - a) + sp.Rational(1, target_val - b)
        if k == 0:
            continue
        # We just need ONE root; the other root may or may not be integer.
        break
    else:
        a, b, target_val = 1, -1, 2
        k = sp.Rational(1, 1) + sp.Rational(1, 3)
    da = A(x, I(-a)) if a != 0 else x
    db = A(x, I(-b)) if b != 0 else x
    lhs = A(P(da, I(-1)), P(db, I(-1)))
    initial = EqState(lhs=lhs, rhs=sp.sympify(k), var=x).with_excluded(I(a), I(b))
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="rational", motif_family="cat_18_multi_denom_lcd",
        params={"a": a, "b": b, "k": k, "target": target_val},
    )


# ===========================================================================
# Category 19: Mixed rational + polynomial (4-6 steps)
# ===========================================================================

def gen_cat_19(rng: random.Random, depth: int = 0) -> MotifInstance:
    """x + k/(x-a) = c, multiply through → quadratic."""
    x = VAR
    for _ in range(30):
        a = _signed(rng, 5)
        target_val = _signed(rng, 6)
        if target_val == a:
            continue
        k = _signed(rng, 6)
        # x + k/(x - a) = c at x = target_val
        c = target_val + sp.Rational(k, target_val - a)
        break
    else:
        a, target_val, k = 1, 3, 4
        c = 3 + sp.Rational(4, 2)
    da = A(x, I(-a)) if a != 0 else x
    lhs = A(x, M(I(k), P(da, I(-1))))
    initial = EqState(lhs=lhs, rhs=sp.sympify(c), var=x).with_excluded(I(a))
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="rational", motif_family="cat_19_mixed_rational_poly",
        params={"a": a, "k": k, "c": c, "target": target_val},
    )


# ===========================================================================
# Category 20: Hidden quadratic in repeated affine (4-7 steps)
# ===========================================================================

def gen_cat_20(rng: random.Random, depth: int = 0) -> MotifInstance:
    """(x+a)² - p*(x+a) + q = 0 with integer roots for u = x+a."""
    x = VAR
    for _ in range(20):
        a = _signed(rng, 5)
        u1, u2 = _distinct_signed(rng, 2, 5)
        p = u1 + u2
        q = u1 * u2
        if p == 0 or q == 0:
            continue
        break
    else:
        a, u1, u2, p, q = 1, 2, 3, 5, 6
    inner = A(x, I(a))
    lhs = A(P(inner, I(2)), M(I(-p), inner), I(q))
    initial = EqState(lhs=lhs, rhs=I(0), var=x)
    # x = u - a, so roots: u1 - a, u2 - a
    r1, r2 = u1 - a, u2 - a
    target = _multi_root_target(x, sorted([r1, r2]))
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_20_hidden_quad_affine",
        params={"a": a, "p": p, "q": q, "u1": u1, "u2": u2},
    )


# ===========================================================================
# Category 21: Symmetric reciprocal x + k/x = c — overlaps cat 11 but with
# larger params to push longer paths (4-8 steps)
# ===========================================================================

def gen_cat_21(rng: random.Random, depth: int = 0) -> MotifInstance:
    """x + k/x = c (extends cat 11 with larger k, possibly negative)."""
    x = VAR
    for _ in range(30):
        r1, r2 = _distinct_signed(rng, 2, 10)
        c = r1 + r2
        k = r1 * r2
        if c == 0 or k == 0:
            continue
        break
    else:
        r1, r2, c, k = 1, 9, 10, 9
    lhs = A(x, M(I(k), P(x, I(-1))))
    initial = EqState(lhs=lhs, rhs=I(c), var=x).with_excluded(I(0))
    target = _multi_root_target(x, sorted([r1, r2]))
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="rational", motif_family="cat_21_symmetric_reciprocal",
        params={"k": k, "c": c, "r1": r1, "r2": r2},
    )


# ===========================================================================
# Category 22: Reducible higher-degree (biquadratic, cubic) (3-7 steps)
# ===========================================================================

def gen_cat_22(rng: random.Random, depth: int = 0) -> MotifInstance:
    """x⁴ - 5x² + 4 = 0 family — factorable polynomial of degree ≥ 3."""
    x = VAR
    variant = rng.choice([0, 1])
    if variant == 0:
        # Biquadratic: (x²-a²)(x²-b²) = 0 expanded
        a, b = _distinct_signed(rng, 2, 4)
        a, b = abs(a), abs(b)
        if a == b:
            return gen_cat_22(rng, depth)
        # x⁴ - (a²+b²)x² + a²b² = 0
        coef_x2 = -(a * a + b * b)
        coef_c = a * a * b * b
        lhs = A(P(x, I(4)), M(I(coef_x2), P(x, I(2))), I(coef_c))
        initial = EqState(lhs=lhs, rhs=I(0), var=x)
        target = _multi_root_target(x, sorted([-a, a, -b, b]))
        params = {"a": a, "b": b}
    else:
        # Cubic via expanded factored form: (x-r1)(x-r2)(x-r3)
        r1, r2, r3 = _distinct_signed(rng, 3, 5)
        expr = (x - r1) * (x - r2) * (x - r3)
        expanded = sp.expand(expr)
        initial = EqState(lhs=expanded, rhs=I(0), var=x)
        target = _multi_root_target(x, sorted([r1, r2, r3]))
        params = {"r1": r1, "r2": r2, "r3": r3}
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_22_reducible_higher_degree",
        params=params,
    )


# ===========================================================================
# Category 23: Denominator-domain trap (3-5 steps)
# ===========================================================================

def gen_cat_23(rng: random.Random, depth: int = 0) -> MotifInstance:
    """(x² - a²)/(x + a) = c — denominator cancels with (x - a) factor:
    (x-a)(x+a)/(x+a) → (x-a), so x - a = c → x = c + a, x != -a.
    """
    x = VAR
    a = _signed(rng, 6)
    c = _signed(rng, 6)
    target_val = c + a
    if target_val == -a:
        return gen_cat_23(rng, depth)
    num = A(P(x, I(2)), I(-(a * a)))
    den = A(x, I(a)) if a != 0 else x
    lhs = M(num, P(den, I(-1)))
    initial = EqState(lhs=lhs, rhs=I(c), var=x).with_excluded(I(-a))
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="rational", motif_family="cat_23_denom_domain_trap",
        params={"a": a, "c": c, "target": target_val},
    )


# ===========================================================================
# Categories 24-30: Adversarial — adapt existing motif templates
# ===========================================================================

def _adapt_motif(family: str) -> Callable[[random.Random, int], MotifInstance]:
    """Wrap ggmr.training.parameter_sampling.sample_motif_instance for a family."""
    def _gen(rng: random.Random, depth: int = 0) -> MotifInstance:
        # Import lazily so this module is importable in environments without rules registered
        from ggmr.training.parameter_sampling import sample_motif_instance
        inst = sample_motif_instance(family, rng, max_attempts=15)
        if inst is None:
            raise RuntimeError(f"motif {family} sampling exhausted")
        # Mutate motif_family to reflect category labeling
        return MotifInstance(
            eq_state=inst.eq_state,
            target_eq_state=inst.target_eq_state,
            category=inst.category,
            motif_family=family,
            params=inst.params,
        )
    return _gen


# ===========================================================================
# Category 31: nth-root (NTH_ROOT_BOTH_SIDES) (1-3 steps)
# ===========================================================================

def gen_cat_31(rng: random.Random, depth: int = 0) -> MotifInstance:
    """(a*x + b)^n = c^n, target = (c - b)/a integer."""
    x = VAR
    for _ in range(30):
        n = rng.choice([3, 3, 3, 4, 4, 5])
        a = rng.choice([1, 1, 1, 2, 2, 3])
        b = rng.randint(-5, 5)
        if n % 2 == 0:
            root_int = rng.randint(1, 5)  # need RHS >= 0 for even n
        else:
            root_int = _signed(rng, 5)
        # target: (a*x + b)^n = root_int^n → a*x + b = root_int → x = (root_int - b)/a
        if (root_int - b) % a != 0:
            continue
        target_val = (root_int - b) // a
        c_val = root_int ** n
        break
    else:
        n, a, b, root_int, target_val = 3, 1, 0, 2, 2
        c_val = 8
    inner = A(M(I(a), x), I(b)) if b != 0 else M(I(a), x)
    if a == 1:
        inner = A(x, I(b)) if b != 0 else x
    lhs = P(inner, I(n))
    initial = EqState(lhs=lhs, rhs=I(c_val), var=x)
    return MotifInstance(
        eq_state=initial, target_eq_state=_linear_target(x, target_val),
        category="polynomial", motif_family="cat_31_nth_root",
        params={"n": n, "a": a, "b": b, "c": c_val, "target": target_val},
    )


# ===========================================================================
# Category 32: Absolute value (SPLIT_ABSOLUTE_VALUE) (2-4 steps)
# ===========================================================================

def gen_cat_32(rng: random.Random, depth: int = 0) -> MotifInstance:
    """|a*x + b| = c, target = positive branch x = (c - b)/a."""
    x = VAR
    for _ in range(30):
        a = rng.choice([1, 1, 2, 3])
        b = rng.randint(-8, 8)
        c = rng.randint(1, 9)
        if (c - b) % a != 0:
            continue
        target_val = (c - b) // a
        other_val = (-c - b) // a if (-c - b) % a == 0 else None
        break
    else:
        a, b, c, target_val, other_val = 1, -3, 5, 8, -2
    inner = A(M(I(a), x), I(b)) if b != 0 else M(I(a), x)
    if a == 1:
        inner = A(x, I(b)) if b != 0 else x
    lhs = sp.Abs(inner)
    initial = EqState(lhs=lhs, rhs=I(c), var=x)
    # SPLIT_ABSOLUTE_VALUE branches into single-root paths; multi-root target
    # would never match. Always target the positive branch (linear x = target_val).
    target = _linear_target(x, target_val)
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_32_absolute_value",
        params={"a": a, "b": b, "c": c, "target": target_val, "other": other_val},
    )


# ===========================================================================
# Category 33: GCF factoring + perfect-square remainder (3-6 steps)
# ===========================================================================

def gen_cat_33(rng: random.Random, depth: int = 0) -> MotifInstance:
    """gcf*(x-r)² = 0 expanded — needs GCF + perfect square trinomial recognition."""
    x = VAR
    gcf = rng.choice([2, 3, 4, 5])
    r = _signed(rng, 6)
    # gcf*(x² - 2rx + r²) = 0 → expand
    coef_x2 = gcf
    coef_x = -2 * gcf * r
    coef_c = gcf * r * r
    terms = [M(I(coef_x2), P(x, I(2)))]
    if coef_x != 0:
        terms.append(M(I(coef_x), x))
    if coef_c != 0:
        terms.append(I(coef_c))
    lhs = A(*terms) if len(terms) > 1 else terms[0]
    initial = EqState(lhs=lhs, rhs=I(0), var=x)
    target = _linear_target(x, r)
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_33_gcf_factoring",
        params={"gcf": gcf, "r": r},
    )


# ===========================================================================
# Category 34: Factor by grouping (4-7 steps)
# ===========================================================================

def gen_cat_34(rng: random.Random, depth: int = 0) -> MotifInstance:
    """x³ + a*x² + b*x + a*b = 0 expanded — groups as (x²+b)(x+a)=0.

    Integer roots when b is negative (b = -c² for some c gives ±c, -a as roots).
    """
    x = VAR
    a = _signed(rng, 4)
    c = rng.randint(1, 4)
    # x²·(x + a) + (-c²)·(x + a) = (x + a)(x² - c²) = (x+a)(x-c)(x+c)
    # Expand: x³ + a*x² - c²*x - a*c²
    coef_x3 = 1
    coef_x2 = a
    coef_x = -c * c
    coef_c = -a * c * c
    terms = [P(x, I(3))]
    if coef_x2 != 0:
        terms.append(M(I(coef_x2), P(x, I(2))))
    if coef_x != 0:
        terms.append(M(I(coef_x), x))
    if coef_c != 0:
        terms.append(I(coef_c))
    lhs = A(*terms)
    initial = EqState(lhs=lhs, rhs=I(0), var=x)
    roots = sorted({-a, c, -c})
    target = _multi_root_target(x, roots)
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_34_factor_by_grouping",
        params={"a": a, "c": c, "roots": roots},
    )


# ===========================================================================
# Category 35: Polynomial division to find roots (5-9 steps)
# ===========================================================================

def gen_cat_35(rng: random.Random, depth: int = 0) -> MotifInstance:
    """Cubic with 3 integer roots — exercises FACTOR_POLYNOMIAL or
    RATIONAL_ROOT_THEOREM + SYNTHETIC_DIVISION."""
    x = VAR
    r1, r2, r3 = _distinct_signed(rng, 3, 5)
    expr = (x - r1) * (x - r2) * (x - r3)
    expanded = sp.expand(expr)
    initial = EqState(lhs=expanded, rhs=I(0), var=x)
    target = _multi_root_target(x, sorted([r1, r2, r3]))
    return MotifInstance(
        eq_state=initial, target_eq_state=target,
        category="polynomial", motif_family="cat_35_polynomial_division",
        params={"r1": r1, "r2": r2, "r3": r3},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, Callable[[random.Random, int], MotifInstance]] = {
    "cat_01_canonical_target":           gen_cat_01,
    "cat_02_affine_isolation":           gen_cat_02,
    "cat_03_already_factored":           gen_cat_03,
    "cat_04_direct_square_root":         gen_cat_04,
    "cat_05_multi_step_linear":          gen_cat_05,
    "cat_06_distributed_linear":         gen_cat_06,
    "cat_07_fractional_coefficient":     gen_cat_07,
    "cat_08_cross_multiply":             gen_cat_08,
    "cat_09_rational_cancellation":      gen_cat_09,
    "cat_10_rational_difference":        gen_cat_10,
    "cat_11_reciprocal_quadratic":       gen_cat_11,
    "cat_12_direct_factorable_quad":     gen_cat_12,
    "cat_13_quad_rearrange":             gen_cat_13,
    "cat_14_perfect_square_trinom":      gen_cat_14,
    "cat_15_irreducible_quad":           gen_cat_15,
    "cat_16_expansion_cancel":           gen_cat_16,
    "cat_17_expansion_quad":             gen_cat_17,
    "cat_18_multi_denom_lcd":            gen_cat_18,
    "cat_19_mixed_rational_poly":        gen_cat_19,
    "cat_20_hidden_quad_affine":         gen_cat_20,
    "cat_21_symmetric_reciprocal":       gen_cat_21,
    "cat_22_reducible_higher_degree":    gen_cat_22,
    "cat_23_denom_domain_trap":          gen_cat_23,
    # Adversarial (motif templates):
    "cat_24_L1_polynomial_shield":       _adapt_motif("L1"),
    "cat_25_L3_distributed_scalar":      _adapt_motif("L3"),
    "cat_26_P3_irreducible_disguise":    _adapt_motif("P3"),
    "cat_27_P4_cubic_cross_ratio":       _adapt_motif("P4"),
    "cat_28_R1_fractional_shield":       _adapt_motif("R1"),
    "cat_29_R2_distributed_fractional":  _adapt_motif("R2"),
    "cat_30_v1_ex1_cross_reciprocal":    _adapt_motif("v1_ex1"),
    # New-rule coverage:
    "cat_31_nth_root":                   gen_cat_31,
    "cat_32_absolute_value":             gen_cat_32,
    "cat_33_gcf_factoring":              gen_cat_33,
    "cat_34_factor_by_grouping":         gen_cat_34,
    "cat_35_polynomial_division":        gen_cat_35,
}


# Tier classification for BFS budget + timeout
TIER_TRIVIAL = {"cat_01_canonical_target", "cat_02_affine_isolation",
                "cat_03_already_factored", "cat_04_direct_square_root"}
TIER_EASY = {"cat_05_multi_step_linear", "cat_06_distributed_linear",
             "cat_07_fractional_coefficient", "cat_08_cross_multiply",
             "cat_09_rational_cancellation", "cat_10_rational_difference",
             "cat_11_reciprocal_quadratic", "cat_12_direct_factorable_quad",
             "cat_13_quad_rearrange"}
TIER_MEDIUM = {"cat_14_perfect_square_trinom", "cat_15_irreducible_quad",
               "cat_16_expansion_cancel", "cat_17_expansion_quad",
               "cat_18_multi_denom_lcd", "cat_19_mixed_rational_poly",
               "cat_20_hidden_quad_affine", "cat_21_symmetric_reciprocal",
               "cat_22_reducible_higher_degree", "cat_23_denom_domain_trap",
               "cat_31_nth_root", "cat_32_absolute_value",
               "cat_33_gcf_factoring", "cat_34_factor_by_grouping",
               "cat_35_polynomial_division"}
TIER_ADVERSARIAL = {"cat_24_L1_polynomial_shield", "cat_25_L3_distributed_scalar",
                    "cat_26_P3_irreducible_disguise", "cat_27_P4_cubic_cross_ratio",
                    "cat_28_R1_fractional_shield", "cat_29_R2_distributed_fractional",
                    "cat_30_v1_ex1_cross_reciprocal"}


def bfs_budget_for(category: str) -> int:
    if category in TIER_TRIVIAL:
        return 1_000
    if category in TIER_EASY:
        return 5_000
    if category in TIER_MEDIUM:
        return 15_000
    return 50_000  # adversarial


def timeout_for(category: str) -> float:
    if category in TIER_TRIVIAL:
        return 30.0
    if category in TIER_EASY:
        return 60.0
    if category in TIER_MEDIUM:
        return 120.0
    return 240.0  # adversarial
