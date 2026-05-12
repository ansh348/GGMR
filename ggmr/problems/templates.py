"""Template families for problem generation.

Each template produces a "seed" canonical-target EqState. The ReverseGenerator
then applies inverse rules to manufacture a problem at the target depth.
"""

from __future__ import annotations

import random

import sympy as sp
from sympy import Add, Integer, Mul, Symbol

from ..state import EqState


def linear_seed(rng: random.Random, var: Symbol = None) -> EqState:
    """Seed: x = a, where a ∈ randint(-10, 10), nonzero."""
    if var is None:
        var = sp.Symbol("x")
    a = rng.randint(-10, 10)
    if a == 0:
        a = 1
    return EqState(lhs=var, rhs=Integer(a), var=var)


def quadratic_seed(rng: random.Random, var: Symbol = None) -> EqState:
    """Seed: (x - r1)*(x - r2) = 0."""
    if var is None:
        var = sp.Symbol("x")
    r1 = rng.randint(-5, 5)
    r2 = rng.randint(-5, 5)
    factor1 = Add(var, Integer(-r1), evaluate=False)
    factor2 = Add(var, Integer(-r2), evaluate=False)
    lhs = Mul(factor1, factor2, evaluate=False)
    return EqState(lhs=lhs, rhs=Integer(0), var=var)


def rational_seed(rng: random.Random, var: Symbol = None) -> EqState:
    """Seed: x = a/b for small integer a, b."""
    if var is None:
        var = sp.Symbol("x")
    a = rng.randint(-5, 5)
    b = rng.choice([2, 3, 5])
    return EqState(lhs=var, rhs=sp.Rational(a, b), var=var)


def polynomial_seed(rng: random.Random, var: Symbol = None) -> EqState:
    """Seed: (x - r1)*(x - r2)*(x - r3) = 0 (cubic)."""
    if var is None:
        var = sp.Symbol("x")
    r1 = rng.randint(-3, 3)
    r2 = rng.randint(-3, 3)
    r3 = rng.randint(-3, 3)
    f1 = Add(var, Integer(-r1), evaluate=False)
    f2 = Add(var, Integer(-r2), evaluate=False)
    f3 = Add(var, Integer(-r3), evaluate=False)
    lhs = Mul(f1, f2, f3, evaluate=False)
    return EqState(lhs=lhs, rhs=Integer(0), var=var)


def mixed_seed(rng: random.Random, var: Symbol = None) -> EqState:
    """Random choice over the four base templates."""
    choice = rng.choice(["linear", "quadratic", "rational", "polynomial"])
    return TEMPLATES[choice](rng, var=var)


def linear_irrational_seed(rng: random.Random, var: Symbol = None) -> EqState:
    """Seed: x = a + sqrt(b) for small integer a and non-perfect-square b.

    The rhs is var-free, so this passes is_linear_target. The forward solve
    after disguise must use COMPLETE_THE_SQUARE + SQRT_BOTH_SIDES — a multi-
    step macro chain the structural heuristic doesn't anticipate.
    """
    if var is None:
        var = sp.Symbol("x")
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    b = rng.choice([2, 3, 5, 6, 7])
    rhs = Add(Integer(a), sp.sqrt(Integer(b)), evaluate=False)
    return EqState(lhs=var, rhs=rhs, var=var)


TEMPLATES = {
    "linear": linear_seed,
    "quadratic": quadratic_seed,
    "rational": rational_seed,
    "polynomial": polynomial_seed,
    "mixed": mixed_seed,
    "linear_irrational": linear_irrational_seed,
}
