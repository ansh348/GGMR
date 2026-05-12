"""Tests for Phase 1b quadratic rules (34-37)."""

from __future__ import annotations

import sympy as sp

from ggmr.rules.core import *  # noqa: F401,F403  (registers)
from ggmr.rules.core.quadratic import (
    FactorByGrouping,
    FactorDifferenceOfSquaresAt,
    PerfectSquareTrinomialAt,
    QuadraticFormula,
)
from ggmr.rules.base import Action
from ggmr.soundness import VERIFY_PASS, verify_transition
from ggmr.state import EqState


x = sp.Symbol("x")


def _solset(s: EqState) -> frozenset:
    return s.solution_set()


def _verify(parent: EqState, child: EqState) -> str:
    verdict, _ = verify_transition(
        parent.lhs,
        parent.rhs,
        child.lhs,
        child.rhs,
        parent.var,
        parent_excluded=parent.excluded,
        child_excluded=child.excluded,
    )
    return verdict


# --- QUADRATIC_FORMULA ----------------------------------------------------


def test_quadratic_formula_apply_principal_branch():
    # x² - 5x + 6 = 0 → x = (5 + sqrt(25-24))/2 = 3 (principal +)
    s = EqState.from_strings("x**2 - 5*x + 6", "0")
    rule = QuadraticFormula()
    action = Action(rule.name, target_side="lhs")
    g = rule.guard(s, action)
    assert g.ok
    child = rule.apply(s, action)
    # The child has x = principal-root expression
    assert child.lhs == x
    # Principal root for this quadratic is 3 (the bigger of {2, 3})
    assert sp.simplify(child.rhs - 3) == 0


def test_quadratic_formula_skips_when_rhs_nonzero():
    s = EqState.from_strings("x**2 - 5*x + 6", "1")
    rule = QuadraticFormula()
    actions = list(rule.enumerate(s))
    assert actions == []


def test_quadratic_formula_skips_when_not_quadratic():
    s = EqState.from_strings("x + 1", "0")
    rule = QuadraticFormula()
    actions = list(rule.enumerate(s))
    assert actions == []


def test_quadratic_formula_guard_rejects_zero_leading_coeff():
    # Synthetic — both sides linear, guard on the lhs side
    s = EqState.from_strings("x + 1", "0")
    rule = QuadraticFormula()
    g = rule.guard(s, Action(rule.name, target_side="lhs"))
    assert g.ok is False


# --- FACTOR_BY_GROUPING ---------------------------------------------------


def test_factor_by_grouping_apply():
    # x² - 5x + 6 = 0 → (x - 2)(x - 3) = 0
    s = EqState.from_strings("x**2 - 5*x + 6", "0")
    rule = FactorByGrouping()
    action = Action(rule.name, target_side="lhs")
    g = rule.guard(s, action)
    assert g.ok
    child = rule.apply(s, action)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_factor_by_grouping_skips_non_quadratic():
    s = EqState.from_strings("x + 1", "0")
    rule = FactorByGrouping()
    actions = list(rule.enumerate(s))
    assert actions == []


def test_factor_by_grouping_skips_non_integer_coeffs():
    # 0.5*x² + x + 1 = 0
    s = EqState(
        lhs=sp.Add(
            sp.Mul(sp.Rational(1, 2), x**2, evaluate=False),
            x,
            sp.Integer(1),
            evaluate=False,
        ),
        rhs=sp.Integer(0),
        var=x,
    )
    rule = FactorByGrouping()
    actions = list(rule.enumerate(s))
    assert actions == []


# --- FACTOR_DIFFERENCE_OF_SQUARES_AT --------------------------------------


def test_factor_difference_of_squares_apply():
    # x² - 4 → (x - 2)(x + 2)
    lhs = sp.Add(
        sp.Pow(x, sp.Integer(2), evaluate=False),
        sp.Integer(-4),
        evaluate=False,
    )
    s = EqState(lhs=lhs, rhs=sp.Integer(0), var=x)
    rule = FactorDifferenceOfSquaresAt()
    actions = list(rule.enumerate(s))
    a = next((a for a in actions if a.target_side == "lhs" and a.target_path == ()), None)
    assert a is not None
    child = rule.apply(s, a)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_factor_difference_of_squares_skips_when_no_diff():
    s = EqState.from_strings("x + 1", "0")
    rule = FactorDifferenceOfSquaresAt()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs" and a.target_path == ()]
    assert actions == []


# --- PERFECT_SQUARE_TRINOMIAL_AT ------------------------------------------


def test_perfect_square_trinomial_apply():
    # x² + 2*x + 1 → (x + 1)²
    s = EqState.from_strings("x**2 + 2*x + 1", "0")
    rule = PerfectSquareTrinomialAt()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs" and a.target_path == ()]
    assert len(actions) >= 1
    a = actions[0]
    child = rule.apply(s, a)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_perfect_square_trinomial_skips_non_perfect_square():
    # x² + 2x + 5 → not a perfect square
    s = EqState.from_strings("x**2 + 2*x + 5", "0")
    rule = PerfectSquareTrinomialAt()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs" and a.target_path == ()]
    assert actions == []
