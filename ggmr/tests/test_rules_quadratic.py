"""Tests for quadratic-family rules."""

from __future__ import annotations

import sympy as sp

from ggmr.rules.core import *  # noqa: F401,F403
from ggmr.rules.core.quadratic import (
    CompleteTheSquare,
    SimplifyNumericAt,
    SqrtBothSides,
)
from ggmr.rules.base import Action
from ggmr.soundness import VERIFY_PASS, verify_transition
from ggmr.state import EqState

x = sp.Symbol("x")


def _verify(parent: EqState, child: EqState) -> str:
    v, _ = verify_transition(
        parent.lhs, parent.rhs, child.lhs, child.rhs, parent.var,
        parent_excluded=parent.excluded, child_excluded=child.excluded,
    )
    return v


def test_complete_the_square_apply():
    """x² - 4x + 1 -> (x - 2)² - 3 (depending on rhs)."""
    s = EqState.from_strings("x**2 - 4*x", "-1")
    rule = CompleteTheSquare()
    actions = list(rule.enumerate(s))
    # No quadratic in lhs since lhs `x^2 - 4x` doesn't have a constant term in its
    # current Add structure — but Poly will accept it. So we expect it to enumerate.
    assert actions, "COMPLETE_THE_SQUARE should enumerate when lhs is a quadratic"
    child = rule.apply(s, actions[0])
    assert child.solution_set() == s.solution_set()
    assert _verify(s, child) == VERIFY_PASS


def test_sqrt_both_sides_on_perfect_square():
    """(x - 2)² = 3 -> (x - 2) = sqrt(3) (positive branch)"""
    lhs = sp.Pow(sp.Add(x, sp.Integer(-2), evaluate=False), sp.Integer(2), evaluate=False)
    s = EqState(lhs=lhs, rhs=sp.Integer(3), var=x)
    rule = SqrtBothSides()
    actions = list(rule.enumerate(s))
    assert actions, "SQRT_BOTH_SIDES should enumerate for (x-2)^2 = nonneg constant"
    child = rule.apply(s, actions[0])
    # Positive branch gives ONE root; child solution set is subset of parent
    assert child.solution_set().issubset(s.solution_set())


def test_simplify_numeric_at_collapses_pure_constants():
    """Add(3, -7) should simplify to -4."""
    inner = sp.Add(sp.Integer(3), sp.Integer(-7), evaluate=False)
    outer = sp.Add(x, inner, evaluate=False)
    s = EqState(lhs=outer, rhs=sp.Integer(0), var=x)
    rule = SimplifyNumericAt()
    actions = list(rule.enumerate(s))
    # one action: collapse the Add(3, -7) at path (1,)
    assert any(a.target_path == (1,) for a in actions)
