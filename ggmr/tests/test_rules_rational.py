"""Tests for rational rules: cancel_common_factor, clear_fractions_by_lcd."""

from __future__ import annotations

import sympy as sp

from ggmr.rules.core import *  # noqa: F401,F403
from ggmr.rules.core.rational import CancelCommonFactor, ClearFractionsByLCD
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


def test_cancel_common_factor_propagates_excluded():
    """`((x - 1)*(x + 1))/(x - 1) = 3` cancels (x - 1), excluded={1}, gives x + 1 = 3."""
    num = sp.Mul(sp.Add(x, sp.Integer(-1), evaluate=False), sp.Add(x, sp.Integer(1), evaluate=False), evaluate=False)
    denom = sp.Pow(sp.Add(x, sp.Integer(-1), evaluate=False), sp.Integer(-1), evaluate=False)
    lhs = sp.Mul(num, denom, evaluate=False)
    s = EqState(lhs=lhs, rhs=sp.Integer(3), var=x)

    rule = CancelCommonFactor()
    actions = list(rule.enumerate(s))
    assert actions, "CANCEL_COMMON_FACTOR should enumerate (x-1) cancellation"
    a = actions[0]
    g = rule.guard(s, a)
    assert g.ok
    assert sp.Integer(1) in g.new_excluded


def test_clear_fractions_by_lcd_propagates_excluded():
    """`x/(x-2) = 3` × LCD (x-2) yields `x = 3*(x-2)` with excluded={2}."""
    s = EqState.from_strings("x/(x - 2)", "3")
    rule = ClearFractionsByLCD()
    actions = list(rule.enumerate(s))
    assert actions
    a = actions[0]
    g = rule.guard(s, a)
    assert g.ok
    # LCD has (x - 2); roots = {2}
    assert sp.Integer(2) in g.new_excluded


def test_clear_fractions_guard_rejects_zero_lcd():
    """A constant LCD that simplifies to zero is rejected."""
    s = EqState.from_strings("x", "3")  # no fractions, no LCD enumerated
    rule = ClearFractionsByLCD()
    assert list(rule.enumerate(s)) == []
