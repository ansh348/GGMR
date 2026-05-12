"""Tests for arithmetic rules: add/multiply/divide/negate/flip both sides.

Each rule gets:
- positive: an action that produces the expected next state
- guard rejection: an action that the guard correctly refuses
- soundness: the produced state has the same solution set as the parent
"""

from __future__ import annotations

import sympy as sp

from ggmr.rules.core import *  # noqa: F401,F403  (registers)
from ggmr.rules.core.arithmetic import (
    AddToBothSides,
    DivideBothSidesBy,
    FlipSides,
    MultiplyBothSidesBy,
    NegateBothSides,
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


# --- ADD_TO_BOTH_SIDES ----------------------------------------------------


def test_add_to_both_sides_apply_subtract_constant():
    s = EqState.from_strings("2*x + 3", "7")
    rule = AddToBothSides()
    action = Action(rule.name, params=(sp.Integer(-3),))
    child = rule.apply(s, action)
    # Solution set preserved
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_add_to_both_sides_enumerates():
    s = EqState.from_strings("2*x + 3", "7")
    rule = AddToBothSides()
    actions = list(rule.enumerate(s))
    # Must include subtracting 3 and subtracting 7
    params_set = {a.params for a in actions}
    assert (sp.Integer(-3),) in params_set
    assert (sp.Integer(-7),) in params_set


# --- MULTIPLY_BOTH_SIDES_BY ----------------------------------------------


def test_multiply_both_sides_apply_preserves_solset():
    s = EqState.from_strings("x", "3")
    rule = MultiplyBothSidesBy()
    action = Action(rule.name, params=(sp.Integer(2),))
    child = rule.apply(s, action)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_multiply_both_sides_guard_rejects_zero():
    s = EqState.from_strings("x", "3")
    rule = MultiplyBothSidesBy()
    action = Action(rule.name, params=(sp.Integer(0),))
    g = rule.guard(s, action)
    assert g.ok is False


def test_multiply_both_sides_guard_rejects_symbolic_zero():
    """Rejects `Add(7, -7, evaluate=False)` which simplifies to 0."""
    s = EqState.from_strings("x", "3")
    rule = MultiplyBothSidesBy()
    sym_zero = sp.Add(sp.Integer(7), sp.Integer(-7), evaluate=False)
    action = Action(rule.name, params=(sym_zero,))
    g = rule.guard(s, action)
    assert g.ok is False


# --- DIVIDE_BOTH_SIDES_BY ------------------------------------------------


def test_divide_both_sides_apply_preserves_solset():
    s = EqState.from_strings("2*x", "4")
    rule = DivideBothSidesBy()
    action = Action(rule.name, params=(sp.Integer(2),))
    child = rule.apply(s, action)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_divide_both_sides_guard_rejects_zero():
    s = EqState.from_strings("2*x", "4")
    rule = DivideBothSidesBy()
    action = Action(rule.name, params=(sp.Integer(0),))
    assert rule.guard(s, action).ok is False


# --- NEGATE_BOTH_SIDES ---------------------------------------------------


def test_negate_both_sides():
    s = EqState.from_strings("-x", "2")
    rule = NegateBothSides()
    action = Action(rule.name)
    child = rule.apply(s, action)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


# --- FLIP_SIDES ---------------------------------------------------------


def test_flip_sides_swaps():
    s = EqState.from_strings("1", "x")
    rule = FlipSides()
    action = Action(rule.name)
    child = rule.apply(s, action)
    assert child.lhs == s.rhs and child.rhs == s.lhs
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS
