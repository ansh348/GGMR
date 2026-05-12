"""Tests for Phase 1b arithmetic rules (16-20).

Coverage: positive apply, guard rejection (where applicable), soundness verdict.
"""

from __future__ import annotations

import sympy as sp

from ggmr.rules.core import *  # noqa: F401,F403  (registers)
from ggmr.rules.core.arithmetic import (
    IsolateVariable,
    MoveAllToLhs,
    MoveAllToRhs,
    ReciprocateBothSides,
    SquareBothSides,
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


# --- MOVE_ALL_TO_LHS ------------------------------------------------------


def test_move_all_to_lhs_apply_zero_rhs():
    s = EqState.from_strings("2*x + 3", "7")
    rule = MoveAllToLhs()
    action = Action(rule.name)
    child = rule.apply(s, action)
    assert child.rhs == sp.Integer(0)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_move_all_to_lhs_skips_if_rhs_already_zero():
    s = EqState.from_strings("2*x + 3", "0")
    rule = MoveAllToLhs()
    actions = list(rule.enumerate(s))
    assert actions == []  # already at desired form


def test_move_all_to_lhs_enumerates_when_rhs_nonzero():
    s = EqState.from_strings("x + 1", "5")
    rule = MoveAllToLhs()
    actions = list(rule.enumerate(s))
    assert len(actions) == 1


# --- MOVE_ALL_TO_RHS ------------------------------------------------------


def test_move_all_to_rhs_apply_zero_lhs():
    s = EqState.from_strings("3", "x + 5")
    rule = MoveAllToRhs()
    action = Action(rule.name)
    child = rule.apply(s, action)
    assert child.lhs == sp.Integer(0)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_move_all_to_rhs_skips_if_lhs_already_zero():
    s = EqState.from_strings("0", "x + 1")
    rule = MoveAllToRhs()
    actions = list(rule.enumerate(s))
    assert actions == []


# --- ISOLATE_VARIABLE -----------------------------------------------------


def test_isolate_variable_simple_linear():
    s = EqState.from_strings("2*x + 3", "7")
    rule = IsolateVariable()
    actions = list(rule.enumerate(s))
    assert len(actions) == 1
    child = rule.apply(s, actions[0])
    assert child.lhs == x
    # Effective solution set preserved
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_isolate_variable_skips_already_isolated():
    s = EqState.from_strings("x", "2")
    rule = IsolateVariable()
    actions = list(rule.enumerate(s))
    assert actions == []


def test_isolate_variable_skips_quadratic():
    s = EqState.from_strings("x**2 + 1", "5")
    rule = IsolateVariable()
    actions = list(rule.enumerate(s))
    assert actions == []


def test_isolate_variable_skips_when_var_on_rhs():
    s = EqState.from_strings("3", "2*x + 1")
    rule = IsolateVariable()
    # Var is on rhs — ISOLATE expects var on lhs (FLIP_SIDES handles symmetry).
    actions = list(rule.enumerate(s))
    assert actions == []


def test_isolate_variable_guard_rejects_zero_coefficient():
    # Construct a state where _linear_in_var would say (a=0, b=...) — synthetic.
    # Easier: state where lhs is a constant in var.
    s = EqState.from_strings("3", "x")
    rule = IsolateVariable()
    # enumerate returns nothing, but we test guard directly with a synthetic action
    # using a state we know gives zero coefficient (lhs has no var). Build manually.
    # The guard fails because _linear_in_var returns None when no var.
    s2 = EqState.from_strings("0", "5")  # both sides var-free
    g = rule.guard(s2, Action(rule.name))
    assert g.ok is False


# --- SQUARE_BOTH_SIDES ----------------------------------------------------


def test_square_both_sides_apply():
    # Construct a state with sqrt on one side so the rule fires.
    s = EqState.from_strings("sqrt(x + 1)", "3")
    rule = SquareBothSides()
    action = Action(rule.name)
    child = rule.apply(s, action)
    assert child.lhs == sp.Pow(s.lhs, sp.Integer(2), evaluate=False)
    assert child.rhs == sp.Pow(s.rhs, sp.Integer(2), evaluate=False)
    # Side condition Eq(lhs - rhs, 0) is recorded
    assert any(isinstance(c, sp.Equality) for c in child.side_conditions)


def test_square_both_sides_enumerates_when_sqrt_present():
    s = EqState.from_strings("sqrt(x)", "3")
    rule = SquareBothSides()
    actions = list(rule.enumerate(s))
    assert len(actions) == 1


def test_square_both_sides_skips_when_no_sqrt():
    """Phase 1b restriction: only fires when a sqrt is present (undoing it)."""
    s = EqState.from_strings("x", "3")
    rule = SquareBothSides()
    actions = list(rule.enumerate(s))
    assert actions == []


def test_square_both_sides_skips_when_lhs_eq_rhs():
    s = EqState.from_strings("sqrt(x)", "sqrt(x)")
    rule = SquareBothSides()
    actions = list(rule.enumerate(s))
    assert actions == []


# --- RECIPROCATE_BOTH_SIDES -----------------------------------------------


def test_reciprocate_both_sides_apply():
    s = EqState.from_strings("x", "3")
    rule = ReciprocateBothSides()
    action = Action(rule.name)
    child = rule.apply(s, action)
    # Apply takes 1/lhs = 1/rhs structurally
    assert child.lhs == sp.Pow(s.lhs, sp.Integer(-1), evaluate=False)
    assert child.rhs == sp.Pow(s.rhs, sp.Integer(-1), evaluate=False)


def test_reciprocate_both_sides_guard_rejects_zero_lhs():
    s = EqState.from_strings("0", "3")
    rule = ReciprocateBothSides()
    g = rule.guard(s, Action(rule.name))
    assert g.ok is False


def test_reciprocate_both_sides_guard_rejects_zero_rhs():
    s = EqState.from_strings("x", "0")
    rule = ReciprocateBothSides()
    g = rule.guard(s, Action(rule.name))
    assert g.ok is False


def test_reciprocate_both_sides_guard_propagates_excluded_for_var_lhs():
    s = EqState.from_strings("x - 5", "3")
    rule = ReciprocateBothSides()
    g = rule.guard(s, Action(rule.name))
    assert g.ok is True
    # x = 5 makes lhs zero; should be in new_excluded
    assert sp.Integer(5) in g.new_excluded


def test_reciprocate_both_sides_skips_when_either_side_is_zero():
    s = EqState.from_strings("0", "x")
    rule = ReciprocateBothSides()
    assert list(rule.enumerate(s)) == []
