"""Tests for Phase 1b rational rules (28-33)."""

from __future__ import annotations

import sympy as sp

from ggmr.rules.core import *  # noqa: F401,F403  (registers)
from ggmr.rules.core.rational import (
    CombineFractionsAt,
    CommonDenominatorAt,
    CrossMultiply,
    PartialFractions,
    SimplifyAt,
    SplitFractionAt,
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


# --- CROSS_MULTIPLY -------------------------------------------------------


def test_cross_multiply_apply():
    # 1/(x-1) = 2/(x+1) → (x+1) = 2*(x-1). Build with sp.parse_expr.
    s = EqState.from_strings("1/(x - 1)", "2/(x + 1)")
    rule = CrossMultiply()
    action = Action(rule.name)
    g = rule.guard(s, action)
    assert g.ok
    child = rule.apply(s, action)
    # Apply guard's excluded into the child for verifier
    child = child.with_excluded(*g.new_excluded)
    assert _verify(s, child) == VERIFY_PASS


def test_cross_multiply_skips_when_no_denominators():
    s = EqState.from_strings("x + 1", "5")
    rule = CrossMultiply()
    actions = list(rule.enumerate(s))
    assert actions == []


def test_cross_multiply_guard_propagates_excluded():
    s = EqState.from_strings("1/(x - 3)", "1/(x + 2)")
    rule = CrossMultiply()
    g = rule.guard(s, Action(rule.name))
    assert g.ok
    excluded_set = set(g.new_excluded)
    assert sp.Integer(3) in excluded_set or sp.Integer(-2) in excluded_set


# --- COMBINE_FRACTIONS_AT -------------------------------------------------


def test_combine_fractions_apply():
    # 1/(x+1) + 2/(x+2) → ((x+2) + 2*(x+1)) / ((x+1)(x+2))
    s = EqState.from_strings("1/(x + 1) + 2/(x + 2)", "0")
    rule = CombineFractionsAt()
    actions = list(rule.enumerate(s))
    assert len(actions) >= 1
    # Find action targeting the lhs Add at root
    a = next((a for a in actions if a.target_side == "lhs"), None)
    assert a is not None
    g = rule.guard(s, a)
    assert g.ok
    child = rule.apply(s, a).with_excluded(*g.new_excluded)
    assert _verify(s, child) == VERIFY_PASS


def test_combine_fractions_skips_when_no_fractions():
    s = EqState.from_strings("x + 1", "0")
    rule = CombineFractionsAt()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs"]
    assert actions == []


# --- SPLIT_FRACTION_AT ----------------------------------------------------


def test_split_fraction_apply():
    # (x + 3)/2: targets the Mul((x+3), Pow(2,-1)).
    inner = sp.Add(x, sp.Integer(3), evaluate=False)
    lhs = sp.Mul(inner, sp.Pow(sp.Integer(2), sp.Integer(-1), evaluate=False), evaluate=False)
    s = EqState(lhs=lhs, rhs=sp.Integer(0), var=x)
    rule = SplitFractionAt()
    actions = list(rule.enumerate(s))
    a = next((a for a in actions if a.target_side == "lhs" and a.target_path == ()), None)
    assert a is not None
    g = rule.guard(s, a)
    assert g.ok
    child = rule.apply(s, a).with_excluded(*g.new_excluded)
    assert _verify(s, child) == VERIFY_PASS


def test_split_fraction_skips_non_fraction():
    s = EqState.from_strings("x + 1", "0")
    rule = SplitFractionAt()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs"]
    assert actions == []


# --- COMMON_DENOMINATOR_AT ------------------------------------------------


def test_common_denominator_apply():
    s = EqState.from_strings("1/(x + 1) + 2/(x + 2)", "0")
    rule = CommonDenominatorAt()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs"]
    assert len(actions) >= 1
    a = actions[0]
    g = rule.guard(s, a)
    assert g.ok
    child = rule.apply(s, a).with_excluded(*g.new_excluded)
    assert _verify(s, child) == VERIFY_PASS


def test_common_denominator_skips_when_single_denom():
    s = EqState.from_strings("1/(x + 1) + 2/(x + 1)", "0")
    rule = CommonDenominatorAt()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs"]
    assert actions == []


# --- SIMPLIFY_AT ----------------------------------------------------------


def test_simplify_at_apply():
    # x*(x + 1) - x simplifies to x*(x+1) - x = x² + x - x = x²
    inner = sp.Add(
        sp.Mul(x, sp.Add(x, sp.Integer(1), evaluate=False), evaluate=False),
        sp.Mul(sp.Integer(-1), x, evaluate=False),
        evaluate=False,
    )
    s = EqState(lhs=inner, rhs=sp.Integer(0), var=x)
    rule = SimplifyAt()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs" and a.target_path == ()]
    assert len(actions) >= 1
    child = rule.apply(s, actions[0])
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_simplify_at_skips_when_no_change():
    s = EqState.from_strings("x", "5")
    rule = SimplifyAt()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs" and a.target_path == ()]
    # x is already simplified
    assert actions == []


# --- PARTIAL_FRACTIONS ----------------------------------------------------


def test_partial_fractions_apply():
    # 1/((x-1)(x+1)) → 1/(2(x-1)) - 1/(2(x+1)) (sp.apart canonical)
    s = EqState.from_strings("1/((x - 1)*(x + 1))", "0")
    rule = PartialFractions()
    actions = list(rule.enumerate(s))
    a = next((a for a in actions if a.target_side == "lhs"), None)
    assert a is not None
    g = rule.guard(s, a)
    assert g.ok
    child = rule.apply(s, a).with_excluded(*g.new_excluded)
    assert _verify(s, child) == VERIFY_PASS


def test_partial_fractions_skips_when_no_denominator():
    s = EqState.from_strings("x + 1", "5")
    rule = PartialFractions()
    actions = list(rule.enumerate(s))
    assert actions == []
