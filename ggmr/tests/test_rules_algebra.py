"""Tests for algebra rules: distribute, expand_product, expand_power, combine_like_terms_at."""

from __future__ import annotations

import sympy as sp

from ggmr.expr.tree import canonical_repr
from ggmr.rules.core import *  # noqa: F401,F403
from ggmr.rules.core.algebra import (
    CombineLikeTermsAt,
    DistributeOverSubtree,
    ExpandPower,
    ExpandProduct,
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


# --- DISTRIBUTE_OVER_SUBTREE -------------------------------------------


def test_distribute_3_times_x_minus_2():
    """3*(x - 2) -> 3*x - 6"""
    s = EqState.from_strings("3*(x - 2)", "0")
    rule = DistributeOverSubtree()
    actions = list(rule.enumerate(s))
    assert any(a.target_side == "lhs" for a in actions)
    a = next(a for a in actions if a.target_side == "lhs" and a.target_path == ())
    child = rule.apply(s, a)
    assert child.solution_set() == s.solution_set()
    assert _verify(s, child) == VERIFY_PASS


# --- EXPAND_PRODUCT -----------------------------------------------------


def test_expand_product_two_binomials():
    """(x - 1)*(x - 2) -> x^2 - 3x + 2 (after expansion)"""
    s = EqState.from_strings("(x - 1)*(x - 2)", "0")
    rule = ExpandProduct()
    actions = list(rule.enumerate(s))
    assert actions, "EXPAND_PRODUCT should enumerate (x-1)*(x-2)"
    child = rule.apply(s, actions[0])
    # Expanded form has the same solution set
    assert child.solution_set() == s.solution_set()
    assert _verify(s, child) == VERIFY_PASS


# --- EXPAND_POWER -----------------------------------------------------


def test_expand_power_of_binomial():
    """(x + 1)^2 -> x^2 + 2x + 1"""
    s = EqState.from_strings("(x + 1)**2", "4")
    rule = ExpandPower()
    actions = list(rule.enumerate(s))
    assert actions
    child = rule.apply(s, actions[0])
    assert child.solution_set() == s.solution_set()
    assert _verify(s, child) == VERIFY_PASS


def test_expand_power_skips_non_binomial_base():
    """Pow(Symbol('x'), 2) is `x^2` — has no Add base, so EXPAND_POWER does not enumerate."""
    s = EqState.from_strings("x**2", "4")
    rule = ExpandPower()
    actions = list(rule.enumerate(s))
    assert not actions


# --- COMBINE_LIKE_TERMS_AT --------------------------------------------


def test_combine_like_terms_2x_plus_3x():
    """2x + 3x -> 5x"""
    # Construct directly to avoid parse_expr's auto-canonicalization
    lhs = sp.Add(
        sp.Mul(sp.Integer(2), x, evaluate=False),
        sp.Mul(sp.Integer(3), x, evaluate=False),
        evaluate=False,
    )
    s = EqState(lhs=lhs, rhs=sp.Integer(10), var=x)
    rule = CombineLikeTermsAt()
    actions = list(rule.enumerate(s))
    assert any(a.target_side == "lhs" for a in actions)
    a = next(a for a in actions if a.target_side == "lhs")
    child = rule.apply(s, a)
    assert child.solution_set() == s.solution_set()
    assert _verify(s, child) == VERIFY_PASS
