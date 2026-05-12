"""Tests for FACTOR_POLYNOMIAL."""

from __future__ import annotations

import sympy as sp

from ggmr.expr.tree import canonical_repr, normalize
from ggmr.rules.core import *  # noqa: F401,F403
from ggmr.rules.core.polynomial import FactorPolynomial
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


def test_factor_quadratic_trinomial():
    """x^2 - 5x + 6 = 0 -> (x-2)(x-3) = 0"""
    s = EqState.from_strings("x**2 - 5*x + 6", "0")
    rule = FactorPolynomial()
    actions = list(rule.enumerate(s))
    assert any(a.target_side == "lhs" for a in actions)
    a = next(a for a in actions if a.target_side == "lhs" and a.target_path == ())
    child = rule.apply(s, a)
    assert child.solution_set() == s.solution_set()
    assert _verify(s, child) == VERIFY_PASS


def test_factor_cubic():
    """x^3 - 6x^2 + 11x - 6 = 0 -> (x-1)(x-2)(x-3) = 0 (one shot)"""
    s = EqState.from_strings("x**3 - 6*x**2 + 11*x - 6", "0")
    rule = FactorPolynomial()
    actions = list(rule.enumerate(s))
    assert actions
    child = rule.apply(s, actions[0])
    assert child.solution_set() == s.solution_set()
    assert _verify(s, child) == VERIFY_PASS


def test_factor_quartic_via_substitution():
    """x^4 - 5x^2 + 4 = 0 -> (x-1)(x+1)(x-2)(x+2) = 0 via SymPy.factor"""
    s = EqState.from_strings("x**4 - 5*x**2 + 4", "0")
    rule = FactorPolynomial()
    actions = list(rule.enumerate(s))
    assert actions
    child = rule.apply(s, actions[0])
    assert child.solution_set() == s.solution_set()
    assert _verify(s, child) == VERIFY_PASS


def test_factor_skips_already_factored():
    """`(x-1)*(x-2)` is already factored — no enumerate."""
    s = EqState(
        lhs=sp.Mul(sp.Add(x, -1, evaluate=False), sp.Add(x, -2, evaluate=False), evaluate=False),
        rhs=sp.Integer(0),
        var=x,
    )
    rule = FactorPolynomial()
    actions = list(rule.enumerate(s))
    # Some entries may still appear at deeper subtree paths (e.g., on (x-1) itself which is degree-1)
    # but lhs path () should not be enumerated since it's already factored
    root_actions = [a for a in actions if a.target_path == () and a.target_side == "lhs"]
    assert not root_actions
