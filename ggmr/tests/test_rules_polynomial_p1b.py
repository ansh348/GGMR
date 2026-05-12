"""Tests for Phase 1b polynomial rules (38-42).

Note: rule 42 is POLY_TO_MONIC, replacing the originally-planned
AUXILIARY_VARIABLE_SUBSTITUTION which was deferred to Phase 1c (multi-variable
state semantics are out of scope for 1b).
"""

from __future__ import annotations

import sympy as sp

from ggmr.rules.core import *  # noqa: F401,F403  (registers)
from ggmr.rules.core.polynomial import (
    PolyToMonic,
    PolynomialLongDivision,
    RationalRootTheorem,
    SyntheticDivision,
    VietasFormulas,
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


# --- POLYNOMIAL_LONG_DIVISION ---------------------------------------------


def test_polynomial_long_division_apply():
    # x³ - 6x² + 11x - 6 ÷ (x - 1) cleanly = x² - 5x + 6
    s = EqState.from_strings("x**3 - 6*x**2 + 11*x - 6", "0")
    rule = PolynomialLongDivision()
    divisor = x - sp.Integer(1)
    action = Action(rule.name, params=(divisor,), target_side="lhs")
    g = rule.guard(s, action)
    assert g.ok
    child = rule.apply(s, action)
    assert _verify(s, child) == VERIFY_PASS


def test_polynomial_long_division_skips_non_polynomial():
    s = EqState.from_strings("1/(x - 1)", "0")
    rule = PolynomialLongDivision()
    actions = list(rule.enumerate(s))
    assert actions == []


def test_polynomial_long_division_guard_rejects_zero_divisor():
    s = EqState.from_strings("x**2 + 1", "0")
    rule = PolynomialLongDivision()
    action = Action(rule.name, params=(sp.Integer(0),), target_side="lhs")
    g = rule.guard(s, action)
    assert g.ok is False


# --- SYNTHETIC_DIVISION ---------------------------------------------------


def test_synthetic_division_apply():
    # x³ - 6x² + 11x - 6 has root x=1 → factors as (x-1)(x² - 5x + 6)
    s = EqState.from_strings("x**3 - 6*x**2 + 11*x - 6", "0")
    rule = SyntheticDivision()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs"]
    # Should find root 1, 2, or 3
    assert len(actions) >= 1
    a = next(a for a in actions if a.params == (sp.Integer(1),))
    child = rule.apply(s, a)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_synthetic_division_skips_when_no_clean_root():
    # x² + 1 has no integer roots in [-5, 5]
    s = EqState.from_strings("x**2 + 1", "0")
    rule = SyntheticDivision()
    actions = list(rule.enumerate(s))
    assert actions == []


# --- RATIONAL_ROOT_THEOREM ------------------------------------------------


def test_rational_root_theorem_apply():
    s = EqState.from_strings("x**3 - 6*x**2 + 11*x - 6", "0")
    rule = RationalRootTheorem()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs"]
    assert len(actions) >= 1
    child = rule.apply(s, actions[0])
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_rational_root_theorem_skips_when_no_rational_roots():
    # x² + 1 has no real roots, hence no rational
    s = EqState.from_strings("x**2 + 1", "0")
    rule = RationalRootTheorem()
    actions = list(rule.enumerate(s))
    assert actions == []


# --- VIETAS_FORMULAS ------------------------------------------------------


def test_vietas_formulas_records_side_conditions():
    # (x - 2)(x - 3) = 0 → roots 2, 3 → vieta_sum=5, vieta_prod=6
    lhs = sp.Mul(
        sp.Add(x, sp.Integer(-2), evaluate=False),
        sp.Add(x, sp.Integer(-3), evaluate=False),
        evaluate=False,
    )
    s = EqState(lhs=lhs, rhs=sp.Integer(0), var=x)
    rule = VietasFormulas()
    actions = list(rule.enumerate(s))
    assert len(actions) >= 1
    child = rule.apply(s, actions[0])
    # Side conditions added
    assert len(child.side_conditions) >= 2
    # Equation itself unchanged → solset preserved
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_vietas_formulas_skips_unfactored():
    s = EqState.from_strings("x**2 + 1", "0")
    rule = VietasFormulas()
    actions = list(rule.enumerate(s))
    assert actions == []


# --- POLY_TO_MONIC --------------------------------------------------------


def test_poly_to_monic_apply():
    # 2x² + 4x + 6 = 0 → x² + 2x + 3 = 0 (divide both sides by 2)
    s = EqState.from_strings("2*x**2 + 4*x + 6", "0")
    rule = PolyToMonic()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs"]
    assert len(actions) >= 1
    child = rule.apply(s, actions[0])
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_poly_to_monic_skips_when_already_monic():
    s = EqState.from_strings("x**2 + 5*x + 6", "0")
    rule = PolyToMonic()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs"]
    assert actions == []


def test_poly_to_monic_skips_non_polynomial():
    s = EqState.from_strings("1/(x - 1)", "0")
    rule = PolyToMonic()
    actions = list(rule.enumerate(s))
    assert actions == []
