"""Tests for Phase 1b algebra rules (21-27)."""

from __future__ import annotations

import sympy as sp

from ggmr.rules.core import *  # noqa: F401,F403  (registers)
from ggmr.rules.core.algebra import (
    CollectLikeVariableTermsAt,
    DistributeNegativeAt,
    DoubleNegationAt,
    FactorOutGcfAt,
    IdentityAddZeroAt,
    IdentityMulOneAt,
    ZeroPropertyAt,
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


# --- FACTOR_OUT_GCF_AT ----------------------------------------------------


def test_factor_out_gcf_apply():
    # 2*x + 4 → 2*(x + 2). Build structurally.
    lhs = sp.Add(sp.Mul(sp.Integer(2), x, evaluate=False), sp.Integer(4), evaluate=False)
    s = EqState(lhs=lhs, rhs=sp.Integer(0), var=x)
    rule = FactorOutGcfAt()
    actions = list(rule.enumerate(s))
    assert len(actions) >= 1
    # Find action targeting the lhs Add (path=())
    a = next(a for a in actions if a.target_side == "lhs" and a.target_path == ())
    child = rule.apply(s, a)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_factor_out_gcf_skips_when_gcf_is_one():
    lhs = sp.Add(sp.Mul(sp.Integer(2), x, evaluate=False), sp.Integer(3), evaluate=False)
    s = EqState(lhs=lhs, rhs=sp.Integer(0), var=x)
    rule = FactorOutGcfAt()
    # gcd(2, 3) = 1; should not enumerate at the root Add path
    actions = list(rule.enumerate(s))
    assert all(not (a.target_side == "lhs" and a.target_path == ()) for a in actions)


def test_factor_out_gcf_handles_three_term_add():
    # 6*x + 4*x**2 + 8 → 2*(3*x + 2*x**2 + 4)
    lhs = sp.Add(
        sp.Mul(sp.Integer(6), x, evaluate=False),
        sp.Mul(sp.Integer(4), sp.Pow(x, 2, evaluate=False), evaluate=False),
        sp.Integer(8),
        evaluate=False,
    )
    s = EqState(lhs=lhs, rhs=sp.Integer(0), var=x)
    rule = FactorOutGcfAt()
    actions = list(rule.enumerate(s))
    a = next(a for a in actions if a.target_side == "lhs" and a.target_path == ())
    child = rule.apply(s, a)
    assert _verify(s, child) == VERIFY_PASS


# --- COLLECT_LIKE_VARIABLE_TERMS_AT ---------------------------------------


def test_collect_like_variable_terms_apply():
    # 2*x + 3*x + 5 → 5*x + 5 (post-collect)
    lhs = sp.Add(
        sp.Mul(sp.Integer(2), x, evaluate=False),
        sp.Mul(sp.Integer(3), x, evaluate=False),
        sp.Integer(5),
        evaluate=False,
    )
    s = EqState(lhs=lhs, rhs=sp.Integer(0), var=x)
    rule = CollectLikeVariableTermsAt()
    actions = list(rule.enumerate(s))
    assert len(actions) >= 1
    a = next(a for a in actions if a.target_side == "lhs" and a.target_path == ())
    child = rule.apply(s, a)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_collect_like_variable_terms_skips_when_no_change():
    s = EqState.from_strings("x + 1", "0")
    rule = CollectLikeVariableTermsAt()
    actions = list(rule.enumerate(s))
    # No like terms to collect
    assert all(not (a.target_side == "lhs" and a.target_path == ()) for a in actions)


# --- DISTRIBUTE_NEGATIVE_AT -----------------------------------------------


def test_distribute_negative_apply():
    # -(x + 3) → -x + -3
    inner = sp.Add(x, sp.Integer(3), evaluate=False)
    lhs = sp.Mul(sp.Integer(-1), inner, evaluate=False)
    s = EqState(lhs=lhs, rhs=sp.Integer(0), var=x)
    rule = DistributeNegativeAt()
    actions = list(rule.enumerate(s))
    a = next(a for a in actions if a.target_side == "lhs" and a.target_path == ())
    child = rule.apply(s, a)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_distribute_negative_skips_when_no_neg_one_mul():
    s = EqState.from_strings("x + 3", "0")
    rule = DistributeNegativeAt()
    assert list(rule.enumerate(s)) == []


# --- IDENTITY_ADD_ZERO_AT -------------------------------------------------


def test_identity_add_zero_apply():
    lhs = sp.Add(x, sp.Integer(0), evaluate=False)
    s = EqState(lhs=lhs, rhs=sp.Integer(5), var=x)
    rule = IdentityAddZeroAt()
    actions = list(rule.enumerate(s))
    a = next(a for a in actions if a.target_side == "lhs" and a.target_path == ())
    child = rule.apply(s, a)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_identity_add_zero_skips_when_no_zero():
    s = EqState.from_strings("x + 1", "0")
    rule = IdentityAddZeroAt()
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs" and a.target_path == ()]
    assert actions == []


# --- IDENTITY_MUL_ONE_AT --------------------------------------------------


def test_identity_mul_one_apply():
    lhs = sp.Mul(x, sp.Integer(1), evaluate=False)
    s = EqState(lhs=lhs, rhs=sp.Integer(2), var=x)
    rule = IdentityMulOneAt()
    actions = list(rule.enumerate(s))
    a = next(a for a in actions if a.target_side == "lhs" and a.target_path == ())
    child = rule.apply(s, a)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_identity_mul_one_skips_when_no_one():
    s = EqState.from_strings("2*x", "4")
    rule = IdentityMulOneAt()
    # 2*x has no *1 child
    actions = [a for a in rule.enumerate(s) if a.target_side == "lhs" and a.target_path == ()]
    assert actions == []


# --- ZERO_PROPERTY_AT -----------------------------------------------------


def test_zero_property_apply():
    # 0*x + 3 = 5: the Mul(0, x) collapses to 0.
    inner = sp.Mul(sp.Integer(0), x, evaluate=False)
    lhs = sp.Add(inner, sp.Integer(3), evaluate=False)
    s = EqState(lhs=lhs, rhs=sp.Integer(5), var=x)
    rule = ZeroPropertyAt()
    actions = list(rule.enumerate(s))
    # Should target the inner Mul at path (0,)
    a = next(a for a in actions if a.target_side == "lhs" and a.target_path == (0,))
    child = rule.apply(s, a)
    # Effective solset preserved (0*x + 3 == 3, child has 0 + 3 == 3, both unsolvable since 5 != 3)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_zero_property_skips_when_no_zero_factor():
    s = EqState.from_strings("2*x", "4")
    rule = ZeroPropertyAt()
    actions = list(rule.enumerate(s))
    assert actions == []


# --- DOUBLE_NEGATION_AT ---------------------------------------------------


def test_double_negation_apply():
    # -(-x) → x. Build Mul(-1, Mul(-1, x)).
    inner = sp.Mul(sp.Integer(-1), x, evaluate=False)
    lhs = sp.Mul(sp.Integer(-1), inner, evaluate=False)
    s = EqState(lhs=lhs, rhs=sp.Integer(2), var=x)
    rule = DoubleNegationAt()
    actions = list(rule.enumerate(s))
    a = next(a for a in actions if a.target_side == "lhs" and a.target_path == ())
    child = rule.apply(s, a)
    assert _solset(child) == _solset(s)
    assert _verify(s, child) == VERIFY_PASS


def test_double_negation_skips_single_negation():
    lhs = sp.Mul(sp.Integer(-1), x, evaluate=False)
    s = EqState(lhs=lhs, rhs=sp.Integer(-2), var=x)
    rule = DoubleNegationAt()
    actions = list(rule.enumerate(s))
    # No double negation present
    assert all(not (a.target_side == "lhs" and a.target_path == ()) for a in actions)
