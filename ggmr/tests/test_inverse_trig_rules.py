"""Smoke tests for the trigonometry inverse rule registry (Phase 1.2a).

`trig_inverse_registry` is used by `TrigReverseGenerator` to manufacture
problems by reverse construction. Every inverse rule must grow tree size
strictly (no-op inverses defeat the generator).
"""

from __future__ import annotations

import random

import pytest
import sympy as sp

import ggmr.rules.core  # noqa: F401  (register forward rules)
from ggmr.problems.inverse_trig_rules import (
    InvPythagoreanIntroOne,
    trig_inverse_registry,
)
from ggmr.state import EqState


def _count_ops(state: EqState) -> int:
    """Total tree-size proxy: count_ops(lhs) + count_ops(rhs)."""
    return int(state.lhs.count_ops()) + int(state.rhs.count_ops())


def test_registry_total_count():
    """v2: 13 explicit inverse rules + 11 wrapped forward expanders = 24.

    Explicit rules: InvPythagoreanIntroOne, InvOneToSecTan, InvOneToCscCot,
    InvTanToSinCos, InvCotToCosSin, InvSecToInvCos, InvCscToInvSin,
    InvSin2ToOneMinusCos2, InvCos2ToOneMinusSin2, InvSinToParity,
    InvCosToParity, InvSinToCofunction, InvCosToCofunction.
    """
    n = len(trig_inverse_registry.all_rules())
    assert n == 24, f"expected 24 inverse rules, got {n}"


def test_registry_names_unique():
    names = [r.name for r in trig_inverse_registry.all_rules()]
    assert len(names) == len(set(names)), f"duplicate inverse names: {names}"


def test_inverse_names_all_have_INV_prefix():
    for rule in trig_inverse_registry.all_rules():
        assert rule.name.startswith("INV_"), f"bad name: {rule.name}"


def test_pythagorean_intro_one_replaces_one_on_rhs():
    state = EqState.from_strings("sin(x)**2 + cos(x)**2", "1")
    rule = InvPythagoreanIntroOne()
    rng = random.Random(0)
    actions = list(rule.enumerate(state, rng))
    assert actions, "no actions enumerated despite RHS = 1"
    new_state = rule.apply(state, actions[0])
    # The `1` literal on RHS should be replaced; the new RHS should contain
    # both sin and cos function calls.
    assert new_state.rhs.has(sp.sin) and new_state.rhs.has(sp.cos), \
        f"replacement missing trig calls: {new_state.rhs}"


def test_pythagorean_intro_one_skips_when_no_one_literal():
    """When neither side has Integer(1), enumerate yields nothing."""
    state = EqState.from_strings("sin(x)", "cos(x)")  # no 1 literal
    rule = InvPythagoreanIntroOne()
    rng = random.Random(0)
    assert list(rule.enumerate(state, rng)) == []


def test_each_wrapped_inverse_grows_state_on_a_compatible_seed():
    """Per-inverse-rule sanity check: each wrapped inverse, when applied to a
    state where its forward rule would fire, increases tree size strictly.

    We pick a state tailored to each rule's enumerate predicate.
    """
    seeds_for = {
        # 11 wrapped forward expanders
        "INV_SIN_SUM":                ("sin(x + y)", "0"),
        "INV_COS_SUM":                ("cos(x + y)", "0"),
        "INV_TAN_SUM":                ("tan(x + y)", "0"),
        "INV_SIN_DIFF":               ("sin(x - y)", "0"),
        "INV_COS_DIFF":               ("cos(x - y)", "0"),
        "INV_SIN_DOUBLE":             ("sin(2*x)", "0"),
        "INV_COS_DOUBLE":             ("cos(2*x)", "0"),
        "INV_SIN_SQUARED_HALF_ANGLE": ("sin(x)**2", "0"),
        "INV_COS_SQUARED_HALF_ANGLE": ("cos(x)**2", "0"),
        "INV_PROD_SIN_COS":           ("sin(x)*cos(y)", "0"),
        "INV_SUM_SIN_TO_PROD":        ("sin(x) + sin(y)", "0"),
        # 12 explicit v2 inverse rules
        "INV_ONE_TO_SEC_TAN":         ("1 + sin(x)", "0"),
        "INV_ONE_TO_CSC_COT":         ("1 + sin(x)", "0"),
        "INV_TAN_TO_SIN_COS":         ("tan(x)", "0"),
        "INV_COT_TO_COS_SIN":         ("cot(x)", "0"),
        "INV_SEC_TO_INV_COS":         ("sec(x)", "0"),
        "INV_CSC_TO_INV_SIN":         ("csc(x)", "0"),
        "INV_SIN2_TO_ONE_MINUS_COS2": ("sin(x)**2", "0"),
        "INV_COS2_TO_ONE_MINUS_SIN2": ("cos(x)**2", "0"),
        "INV_SIN_TO_PARITY":          ("sin(x)", "0"),
        "INV_COS_TO_PARITY":          ("cos(x)", "0"),
        "INV_SIN_TO_COFUNCTION":      ("sin(x)", "0"),
        "INV_COS_TO_COFUNCTION":      ("cos(x)", "0"),
    }
    rng = random.Random(42)
    missing_seeds = []
    for rule in trig_inverse_registry.all_rules():
        if rule.name == "INV_PYTHAGOREAN_INTRO_ONE":
            continue  # tested separately above
        if rule.name not in seeds_for:
            missing_seeds.append(rule.name)
            continue
        lhs, rhs = seeds_for[rule.name]
        state = EqState.from_strings(lhs, rhs)
        actions = list(rule.enumerate(state, rng))
        if not actions:
            pytest.fail(f"{rule.name}: no actions on compatible seed {lhs} = {rhs}")
        new_state = rule.apply(state, actions[0])
        before = _count_ops(state)
        after = _count_ops(new_state)
        assert after > before, (
            f"{rule.name}: tree size did not grow "
            f"({before} -> {after}); state: {state} -> {new_state}"
        )
    assert not missing_seeds, (
        f"new inverse rules lack test seeds: {missing_seeds}. "
        f"Add to `seeds_for` to exercise them."
    )


def test_inverse_rules_inert_on_pure_algebra_state():
    """Trig inverse rules must not fire on algebra-only states (2x+3=7)."""
    state = EqState.from_strings("2*x + 3", "7")
    rng = random.Random(0)
    for rule in trig_inverse_registry.all_rules():
        # InvPythagoreanIntroOne CAN fire here (RHS contains `7` not `1`,
        # but the LHS contains `3` not `1`, so should skip). Most should skip.
        actions = list(rule.enumerate(state, rng))
        # Allow InvPythagoreanIntroOne to skip cleanly (no `1` in algebra state);
        # all wrapped trig forwards must also skip due to _has_trig_in_state.
        if rule.name == "INV_PYTHAGOREAN_INTRO_ONE":
            assert actions == [], "InvPythagoreanIntroOne fired on algebra state"
        else:
            assert actions == [], (
                f"{rule.name} fired on algebra-only state — trig forward should skip"
            )
