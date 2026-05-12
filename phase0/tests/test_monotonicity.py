"""Tests for monotonicity rate computation."""

import pytest

from phase0.src.monotonicity import (
    decision,
    step_rate,
    variant_sigma,
)


def test_step_rate_strictly_decreasing():
    assert step_rate([5, 4, 3, 2, 1]) == 1.0


def test_step_rate_strictly_increasing():
    assert step_rate([1, 2, 3, 4, 5]) == 0.0


def test_step_rate_plateau_counts_as_monotone_nonstrict():
    assert step_rate([3, 3, 3]) == 1.0


def test_step_rate_plateau_zero_strict():
    assert step_rate([3, 3, 3], strict=True) == 0.0


def test_step_rate_mixed():
    # 4 transitions: 5→4 (decrease), 4→6 (increase), 6→3 (decrease), 3→3 (plateau, nonstrict ok)
    assert step_rate([5, 4, 6, 3, 3]) == pytest.approx(3 / 4)


def test_step_rate_singleton_vacuous():
    assert step_rate([5]) == 1.0
    assert step_rate([]) == 1.0


def test_decision_high():
    assert decision(0.85) == "PHASE_A_MEANINGFUL_BASELINE"


def test_decision_borderline_high_conservative():
    # 0.81 < 0.80 + 0.02 = 0.82 → conservative side: weak
    assert decision(0.81) == "PHASE_A_WEAK_NONMYOPIC_PRIMARY"


def test_decision_clearly_meaningful():
    # 0.83 > 0.82 → meaningful baseline
    assert decision(0.83) == "PHASE_A_MEANINGFUL_BASELINE"


def test_decision_mid():
    assert decision(0.65) == "PHASE_A_WEAK_NONMYOPIC_PRIMARY"


def test_decision_borderline_low_conservative():
    # 0.51 < 0.50 + 0.02 = 0.52 → conservative side: negative
    assert decision(0.51) == "PHASE_A_NEGATIVE_RESULT"


def test_decision_low():
    assert decision(0.30) == "PHASE_A_NEGATIVE_RESULT"


def test_variant_sigma_zero_when_all_equal():
    assert variant_sigma({"a": 0.5, "b": 0.5, "c": 0.5}) == pytest.approx(0.0)


def test_variant_sigma_positive_when_varied():
    assert variant_sigma({"a": 0.3, "b": 0.7}) > 0


def test_variant_sigma_too_small():
    assert variant_sigma({"a": 0.5}) == 0.0
    assert variant_sigma({}) == 0.0
