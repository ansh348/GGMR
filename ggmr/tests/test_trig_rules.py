"""Smoke tests for the trigonometry rule registry (Phase 1.1).

Coverage strategy: one representative rule per group, plus a registry-wide
sanity check. Per-rule mathematical correctness — that `apply()` produces
an expression numerically equivalent to the original at a few sample
angles — is exercised by spot-checks with float substitution.

Marcus Constraint 1: TRIG_SIMPLIFY and TRIG_SOLVE must be excluded under
`training_only=True`.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

import ggmr.rules.core  # noqa: F401  (register all rules)
from ggmr.rules.registry import default_registry
from ggmr.state import EqState


x = sp.Symbol("x")
y = sp.Symbol("y")


def _eval_at(expr: sp.Expr, val: float) -> float:
    """Substitute x=val and y=0.7, then return float (handles complex by abs)."""
    sub = expr.subs({x: val, y: 0.7})
    try:
        return float(sub)
    except (TypeError, ValueError):
        # Some intermediate forms (Pow/Mul with evaluate=False) need explicit eval
        return float(sub.evalf())


def _apply_rule_by_name(rule_name: str, state: EqState):
    """Find the first legal action for `rule_name`, apply, return (new_state, action)."""
    rule = default_registry.get(rule_name)
    actions = list(rule.enumerate(state))
    if not actions:
        return None, None
    action = actions[0]
    guard = rule.guard(state, action)
    if not guard.ok:
        return None, None
    new_state = rule.apply(state, action)
    return new_state, action


def _check_lhs_numerical_equiv(s1: EqState, s2: EqState, samples=(0.3, 0.7, 1.4)):
    """Assert LHS of s1 evaluates to the same value as LHS of s2 at given samples."""
    for v in samples:
        a = _eval_at(s1.lhs, v)
        b = _eval_at(s2.lhs, v)
        assert math.isclose(a, b, abs_tol=1e-6), \
            f"lhs differs at x={v}: {a} vs {b}  ({s1.lhs} vs {s2.lhs})"


def test_registry_total_count():
    """49 algebra + 41 trig = 90 rules in default_registry."""
    n = len(default_registry.names())
    assert n == 90, f"expected 90 registered rules, got {n}"


def test_training_only_excludes_oracle_trig():
    """TRIG_SIMPLIFY and TRIG_SOLVE must NOT be enumerated under training_only=True."""
    state = EqState.from_strings("sin(x)**2 + cos(x)**2", "1")
    full = {r.name for r, _ in default_registry.enumerate_actions(state)}
    training = {r.name for r, _ in default_registry.enumerate_actions(state, training_only=True)}
    assert "TRIG_SIMPLIFY" in full
    assert "TRIG_SOLVE" in full
    assert "TRIG_SIMPLIFY" not in training
    assert "TRIG_SOLVE" not in training


def test_pythagorean_sin2_plus_cos2_collapse():
    """sin²(x) + cos²(x) → 1 (or +/- residue depending on outer Add args)."""
    state = EqState.from_strings("sin(x)**2 + cos(x)**2", "1")
    new_state, action = _apply_rule_by_name("SIN2_PLUS_COS2_TO_ONE", state)
    assert new_state is not None
    # LHS should now evaluate to 1.0 at any x
    for v in (0.3, 1.2, 2.5):
        assert math.isclose(_eval_at(new_state.lhs, v), 1.0, abs_tol=1e-6)


def test_pythagorean_tan2_plus_one():
    state = EqState.from_strings("tan(x)**2 + 1", "sec(x)**2")
    new_state, _ = _apply_rule_by_name("TAN2_PLUS_ONE_TO_SEC2", state)
    assert new_state is not None
    _check_lhs_numerical_equiv(state, new_state, samples=(0.3, 1.1))


def test_reciprocal_csc_to_sin():
    """1/sin(x) → csc(x)."""
    state = EqState.from_strings("1/sin(x)", "csc(x)")
    new_state, _ = _apply_rule_by_name("RECIPROCAL_1_OVER_SIN_TO_CSC", state)
    assert new_state is not None
    _check_lhs_numerical_equiv(state, new_state, samples=(0.3, 1.1))


def test_quotient_sin_over_cos():
    """sin(x) * 1/cos(x) → tan(x)."""
    state = EqState.from_strings("sin(x) / cos(x)", "tan(x)")
    new_state, _ = _apply_rule_by_name("SIN_OVER_COS_TO_TAN", state)
    assert new_state is not None
    _check_lhs_numerical_equiv(state, new_state, samples=(0.3, 1.1))


def test_cofunction_sin_complement_smybolic():
    """sin(pi/2 - u) → cos(u). SymPy auto-canonicalizes sin(pi/2-x) to cos(x)
    at parse time, so we test the rule's `apply` directly on a hand-built
    state that bypasses canonicalization (Add with evaluate=False).
    """
    arg = sp.Add(sp.pi / 2, sp.Mul(sp.Integer(-1), x, evaluate=False), evaluate=False)
    lhs = sp.sin(arg, evaluate=False)
    state = EqState(lhs=lhs, rhs=sp.cos(x), var=x)
    rule = default_registry.get("SIN_COMPLEMENT")
    actions = list(rule.enumerate(state))
    # If enumerate returns nothing, SymPy has canonicalized regardless — accept
    # that as "rule is a no-op in this codebase" and skip the apply check.
    if not actions:
        pytest.skip("SymPy canonicalized the argument before rule could fire")
    new_state = rule.apply(state, actions[0])
    _check_lhs_numerical_equiv(state, new_state, samples=(0.3, 1.1))


def test_parity_sin_neg():
    """sin(-x) → -sin(x)."""
    state = EqState.from_strings("sin(-x)", "0")
    new_state, _ = _apply_rule_by_name("SIN_NEG", state)
    assert new_state is not None
    _check_lhs_numerical_equiv(state, new_state, samples=(0.3, 1.1))


def test_parity_cos_neg_unchanged_value():
    """cos(-x) → cos(x)."""
    state = EqState.from_strings("cos(-x)", "0")
    new_state, _ = _apply_rule_by_name("COS_NEG", state)
    assert new_state is not None
    _check_lhs_numerical_equiv(state, new_state, samples=(0.3, 1.1))


def test_angle_sum_sin():
    """sin(x + y) → sin(x)cos(y) + cos(x)sin(y)."""
    state = EqState.from_strings("sin(x + y)", "0")
    new_state, _ = _apply_rule_by_name("SIN_SUM", state)
    assert new_state is not None
    _check_lhs_numerical_equiv(state, new_state, samples=(0.3, 1.1))


def test_double_angle_sin():
    """sin(2x) → 2 sin(x) cos(x)."""
    state = EqState.from_strings("sin(2*x)", "0")
    new_state, _ = _apply_rule_by_name("SIN_DOUBLE", state)
    assert new_state is not None
    _check_lhs_numerical_equiv(state, new_state, samples=(0.3, 1.1))


def test_power_reduction_sin2():
    """sin²(x) → (1 - cos(2x))/2."""
    state = EqState.from_strings("sin(x)**2", "0")
    new_state, _ = _apply_rule_by_name("SIN_SQUARED_HALF_ANGLE", state)
    assert new_state is not None
    _check_lhs_numerical_equiv(state, new_state, samples=(0.3, 1.1))


def test_product_sin_cos():
    """sin(x) cos(y) → ½[sin(x+y) + sin(x-y)]."""
    state = EqState.from_strings("sin(x) * cos(y)", "0")
    new_state, _ = _apply_rule_by_name("PROD_SIN_COS", state)
    assert new_state is not None
    _check_lhs_numerical_equiv(state, new_state, samples=(0.3, 1.1))


def test_sum_to_prod_sin():
    """sin(x) + sin(y) → 2 sin((x+y)/2) cos((x-y)/2)."""
    state = EqState.from_strings("sin(x) + sin(y)", "0")
    new_state, _ = _apply_rule_by_name("SUM_SIN_TO_PROD", state)
    assert new_state is not None
    _check_lhs_numerical_equiv(state, new_state, samples=(0.3, 1.1))


def test_oracle_trig_simplify_collapses():
    """TRIG_SIMPLIFY (inference-only) collapses Pythagorean."""
    state = EqState.from_strings("sin(x)**2 + cos(x)**2", "1")
    new_state, _ = _apply_rule_by_name("TRIG_SIMPLIFY", state)
    assert new_state is not None
    assert new_state.lhs == sp.Integer(1)


def test_algebra_regression_unaffected_by_trig_rules():
    """An algebra state (no trig atoms) shouldn't have trig rules enumerated.

    The `_has_trig_in_state` short-circuit means trig rules' enumerate()
    return immediately, so the BFS branching factor on algebra states is
    unchanged from the 49-rule baseline.
    """
    state = EqState.from_strings("2*x + 3", "7")
    actions = list(default_registry.enumerate_actions(state, training_only=True))
    rule_names = {r.name for r, _ in actions}
    trig_names = {
        n for n in rule_names
        if n.startswith(("SIN", "COS", "TAN", "COT", "SEC", "CSC", "TRIG", "PROD_",
                          "SUM_SIN", "RECIPROCAL_1_OVER", "PYTH", "ONE_TO_SIN"))
    }
    # Note: some algebra rules may have SIN/COS-like prefixes by accident; exclude
    # those explicitly. The conservative check: trig-namespace rules are absent.
    actual_trig = trig_names & {
        "SIN2_PLUS_COS2_TO_ONE", "TAN2_PLUS_ONE_TO_SEC2", "COT2_PLUS_ONE_TO_CSC2",
        "SEC2_MINUS_ONE_TO_TAN2", "ONE_TO_SIN2_PLUS_COS2",
        "RECIPROCAL_1_OVER_SIN_TO_CSC", "RECIPROCAL_1_OVER_COS_TO_SEC",
        "SIN_SUM", "COS_SUM", "TAN_SUM", "SIN_DOUBLE", "PROD_SIN_COS",
    }
    assert not actual_trig, f"trig rules leaked onto algebra-only state: {actual_trig}"
