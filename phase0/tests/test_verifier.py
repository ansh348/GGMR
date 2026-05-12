"""Tests for step-legality verifier."""

import pytest
import sympy as sp
from sympy import Eq, Symbol
from sympy.parsing.sympy_parser import parse_expr

from phase0.src.trace_loader import Problem, Step
from phase0.src.verifier import (
    IllegalStepError,
    verify_all,
    verify_step,
)


x = Symbol("x")


def _p(s: str):
    return parse_expr(s, local_dict={"x": x}, evaluate=False)


def _eq(lhs: str, rhs: str) -> Eq:
    return Eq(_p(lhs), _p(rhs), evaluate=False)


def test_legal_subtract_step():
    eq_t = _eq("2*x + 3", "7")
    eq_n = _eq("2*x", "4")
    ok, _ = verify_step(eq_t, eq_n, x)
    assert ok


def test_legal_divide_step():
    eq_t = _eq("2*x", "4")
    eq_n = _eq("x", "2")
    ok, _ = verify_step(eq_t, eq_n, x)
    assert ok


def test_illegal_step_changes_solution():
    eq_t = _eq("2*x + 3", "7")  # x = 2
    eq_n = _eq("2*x", "5")  # x = 5/2
    ok, reason = verify_step(eq_t, eq_n, x)
    assert not ok
    assert "differ" in reason.lower()


def test_legal_cancellation_subset_allowed():
    # (x^2 - 1)/(x - 1) has solution {2 only via constraint}; but solve treats it as
    # full simplification. Original eq solve gives [2]; cancelled gives [2]. Both same.
    eq_t = _eq("(x**2 - 1)/(x - 1)", "3")
    eq_n = _eq("x + 1", "3")
    ok, _ = verify_step(eq_t, eq_n, x)
    assert ok


def test_verify_all_strict_raises():
    p = Problem(
        id="bogus01",
        category="linear",
        variable=x,
        source="test fixture",
        initial=_eq("2*x + 3", "7"),
        canonical_target=_eq("x", "5"),  # wrong target — but verify_all checks step legality, not target
        trace=(
            Step(rule="WRONG", eq=_eq("2*x", "5")),  # illegal: introduces x = 5/2
            Step(rule="WRONG", eq=_eq("x", "5")),
        ),
    )
    with pytest.raises(IllegalStepError) as excinfo:
        verify_all([p], strict=True)
    assert excinfo.value.problem_id == "bogus01"
    assert excinfo.value.step_idx == 0


def test_verify_all_legal_passes():
    p = Problem(
        id="ok01",
        category="linear",
        variable=x,
        source="test fixture",
        initial=_eq("2*x + 3", "7"),
        canonical_target=_eq("x", "2"),
        trace=(
            Step(rule="SUB3", eq=_eq("2*x", "4")),
            Step(rule="DIV2", eq=_eq("x", "2")),
        ),
    )
    checks = verify_all([p], strict=True)
    assert all(c.ok for c in checks)
    assert len(checks) == 2
