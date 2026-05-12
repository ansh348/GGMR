"""Tests for EqState: hashing, equality, AC-permutation stability, target detection."""

from __future__ import annotations

import sympy as sp

from ggmr.state import EqState
from ggmr.targets import is_canonical_target


x = sp.Symbol("x")


# --- AC-permutation hash stability --------------------------------------


def test_eqstate_hash_ac_permuted_add():
    """`x + 1 = 0` and `1 + x = 0` hash equal."""
    a = EqState(lhs=sp.Add(x, sp.Integer(1), evaluate=False), rhs=sp.Integer(0), var=x)
    b = EqState(lhs=sp.Add(sp.Integer(1), x, evaluate=False), rhs=sp.Integer(0), var=x)
    assert hash(a) == hash(b)
    assert a == b


def test_eqstate_hash_ac_permuted_mul():
    a = EqState(lhs=sp.Mul(sp.Integer(2), x, evaluate=False), rhs=sp.Integer(4), var=x)
    b = EqState(lhs=sp.Mul(x, sp.Integer(2), evaluate=False), rhs=sp.Integer(4), var=x)
    assert hash(a) == hash(b)


def test_eqstate_hash_distinguishes_structurally_different():
    a = EqState(lhs=sp.Add(x, sp.Integer(1), evaluate=False), rhs=sp.Integer(0), var=x)
    b = EqState(lhs=sp.Mul(x, sp.Integer(1), evaluate=False), rhs=sp.Integer(0), var=x)
    assert hash(a) != hash(b) or a != b


def test_eqstate_hash_includes_excluded():
    """Two states differing only in their `excluded` set must hash differently."""
    base = EqState.from_strings("x", "2")
    with_excluded = base.with_excluded(sp.Integer(1))
    assert hash(base) != hash(with_excluded)


# --- solution_set with excluded ----------------------------------------


def test_solution_set_subtracts_excluded():
    """`(x-1)(x-2) = 0` with excluded={1} has effective solution {2}."""
    s = EqState(
        lhs=sp.Mul(sp.Add(x, -1, evaluate=False), sp.Add(x, -2, evaluate=False), evaluate=False),
        rhs=sp.Integer(0),
        var=x,
        excluded=frozenset({sp.Integer(1)}),
    )
    assert s.solution_set() == frozenset({sp.Integer(2)})


# --- is_canonical_target ------------------------------------------------


def test_is_canonical_target_linear_x_eq_const():
    s = EqState(lhs=x, rhs=sp.Integer(2), var=x)
    assert s.is_canonical_target()


def test_is_canonical_target_const_eq_x_symmetric():
    s = EqState(lhs=sp.Integer(2), rhs=x, var=x)
    assert s.is_canonical_target()


def test_is_canonical_target_factored_zero_form():
    """(x - 1) * (x - 2) = 0"""
    factored = sp.Mul(
        sp.Add(x, -1, evaluate=False),
        sp.Add(x, -2, evaluate=False),
        evaluate=False,
    )
    s = EqState(lhs=factored, rhs=sp.Integer(0), var=x)
    assert s.is_canonical_target()


def test_is_canonical_target_rejects_unfactored_quadratic():
    """`x^2 - 5x + 6 = 0` is NOT a canonical target (must be factored)."""
    s = EqState.from_strings("x**2 - 5*x + 6", "0")
    assert not s.is_canonical_target()


def test_is_canonical_target_rejects_scaled_linear():
    """`2x - 4 = 0` should not be canonical — must be `x = 2` or `2*(x-2) = 0`."""
    s = EqState.from_strings("2*x - 4", "0")
    assert not s.is_canonical_target()
