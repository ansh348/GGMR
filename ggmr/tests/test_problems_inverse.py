"""Tests for inverse rules: each inverse application produces a state from
which forward BFS can recover.
"""

from __future__ import annotations

import random

import pytest
import sympy as sp

from ggmr.expr.tree import canonical_repr
from ggmr.problems.inverse_rules import (
    InvAddToBothSides,
    InvClearFractions,
    InvCombineLikeTerms,
    InvDistributeOverSum,
    InvExpandProduct,
    InvFlipSides,
    InvMultiplyBothSides,
    default_inverse_registry,
)
from ggmr.problems.templates import linear_seed, quadratic_seed
from ggmr.rules.core import *  # noqa: F401,F403  (registers forward rules)
from ggmr.search.bfs import bfs
from ggmr.state import EqState


def _verify_inverse_then_forward(
    seed_state: EqState, inverse_rule, action, max_nodes: int = 5000
) -> bool:
    """Apply the inverse to seed_state, then run forward BFS; return True if BFS solves."""
    new_state = inverse_rule.apply(seed_state, action)
    target_lhs_repr = canonical_repr(seed_state.lhs)
    target_rhs_repr = canonical_repr(seed_state.rhs)

    def is_target(s):
        return (
            canonical_repr(s.lhs) == target_lhs_repr
            and canonical_repr(s.rhs) == target_rhs_repr
        ) or s.is_canonical_target()

    result = bfs(new_state, is_target, max_nodes=max_nodes, max_depth=20, check_soundness=False)
    return result.found


def test_inv_add_to_both_sides_roundtrip():
    rng = random.Random(0)
    seed = linear_seed(rng)
    rule = InvAddToBothSides()
    action = next(iter(rule.enumerate(seed, rng)))
    assert _verify_inverse_then_forward(seed, rule, action)


def test_inv_multiply_both_sides_roundtrip():
    rng = random.Random(1)
    seed = linear_seed(rng)
    rule = InvMultiplyBothSides()
    action = next(iter(rule.enumerate(seed, rng)))
    assert _verify_inverse_then_forward(seed, rule, action)


def test_inv_flip_sides_roundtrip():
    rng = random.Random(2)
    seed = linear_seed(rng)
    rule = InvFlipSides()
    action = next(iter(rule.enumerate(seed, rng)))
    assert _verify_inverse_then_forward(seed, rule, action)


def test_inv_distribute_over_sum_roundtrip():
    rng = random.Random(3)
    seed = linear_seed(rng)
    rule = InvDistributeOverSum()
    action = next(iter(rule.enumerate(seed, rng)))
    # Distribution-then-combine roundtrip — BFS may need a few steps
    assert _verify_inverse_then_forward(seed, rule, action, max_nodes=10000)


def test_inv_combine_like_terms_roundtrip():
    rng = random.Random(4)
    seed = linear_seed(rng)
    rule = InvCombineLikeTerms()
    action = next(iter(rule.enumerate(seed, rng)))
    assert _verify_inverse_then_forward(seed, rule, action, max_nodes=10000)


def test_inv_clear_fractions_roundtrip():
    rng = random.Random(5)
    seed = linear_seed(rng)
    rule = InvClearFractions()
    action = next(iter(rule.enumerate(seed, rng)))
    # Apply inverse: state is now a fraction equation. CROSS_MULTIPLY or
    # CLEAR_FRACTIONS_BY_LCD should solve it.
    assert _verify_inverse_then_forward(seed, rule, action, max_nodes=10000)


def test_inv_expand_product_roundtrip():
    rng = random.Random(6)
    seed = quadratic_seed(rng)
    rule = InvExpandProduct()
    actions = list(rule.enumerate(seed, rng))
    # Quadratic seed has a Mul lhs with Add factors → enumerable
    assert len(actions) >= 1
    assert _verify_inverse_then_forward(seed, rule, actions[0], max_nodes=10000)


def test_default_inverse_registry_has_all_rules():
    rules = default_inverse_registry.all_rules()
    names = {r.name for r in rules}
    assert "INV_ADD_TO_BOTH_SIDES" in names
    assert "INV_MULTIPLY_BOTH_SIDES" in names
    assert "INV_FLIP_SIDES" in names
    assert "INV_DISTRIBUTE_OVER_SUM" in names
    assert "INV_COMBINE_LIKE_TERMS" in names
    assert "INV_CLEAR_FRACTIONS" in names
    assert "INV_EXPAND_PRODUCT" in names
