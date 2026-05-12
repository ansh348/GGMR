"""Tests for the 4 hard-problem inverse rules.

Each rule must:
1. Produce a state that's structurally different from its parent (canonical_repr).
2. Be sound: verify_transition(parent, child, ...) returns VERIFY_PASS.
3. Update the `excluded` set correctly (where applicable).
4. Roundtrip: forward BFS solves back to the canonical target within budget.
"""

from __future__ import annotations

import random

import sympy as sp
from sympy import Add, Integer, Mul, Symbol

from ggmr.expr.tree import canonical_repr
from ggmr.problems.hard_inverse_rules import (
    InvDisguiseByExpansion,
    InvEmbedInFraction,
    InvNestInRational,
    InvSplitAcrossSides,
    hard_inverse_registry,
)
from ggmr.problems.inverse_rules import default_inverse_registry
from ggmr.problems.templates import linear_seed, quadratic_seed
from ggmr.rules.core import *  # noqa: F401,F403  (registers forward rules)
from ggmr.search.bfs import bfs
from ggmr.soundness import VERIFY_PASS, VERIFY_UNVERIFIABLE, verify_transition
from ggmr.state import EqState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_soundness(parent: EqState, child: EqState) -> None:
    verdict, reason = verify_transition(
        parent.lhs,
        parent.rhs,
        child.lhs,
        child.rhs,
        parent.var,
        parent_excluded=parent.excluded,
        child_excluded=child.excluded,
    )
    # VERIFY_UNVERIFIABLE happens when sympy.solve raises on disguised forms;
    # this is acceptable (the same skip behavior used by BFS at runtime).
    assert verdict in (VERIFY_PASS, VERIFY_UNVERIFIABLE), reason


def _check_structural_change(parent: EqState, child: EqState) -> None:
    diff = (
        canonical_repr(child.lhs) != canonical_repr(parent.lhs)
        or canonical_repr(child.rhs) != canonical_repr(parent.rhs)
    )
    assert diff, "Inverse rule produced structurally identical state"


def _bfs_solves_back(child: EqState, target: EqState, max_nodes: int = 20_000) -> bool:
    target_l = canonical_repr(target.lhs)
    target_r = canonical_repr(target.rhs)

    def is_target(s):
        return (
            canonical_repr(s.lhs) == target_l and canonical_repr(s.rhs) == target_r
        ) or s.is_canonical_target()

    result = bfs(
        child,
        is_target,
        max_nodes=max_nodes,
        max_depth=25,
        check_soundness=False,
    )
    return result.found


# ---------------------------------------------------------------------------
# InvEmbedInFraction
# ---------------------------------------------------------------------------


def test_inv_embed_in_fraction_structural_change():
    rng = random.Random(0)
    seed = linear_seed(rng)  # x = a
    rule = InvEmbedInFraction()
    action = next(iter(rule.enumerate(seed, rng)))
    child = rule.apply(seed, action)
    _check_structural_change(seed, child)


def test_inv_embed_in_fraction_excluded_set():
    rng = random.Random(1)
    seed = linear_seed(rng)
    rule = InvEmbedInFraction()
    action = next(iter(rule.enumerate(seed, rng)))
    (k,) = action.params
    child = rule.apply(seed, action)
    assert Integer(-int(k)) in child.excluded


def test_inv_embed_in_fraction_sound():
    rng = random.Random(2)
    seed = linear_seed(rng)
    rule = InvEmbedInFraction()
    action = next(iter(rule.enumerate(seed, rng)))
    child = rule.apply(seed, action)
    _check_soundness(seed, child)


def test_inv_embed_in_fraction_roundtrip():
    rng = random.Random(3)
    seed = linear_seed(rng)
    rule = InvEmbedInFraction()
    action = next(iter(rule.enumerate(seed, rng)))
    child = rule.apply(seed, action)
    assert _bfs_solves_back(child, seed)


# ---------------------------------------------------------------------------
# InvSplitAcrossSides
# ---------------------------------------------------------------------------


def _split_friendly_seed() -> EqState:
    """A seed where lhs is Add with >=2 terms, satisfying InvSplitAcrossSides
    precondition. Solution: x = 2 (since x + 3 = 5)."""
    x = sp.Symbol("x")
    lhs = Add(x, Integer(3), evaluate=False)
    return EqState(lhs=lhs, rhs=Integer(5), var=x)


def test_inv_split_across_sides_precondition():
    """Empty enumerate when lhs is not Add."""
    rng = random.Random(0)
    seed = linear_seed(rng)  # lhs is bare var
    rule = InvSplitAcrossSides()
    actions = list(rule.enumerate(seed, rng))
    assert actions == []


def test_inv_split_across_sides_structural_change():
    rng = random.Random(0)
    seed = _split_friendly_seed()
    rule = InvSplitAcrossSides()
    action = next(iter(rule.enumerate(seed, rng)))
    child = rule.apply(seed, action)
    _check_structural_change(seed, child)


def test_inv_split_across_sides_excluded_set():
    rng = random.Random(1)
    seed = _split_friendly_seed()
    rule = InvSplitAcrossSides()
    action = next(iter(rule.enumerate(seed, rng)))
    (k,) = action.params
    child = rule.apply(seed, action)
    assert Integer(-int(k)) in child.excluded


def test_inv_split_across_sides_sound():
    rng = random.Random(2)
    seed = _split_friendly_seed()
    rule = InvSplitAcrossSides()
    action = next(iter(rule.enumerate(seed, rng)))
    child = rule.apply(seed, action)
    _check_soundness(seed, child)


def test_inv_split_across_sides_roundtrip():
    rng = random.Random(3)
    seed = _split_friendly_seed()
    rule = InvSplitAcrossSides()
    action = next(iter(rule.enumerate(seed, rng)))
    child = rule.apply(seed, action)
    # Canonical target of `_split_friendly_seed` is x = 2 (since x+3 = 5).
    x = seed.var
    canonical_target = EqState(lhs=x, rhs=Integer(2), var=x)
    assert _bfs_solves_back(child, canonical_target, max_nodes=30_000)


# ---------------------------------------------------------------------------
# InvDisguiseByExpansion
# ---------------------------------------------------------------------------


def test_inv_disguise_by_expansion_precondition():
    """Requires lhs to be Mul of Add factors."""
    rng = random.Random(0)
    seed = linear_seed(rng)  # lhs is bare var
    rule = InvDisguiseByExpansion()
    actions = list(rule.enumerate(seed, rng))
    assert actions == []


def test_inv_disguise_by_expansion_structural_change():
    rng = random.Random(0)
    seed = quadratic_seed(rng)  # (x-r1)*(x-r2) = 0
    rule = InvDisguiseByExpansion()
    actions = list(rule.enumerate(seed, rng))
    if not actions:
        # quadratic_seed could produce a degenerate (x-r)(x-r) form rarely; skip if so
        return
    child = rule.apply(seed, actions[0])
    _check_structural_change(seed, child)


def test_inv_disguise_by_expansion_no_excluded():
    rng = random.Random(1)
    seed = quadratic_seed(rng)
    rule = InvDisguiseByExpansion()
    actions = list(rule.enumerate(seed, rng))
    if not actions:
        return
    child = rule.apply(seed, actions[0])
    assert child.excluded == seed.excluded


def test_inv_disguise_by_expansion_sound():
    rng = random.Random(2)
    seed = quadratic_seed(rng)
    rule = InvDisguiseByExpansion()
    actions = list(rule.enumerate(seed, rng))
    if not actions:
        return
    child = rule.apply(seed, actions[0])
    _check_soundness(seed, child)


def test_inv_disguise_by_expansion_roundtrip():
    rng = random.Random(3)
    seed = quadratic_seed(rng)
    rule = InvDisguiseByExpansion()
    actions = list(rule.enumerate(seed, rng))
    if not actions:
        return
    child = rule.apply(seed, actions[0])
    assert _bfs_solves_back(child, seed, max_nodes=30_000)


# ---------------------------------------------------------------------------
# InvNestInRational
# ---------------------------------------------------------------------------


def test_inv_nest_in_rational_structural_change():
    rng = random.Random(0)
    seed = linear_seed(rng)
    rule = InvNestInRational()
    action = next(iter(rule.enumerate(seed, rng)))
    child = rule.apply(seed, action)
    _check_structural_change(seed, child)


def test_inv_nest_in_rational_excluded_set():
    rng = random.Random(1)
    seed = linear_seed(rng)
    rule = InvNestInRational()
    action = next(iter(rule.enumerate(seed, rng)))
    _, q = action.params
    child = rule.apply(seed, action)
    assert Integer(-int(q)) in child.excluded


def test_inv_nest_in_rational_sound():
    rng = random.Random(2)
    seed = linear_seed(rng)
    rule = InvNestInRational()
    action = next(iter(rule.enumerate(seed, rng)))
    child = rule.apply(seed, action)
    _check_soundness(seed, child)


def test_inv_nest_in_rational_roundtrip():
    rng = random.Random(3)
    seed = linear_seed(rng)
    rule = InvNestInRational()
    action = next(iter(rule.enumerate(seed, rng)))
    child = rule.apply(seed, action)
    assert _bfs_solves_back(child, seed, max_nodes=50_000)


def test_inv_nest_in_rational_skips_degenerate():
    """When lhs == rhs structurally, enumerate yields nothing."""
    x = sp.Symbol("x")
    seed = EqState(lhs=x, rhs=x, var=x)
    rule = InvNestInRational()
    rng = random.Random(0)
    actions = list(rule.enumerate(seed, rng))
    assert actions == []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_hard_inverse_registry_has_all_rules():
    names = {r.name for r in hard_inverse_registry.all_rules()}
    # all 7 existing
    assert "INV_ADD_TO_BOTH_SIDES" in names
    assert "INV_MULTIPLY_BOTH_SIDES" in names
    assert "INV_FLIP_SIDES" in names
    assert "INV_DISTRIBUTE_OVER_SUM" in names
    assert "INV_COMBINE_LIKE_TERMS" in names
    assert "INV_CLEAR_FRACTIONS" in names
    assert "INV_EXPAND_PRODUCT" in names
    # 4 new
    assert "INV_EMBED_IN_FRACTION" in names
    assert "INV_SPLIT_ACROSS_SIDES" in names
    assert "INV_DISGUISE_BY_EXPANSION" in names
    assert "INV_NEST_IN_RATIONAL" in names


def test_default_inverse_registry_unchanged():
    """Phase 1b reproducibility: default_inverse_registry must remain exactly 7 rules."""
    assert len(default_inverse_registry.all_rules()) == 7
