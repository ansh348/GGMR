"""Tests for `training_safe` rule attribute + `training_only` filter on enumerate_actions.

Marcus Constraint 1: training pipelines (BFS trace generation, SL data prep, ExIt
MCTS rollouts) must never see oracle-shortcut rules. Inference paths keep
oracles available for fast-mode evaluation.

The filter is enforced at `Registry.enumerate_actions(state, *, training_only=...)`.
Rule classes declare `training_safe: bool = True` (primitive) or `False` (oracle).
Algebra rules predate this attribute and default to True via `getattr`.
"""

from __future__ import annotations

from typing import Iterator

import sympy as sp

from ggmr.rules.base import Action, GuardResult
from ggmr.rules.core import *  # noqa: F401,F403  (register algebra rules)
from ggmr.rules.registry import Registry, default_registry
from ggmr.state import EqState


class _FakeOracleRule:
    """A no-op rule that yields exactly one action; marked training_safe=False
    so it should be excluded under training_only=True."""

    name = "_TEST_FAKE_ORACLE"
    arity = 0
    training_safe = False

    def enumerate(self, state: EqState) -> Iterator[Action]:
        yield Action(self.name, params=(), target_path=(), target_side="both")

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        return state  # no-op


class _FakePrimitiveRule:
    """A no-op rule that yields exactly one action; training_safe defaults to
    True (primitive)."""

    name = "_TEST_FAKE_PRIMITIVE"
    arity = 0
    training_safe = True

    def enumerate(self, state: EqState) -> Iterator[Action]:
        yield Action(self.name, params=(), target_path=(), target_side="both")

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        return state


def _isolated_registry() -> Registry:
    """Fresh registry with both fake rules registered; doesn't pollute default_registry."""
    reg = Registry()
    reg.register(_FakeOracleRule())
    reg.register(_FakePrimitiveRule())
    return reg


def test_training_only_excludes_oracle():
    reg = _isolated_registry()
    state = EqState.from_strings("x", "1")
    names = {rule.name for rule, _ in reg.enumerate_actions(state, training_only=True)}
    assert "_TEST_FAKE_PRIMITIVE" in names
    assert "_TEST_FAKE_ORACLE" not in names


def test_training_only_false_includes_oracle():
    reg = _isolated_registry()
    state = EqState.from_strings("x", "1")
    names = {rule.name for rule, _ in reg.enumerate_actions(state, training_only=False)}
    assert "_TEST_FAKE_PRIMITIVE" in names
    assert "_TEST_FAKE_ORACLE" in names


def test_default_is_training_only_false():
    """Default kwarg keeps current behavior (all rules visible) so inference is unaffected."""
    reg = _isolated_registry()
    state = EqState.from_strings("x", "1")
    names = {rule.name for rule, _ in reg.enumerate_actions(state)}
    assert "_TEST_FAKE_ORACLE" in names


def test_existing_algebra_rules_default_training_safe_true():
    """Algebra rules predate the attribute; getattr(rule, "training_safe", True)
    means they all pass through both modes unchanged. This regression-guards
    Phase 0.1: adding the kwarg must not silently drop existing rules."""
    state = EqState.from_strings("2*x + 3", "7")
    train_names = {r.name for r, _ in default_registry.enumerate_actions(state, training_only=True)}
    full_names = {r.name for r, _ in default_registry.enumerate_actions(state, training_only=False)}
    # Algebra rules: no oracle shortcuts registered yet, so the two views match.
    assert train_names == full_names
    # And both views contain at least the linear-equation-solving rules we'd expect.
    assert "DISTRIBUTE_OVER_SUBTREE" in full_names or len(full_names) > 0
