"""Tests for MCTS engine. The headline gate is the oracle sanity check:
MCTS with a BFS-derived value oracle + uniform policy solves trivial problems."""

from __future__ import annotations

import pytest

from ggmr.rules.core import *  # noqa: F401,F403  (register rules)
from ggmr.search.mcts import (
    MCTSNode,
    mcts_search,
    oracle_value_factory,
    uniform_policy,
)
from ggmr.state import EqState
from ggmr.training.extract_pairs import _build_is_target


def _make_problem(lhs: str, rhs: str, tlhs: str, trhs: str):
    initial = EqState.from_strings(lhs, rhs)
    target = EqState.from_strings(tlhs, trhs)
    return initial, target, _build_is_target(target)


def test_already_solved_returns_immediately():
    initial, _, is_target = _make_problem("x", "5", "x", "5")
    result = mcts_search(
        initial,
        is_target,
        value_fn=lambda s: 1.0,
        policy_fn=uniform_policy,
        num_simulations=1,
        max_moves=1,
    )
    assert result.found
    assert result.num_steps == 0


def test_oracle_solves_one_step_linear():
    initial, target, is_target = _make_problem("x + 3", "5", "x", "2")
    value_fn = oracle_value_factory(is_target)
    result = mcts_search(
        initial,
        is_target,
        value_fn=value_fn,
        policy_fn=uniform_policy,
        num_simulations=50,
        max_moves=5,
    )
    assert result.found, f"MCTS failed: stats={result.stats.to_dict()}"
    assert result.num_steps <= 3


def test_oracle_solves_two_step_linear():
    initial, target, is_target = _make_problem("2*x + 3", "7", "x", "2")
    value_fn = oracle_value_factory(is_target)
    result = mcts_search(
        initial,
        is_target,
        value_fn=value_fn,
        policy_fn=uniform_policy,
        num_simulations=80,
        max_moves=6,
    )
    assert result.found, f"MCTS failed: stats={result.stats.to_dict()}"
    assert result.num_steps <= 4


def test_oracle_solves_quadratic_factor():
    initial, target, is_target = _make_problem("x**2 - 4", "0", "(x-2)*(x+2)", "0")
    value_fn = oracle_value_factory(is_target)
    result = mcts_search(
        initial,
        is_target,
        value_fn=value_fn,
        policy_fn=uniform_policy,
        num_simulations=80,
        max_moves=6,
    )
    assert result.found, f"MCTS failed: stats={result.stats.to_dict()}"


def test_visit_distribution_is_valid_probability():
    """Each visit distribution sums to 1.0 (or is empty if no children visited)."""
    initial, _, is_target = _make_problem("2*x + 3", "7", "x", "2")
    value_fn = oracle_value_factory(is_target)
    result = mcts_search(
        initial,
        is_target,
        value_fn=value_fn,
        policy_fn=uniform_policy,
        num_simulations=50,
        max_moves=5,
    )
    for dist in result.visit_distributions:
        if not dist:
            continue
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6, f"distribution sum = {total}, dist = {dist}"


def test_puct_score_handles_zero_visits():
    """A fresh child has visit_count=0 and Q=0; PUCT must equal c_puct * prior * sqrt(N) / 1."""
    state = EqState.from_strings("x", "1")
    parent = MCTSNode(state=state, visit_count=1)
    child = MCTSNode(state=state, parent=parent, prior=0.5)
    score = child.puct_score(c_puct=1.5, parent_visits=1)
    assert abs(score - 0.5 * 1.5 * 1.0 / 1.0) < 1e-9


def test_stats_track_simulations():
    initial, _, is_target = _make_problem("x + 1", "3", "x", "2")
    value_fn = oracle_value_factory(is_target)
    result = mcts_search(
        initial,
        is_target,
        value_fn=value_fn,
        policy_fn=uniform_policy,
        num_simulations=30,
        max_moves=3,
    )
    assert result.stats.total_simulations >= 30
    assert result.stats.value_evals + result.stats.policy_evals > 0


def test_uniform_policy_sums_to_one():
    assert uniform_policy(EqState.from_strings("x", "1"), []) == {}
    d = uniform_policy(EqState.from_strings("x", "1"), ["A", "B", "C"])
    assert abs(sum(d.values()) - 1.0) < 1e-9
    assert all(abs(v - 1.0 / 3) < 1e-9 for v in d.values())
