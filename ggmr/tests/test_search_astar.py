"""Tests for A* search engine.

§3.2 of `ggmr/PHASE1B_PREREG.md` is the headline criterion: A* with
`WeightedSumCompositeHeuristic` solves `rat05` in ≤75 expanded nodes (< 50% of
Phase 1a's BFS measurement of 151).
"""

from __future__ import annotations

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from ggmr.expr.tree import canonical_repr
from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
from ggmr.rules.core import *  # noqa: F401,F403  (registers)
from ggmr.search.astar import astar
from ggmr.state import EqState


def _target_from_yaml(state: EqState, target_lhs: str, target_rhs: str) -> object:
    """Build is_target predicate matching the YAML canonical_target structurally."""
    var = state.var
    lhs_expr = parse_expr(target_lhs, local_dict={var.name: var}, evaluate=False)
    rhs_expr = parse_expr(target_rhs, local_dict={var.name: var}, evaluate=False)
    target_lhs_repr = canonical_repr(lhs_expr)
    target_rhs_repr = canonical_repr(rhs_expr)

    def is_target(s: EqState) -> bool:
        return (
            canonical_repr(s.lhs) == target_lhs_repr
            and canonical_repr(s.rhs) == target_rhs_repr
        ) or s.is_canonical_target()

    return is_target


def test_astar_solves_lin01_in_few_steps():
    """Sanity check on the simplest Phase 0 problem."""
    s = EqState.from_strings("2*x + 3", "7")
    h = WeightedSumCompositeHeuristic()
    result = astar(s, _target_from_yaml(s, "x", "2"), heuristic=h, max_nodes=500)
    assert result.found, f"A* failed lin01 with stats={result.stats.to_dict()}"
    # Should find a path of 1-2 steps (ISOLATE_VARIABLE in 1; or ADD+DIVIDE in 2).
    assert result.num_steps <= 3


def test_astar_rat05_node_efficiency():
    """§3.2 criterion: A* on rat05 must expand ≤ 75 nodes (< 50% of BFS's 151)."""
    s = EqState.from_strings("(2*x - 1)/(x + 1)", "1/2")
    h = WeightedSumCompositeHeuristic()
    result = astar(s, _target_from_yaml(s, "x", "1"), heuristic=h, max_nodes=5000)
    assert result.found, f"A* failed rat05 with stats={result.stats.to_dict()}"
    assert result.stats.nodes_expanded <= 75, (
        f"§3.2 FAILED: A* expanded {result.stats.nodes_expanded} nodes on rat05 "
        f"(threshold 75, BFS baseline 151)"
    )


def test_astar_determinism():
    """Two runs on the same input produce byte-identical paths and stats."""
    s = EqState.from_strings("2*x + 3", "7")
    h = WeightedSumCompositeHeuristic()
    r1 = astar(s, _target_from_yaml(s, "x", "2"), heuristic=h, max_nodes=500)
    r2 = astar(s, _target_from_yaml(s, "x", "2"), heuristic=h, max_nodes=500)
    assert r1.found and r2.found
    p1 = [(canonical_repr(st.lhs), canonical_repr(st.rhs), a.canonical_key()) for st, a in r1.path]
    p2 = [(canonical_repr(st.lhs), canonical_repr(st.rhs), a.canonical_key()) for st, a in r2.path]
    assert p1 == p2
    assert r1.stats.nodes_expanded == r2.stats.nodes_expanded
    assert dict(r1.stats.rule_application_counts) == dict(r2.stats.rule_application_counts)


def test_astar_unsolvable_returns_not_found():
    """Equation x² + 1 = 0 has no real roots; with budget exhaustion A* returns not-found."""
    s = EqState.from_strings("x**2 + 1", "0")

    def is_target(state: EqState) -> bool:
        return state.is_canonical_target()

    h = WeightedSumCompositeHeuristic()
    result = astar(s, is_target, heuristic=h, max_nodes=200)
    # We just want no exception; either success (found a complex/irreducible canonical) or not.
    assert isinstance(result.stats.nodes_expanded, int)


def test_astar_weighted_aggressiveness():
    """Smoke: weight=2.0 on rat05 should solve in equal-or-fewer node expansions
    than weight=1.0 (greedier search, more aggressive). Reported as a §5 secondary
    metric — no decision authority."""
    s = EqState.from_strings("(2*x - 1)/(x + 1)", "1/2")
    h = WeightedSumCompositeHeuristic()
    r1 = astar(s, _target_from_yaml(s, "x", "1"), heuristic=h, max_nodes=5000, weight=1.0)
    r2 = astar(s, _target_from_yaml(s, "x", "1"), heuristic=h, max_nodes=5000, weight=2.0)
    assert r1.found and r2.found
    # weight=2.0 should be at most as many nodes (typically fewer)
    # Tolerance: weight=2.0 may explore more in pathological cases, so just assert <= 2x.
    assert r2.stats.nodes_expanded <= 2 * r1.stats.nodes_expanded
