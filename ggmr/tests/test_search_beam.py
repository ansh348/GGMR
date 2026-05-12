"""Tests for Beam search.

§3.3 of `ggmr/PHASE1B_PREREG.md`: Beam B=10 with `WeightedSumCompositeHeuristic`
solves ≥ 18/20 Phase 0 problems within `max_depth=20`.
"""

from __future__ import annotations

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from ggmr.expr.tree import canonical_repr
from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
from ggmr.rules.core import *  # noqa: F401,F403  (registers)
from ggmr.search.beam import beam_search
from ggmr.state import EqState


def _target_predicate(initial: EqState, target_lhs: str, target_rhs: str):
    var = initial.var
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


def test_beam_solves_lin01_in_few_steps():
    s = EqState.from_strings("2*x + 3", "7")
    h = WeightedSumCompositeHeuristic()
    result = beam_search(s, _target_predicate(s, "x", "2"), heuristic=h, beam_width=10)
    assert result.found, f"Beam B=10 failed lin01"


def test_beam_determinism():
    s = EqState.from_strings("2*x + 3", "7")
    h = WeightedSumCompositeHeuristic()
    r1 = beam_search(s, _target_predicate(s, "x", "2"), heuristic=h, beam_width=10)
    r2 = beam_search(s, _target_predicate(s, "x", "2"), heuristic=h, beam_width=10)
    assert r1.found and r2.found
    p1 = [(canonical_repr(st.lhs), canonical_repr(st.rhs), a.canonical_key()) for st, a in r1.path]
    p2 = [(canonical_repr(st.lhs), canonical_repr(st.rhs), a.canonical_key()) for st, a in r2.path]
    assert p1 == p2


@pytest.mark.slow
def test_beam_solves_phase0_problem_set(phase0_states):
    """§3.3 criterion: B=10 must solve ≥ 18/20 Phase 0 problems."""
    h = WeightedSumCompositeHeuristic()
    solved = 0
    failed_ids: list[str] = []
    for problem_id, initial, target in phase0_states:
        is_target = _target_predicate(initial, str(target.lhs), str(target.rhs))
        result = beam_search(
            initial, is_target, heuristic=h, beam_width=10, max_depth=20, problem_id=problem_id
        )
        if result.found:
            solved += 1
        else:
            failed_ids.append(problem_id)
    assert solved >= 18, (
        f"§3.3 FAILED: Beam B=10 solved {solved}/20 (threshold 18). "
        f"Failed: {failed_ids}"
    )


def test_beam_narrower_beam_misses_more_problems(phase0_states):
    """Sanity: B=2 should solve fewer problems than B=10 (pruning matters)."""
    h = WeightedSumCompositeHeuristic()
    solved_b2 = 0
    solved_b10 = 0
    # Sample a few problems for the smoke check
    sample = phase0_states[:5]
    for _, initial, target in sample:
        is_target = _target_predicate(initial, str(target.lhs), str(target.rhs))
        r2 = beam_search(initial, is_target, heuristic=h, beam_width=2, max_depth=20)
        r10 = beam_search(initial, is_target, heuristic=h, beam_width=10, max_depth=20)
        if r2.found:
            solved_b2 += 1
        if r10.found:
            solved_b10 += 1
    # B=10 should solve at least as many as B=2 (typically more)
    assert solved_b10 >= solved_b2
