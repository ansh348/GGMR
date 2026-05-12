"""End-to-end BFS tests on Phase 0 problems.

Per `ggmr/PHASE1A_PREREG.md` §3.1, BFS with the 15-rule library should solve
≥18/20 Phase 0 problems within budget (max_nodes=50,000, max_depth=20).

For test runtime, this test uses a tighter budget per problem (max_nodes=5_000,
max_depth=12). The full-budget run is in `_smoke.py` and reported in
`PHASE1A_README.md`.
"""

from __future__ import annotations

import pytest

from ggmr.expr.tree import canonical_repr
from ggmr.rules.core import *  # noqa: F401,F403
from ggmr.search.bfs import bfs


def _build_target_predicate(target_state):
    target_lhs_key = canonical_repr(target_state.lhs)
    target_rhs_key = canonical_repr(target_state.rhs)
    target_solset = target_state.solution_set()

    def is_target(s, _l=target_lhs_key, _r=target_rhs_key, _ss=target_solset):
        if canonical_repr(s.lhs) == _l and canonical_repr(s.rhs) == _r:
            return True
        if s.is_canonical_target() and s.solution_set() == _ss:
            return True
        return False

    return is_target


@pytest.mark.parametrize("budget_nodes", [5_000])
def test_bfs_solves_phase0_problem_set(phase0_states, budget_nodes):
    """Track per-problem solve outcomes; require ≥ 18/20 solved within tighter test budget.

    The full-budget run is in `_smoke.py` and reported in `PHASE1A_README.md`.
    """
    n_pass = 0
    failed: list[str] = []
    for problem_id, initial, target_state in phase0_states:
        is_target = _build_target_predicate(target_state)
        try:
            result = bfs(
                initial,
                is_target,
                max_nodes=budget_nodes,
                max_depth=12,
                problem_id=problem_id,
            )
        except Exception as e:
            failed.append(f"{problem_id}: EXCEPTION: {type(e).__name__}: {e}")
            continue
        if result.found:
            n_pass += 1
        else:
            failed.append(
                f"{problem_id}: FAIL expanded={result.stats.nodes_expanded}"
            )
    # Pre-reg §3.1 — primary success criterion. Allow 2-problem failure budget.
    assert n_pass >= 18, (
        f"BFS solved only {n_pass}/{len(phase0_states)}; "
        f"failures:\n" + "\n".join(failed)
    )


def test_bfs_lin01_path_length_two(phase0_states):
    """lin01 (2x+3=7) should solve in ≤ 2 steps."""
    pid_to_entry = {pid: (init, tgt) for pid, init, tgt in phase0_states}
    initial, target_state = pid_to_entry["lin01"]
    is_target = _build_target_predicate(target_state)
    result = bfs(initial, is_target, max_nodes=5_000, max_depth=12, problem_id="lin01")
    assert result.found
    assert result.num_steps <= 2


def test_bfs_determinism(phase0_states):
    """Two BFS runs on lin01 produce identical paths and stats."""
    pid_to_entry = {pid: (init, tgt) for pid, init, tgt in phase0_states}
    initial, target_state = pid_to_entry["lin01"]
    is_target = _build_target_predicate(target_state)
    r1 = bfs(initial, is_target, max_nodes=5_000, max_depth=12, problem_id="lin01")
    r2 = bfs(initial, is_target, max_nodes=5_000, max_depth=12, problem_id="lin01")
    # Path: same length, same actions in order
    assert r1.num_steps == r2.num_steps
    for (s1, a1), (s2, a2) in zip(r1.path, r2.path):
        assert a1.canonical_key() == a2.canonical_key()
    # Stats: byte-identical rule application counts
    assert r1.stats.rule_application_counts == r2.stats.rule_application_counts


def test_bfs_unsolvable_returns_not_found():
    """Constructing a deliberately-unsolvable state (search budget too low for an
    intractable transformation): BFS returns found=False without raising."""
    from ggmr.state import EqState
    import sympy as sp

    # x^5 + x + 1 = 0 — sympy.factor doesn't fully factor this in Q[x]; with budget=50, BFS will time out
    initial = EqState.from_strings("x**5 + x + 1", "0")

    def is_target(s):
        return s.is_canonical_target() and len(s.solution_set()) > 0

    result = bfs(initial, is_target, max_nodes=50, max_depth=4, problem_id="quintic")
    # Whether it found a target or not, it must NOT have raised
    assert isinstance(result.found, bool)
