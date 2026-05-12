"""Beam search engine. Phase 2's learned-value-network plug-in point.

At each depth level, all current beam members are expanded to children, and
the top-`beam_width` children (by `heuristic.evaluate`) are kept for the next
level. Greedy pruning — may miss optimal paths if `beam_width` is too small.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from ..expr.tree import normalize
from ..heuristics.composite import Heuristic
from ..rules.base import Action, merge_guard_into_state
from ..rules.registry import Registry, default_registry
from ..soundness import (
    IllegalStepError,
    VERIFY_PASS,
    VERIFY_UNSOUND,
    VERIFY_UNVERIFIABLE,
    verify_transition,
)
from ..state import EqState
from .bfs import SearchResult
from .stats import SearchStats


def beam_search(
    initial: EqState,
    is_target: Callable[[EqState], bool],
    *,
    heuristic: Heuristic,
    beam_width: int = 10,
    max_depth: int = 20,
    rules: Optional[Registry] = None,
    check_soundness: bool = True,
    problem_id: str = "<beam>",
) -> SearchResult:
    """Beam search from `initial` until `is_target(state)` is True.

    Args:
        initial: starting state.
        is_target: termination predicate.
        heuristic: callable with `.evaluate(state) -> float`. Lower = closer to goal.
        beam_width: number of states retained per depth level.
        max_depth: hard cap on path length.
        rules: rule registry; defaults to `default_registry`.
        check_soundness: run `verify_transition` on every (parent, child).
        problem_id: tag for IllegalStepError diagnostics.
    """
    if rules is None:
        rules = default_registry
    stats = SearchStats()
    t0 = time.perf_counter()

    if is_target(initial):
        stats.time_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(found=True, final_state=initial, path=[], stats=stats)

    parent_map: dict[EqState, tuple[EqState, Action]] = {}
    visited: set[EqState] = {initial}
    current_beam: list[EqState] = [initial]

    for depth in range(max_depth):
        next_candidates: list[tuple[float, str, EqState]] = []  # (h_score, tie_key, state)
        for state in current_beam:
            stats.nodes_expanded += 1
            for rule, action in rules.enumerate_actions(state):
                stats.nodes_generated += 1
                guard = rule.guard(state, action)
                if not guard.ok:
                    stats.guard_rejections += 1
                    continue
                try:
                    child = rule.apply(state, action)
                except Exception:
                    stats.guard_rejections += 1
                    continue
                child = (
                    merge_guard_into_state(child, guard)
                    if (guard.new_excluded or guard.new_side_conditions)
                    else child
                )
                child = child.with_lhs_rhs(normalize(child.lhs), normalize(child.rhs))

                if child in visited:
                    stats.dedup_hits += 1
                    continue

                if check_soundness:
                    verdict, reason = verify_transition(
                        state.lhs,
                        state.rhs,
                        child.lhs,
                        child.rhs,
                        state.var,
                        parent_excluded=state.excluded,
                        child_excluded=child.excluded,
                    )
                    if verdict == VERIFY_UNSOUND:
                        raise IllegalStepError(problem_id, depth, f"{rule.name}: {reason}")
                    if verdict == VERIFY_UNVERIFIABLE:
                        stats.guard_rejections += 1
                        continue

                visited.add(child)
                parent_map[child] = (state, action)
                stats.rule_application_counts[rule.name] = (
                    stats.rule_application_counts.get(rule.name, 0) + 1
                )

                if is_target(child):
                    stats.time_ms = (time.perf_counter() - t0) * 1000
                    stats.max_depth_reached = depth + 1
                    return SearchResult(
                        found=True,
                        final_state=child,
                        path=_reconstruct_beam_path(parent_map, initial, child),
                        stats=stats,
                    )

                try:
                    h = float(heuristic.evaluate(child))
                except Exception:
                    h = float("inf")
                next_candidates.append((h, action.canonical_key(), child))

        if not next_candidates:
            break

        # Keep top-`beam_width` by (h_score, tie_key) ascending
        next_candidates.sort(key=lambda t: (t[0], t[1]))
        current_beam = [s for _, _, s in next_candidates[:beam_width]]
        stats.max_depth_reached = depth + 1

    stats.time_ms = (time.perf_counter() - t0) * 1000
    return SearchResult(found=False, final_state=None, path=[], stats=stats)


def _reconstruct_beam_path(
    parent_map: dict[EqState, tuple[EqState, Action]],
    initial: EqState,
    final: EqState,
) -> list[tuple[EqState, Action]]:
    rev: list[tuple[EqState, Action]] = []
    cur = final
    while cur != initial:
        parent, action = parent_map[cur]
        rev.append((parent, action))
        cur = parent
    rev.reverse()
    return rev
