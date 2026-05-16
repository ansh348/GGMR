"""A* search engine with pluggable heuristic. D1-baseline progression per
`ggmr_v10.pdf` §5.3 and Phase 1b §3.2.

Usage:

    from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
    h = WeightedSumCompositeHeuristic()
    result = astar(initial, lambda s: s.is_canonical_target(), heuristic=h)

Same `SearchResult` / `SearchStats` shape as `bfs()`. Tie-breaking on equal
`f_score` is via a monotonic counter (insertion order), guaranteeing
byte-identical paths across runs.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
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
from .bfs import SearchResult, _reconstruct_path
from .stats import SearchStats


def astar(
    initial: EqState,
    is_target: Callable[[EqState], bool],
    *,
    heuristic: Heuristic,
    max_nodes: int = 50_000,
    max_depth: int = 20,
    rules: Optional[Registry] = None,
    check_soundness: bool = True,
    weight: float = 1.0,
    problem_id: str = "<astar>",
    training_only: bool = False,
) -> SearchResult:
    """A* search from `initial` until `is_target(state)` is True.

    Args:
        initial: starting state.
        is_target: termination predicate.
        heuristic: callable with `.evaluate(state) -> float`. Lower = closer to goal.
        max_nodes: hard cap on `nodes_expanded`.
        max_depth: hard cap on path length.
        rules: rule registry; defaults to `default_registry`.
        check_soundness: run `verify_transition` on every (parent, child).
        weight: f = g + weight * h. weight=1.0 is standard A*; >1.0 trades
            optimality for speed (weighted A*).
        problem_id: tag for IllegalStepError diagnostics.
    """
    if rules is None:
        rules = default_registry
    stats = SearchStats()
    t0 = time.perf_counter()

    if is_target(initial):
        stats.time_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(found=True, final_state=initial, path=[], stats=stats)

    counter = 0
    g_score: dict[EqState, int] = {initial: 0}
    parent_map: dict[EqState, tuple[EqState, Action]] = {}
    closed: set[EqState] = set()

    open_heap: list[tuple[float, int, EqState]] = []
    h_initial = float(heuristic.evaluate(initial))
    heapq.heappush(open_heap, (weight * h_initial, counter, initial))
    counter += 1

    while open_heap:
        if stats.nodes_expanded >= max_nodes:
            break
        _, _, state = heapq.heappop(open_heap)
        if state in closed:
            stats.dedup_hits += 1
            continue
        closed.add(state)
        stats.nodes_expanded += 1
        depth = g_score[state]
        if depth >= max_depth:
            continue

        for rule, action in rules.enumerate_actions(state, training_only=training_only):
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

            tentative_g = depth + 1
            if child in g_score and g_score[child] <= tentative_g:
                stats.dedup_hits += 1
                continue
            g_score[child] = tentative_g
            parent_map[child] = (state, action)
            stats.rule_application_counts[rule.name] = (
                stats.rule_application_counts.get(rule.name, 0) + 1
            )
            stats.max_depth_reached = max(stats.max_depth_reached, tentative_g)

            if is_target(child):
                stats.time_ms = (time.perf_counter() - t0) * 1000
                return SearchResult(
                    found=True,
                    final_state=child,
                    path=_reconstruct_path(parent_map, initial, child),
                    stats=stats,
                )

            try:
                h_child = float(heuristic.evaluate(child))
            except Exception:
                h_child = float("inf")
            f = tentative_g + weight * h_child
            heapq.heappush(open_heap, (f, counter, child))
            counter += 1

    stats.time_ms = (time.perf_counter() - t0) * 1000
    return SearchResult(found=False, final_state=None, path=[], stats=stats)
