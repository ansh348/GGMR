"""BFS search engine with deduplication. D1-baseline scaffolding per
`ggmr_v10.pdf` §5.3.

Soundness assertion is enabled by default in Phase 1a per pre-reg §3.2: every
expanded child is checked for solution-set ⊆ parent. Phase 1b adds an opt-out
for performance.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from ..expr.tree import normalize
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
from .stats import SearchStats


@dataclass
class SearchResult:
    found: bool
    final_state: Optional[EqState]
    path: list[tuple[EqState, Action]]
    stats: SearchStats

    @property
    def num_steps(self) -> int:
        return len(self.path)


def bfs(
    initial: EqState,
    is_target: Callable[[EqState], bool],
    *,
    max_nodes: int = 50_000,
    max_depth: int = 20,
    rules: Optional[Registry] = None,
    check_soundness: bool = True,
    problem_id: str = "<bfs>",
) -> SearchResult:
    """Breadth-first search from `initial` until `is_target(state)` is True.

    Args:
        initial: starting state.
        is_target: termination predicate. Typically problem-specific.
        max_nodes: hard cap on `nodes_expanded`.
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

    visited: set[EqState] = {initial}
    parent_map: dict[EqState, tuple[EqState, Action]] = {}
    queue: deque[tuple[EqState, int]] = deque([(initial, 0)])

    while queue:
        if stats.nodes_expanded >= max_nodes:
            break
        state, depth = queue.popleft()
        stats.nodes_expanded += 1
        if depth >= max_depth:
            continue
        for rule, action in rules.enumerate_actions(state):
            stats.nodes_generated += 1
            guard = rule.guard(state, action)
            if not guard.ok:
                stats.guard_rejections += 1
                continue
            try:
                child = rule.apply(state, action)
            except Exception:
                # an apply failure is a bug; skip rather than aborting BFS,
                # but it is reflected in guard_rejections for diagnostics
                stats.guard_rejections += 1
                continue
            child = merge_guard_into_state(child, guard) if (guard.new_excluded or guard.new_side_conditions) else child
            # Normalize lhs/rhs: flatten nested Add/Mul + fold pure-numeric
            # subtrees. Keeps the BFS state space tight without expanding products.
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
                    # Degenerate state (typically division-by-zero artifacts);
                    # skip rather than abort. Counted in guard_rejections for stats.
                    stats.guard_rejections += 1
                    continue
            visited.add(child)
            parent_map[child] = (state, action)
            stats.rule_application_counts[rule.name] = (
                stats.rule_application_counts.get(rule.name, 0) + 1
            )
            stats.max_depth_reached = max(stats.max_depth_reached, depth + 1)
            if is_target(child):
                stats.time_ms = (time.perf_counter() - t0) * 1000
                return SearchResult(
                    found=True,
                    final_state=child,
                    path=_reconstruct_path(parent_map, initial, child),
                    stats=stats,
                )
            queue.append((child, depth + 1))

    stats.time_ms = (time.perf_counter() - t0) * 1000
    return SearchResult(found=False, final_state=None, path=[], stats=stats)


def _reconstruct_path(
    parent_map: dict[EqState, tuple[EqState, Action]],
    initial: EqState,
    final: EqState,
) -> list[tuple[EqState, Action]]:
    """Reconstruct (state, action) pairs from initial to final, in order."""
    rev: list[tuple[EqState, Action]] = []
    cur = final
    while cur != initial:
        parent, action = parent_map[cur]
        rev.append((parent, action))
        cur = parent
    rev.reverse()
    return rev
