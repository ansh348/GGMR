"""Extract (state, remaining_steps) training pairs from BFS solution traces."""
from __future__ import annotations

from typing import Optional

import sympy as sp

from ggmr.expr.tree import canonical_repr
from ggmr.rules.core import *  # noqa: F401,F403  (registers forward rules)
from ggmr.search.bfs import bfs
from ggmr.state import EqState


def _state_to_record_partial(state: EqState) -> dict:
    return {
        "state_lhs_srepr": sp.srepr(state.lhs),
        "state_rhs_srepr": sp.srepr(state.rhs),
        "var": state.var.name,
        "excluded_srepr": sorted(sp.srepr(e) for e in state.excluded),
    }


def _build_is_target(target: EqState):
    target_lhs_repr = canonical_repr(target.lhs)
    target_rhs_repr = canonical_repr(target.rhs)
    try:
        target_solset = target.solution_set()
    except Exception:
        target_solset = None

    def is_target(s: EqState) -> bool:
        if (
            canonical_repr(s.lhs) == target_lhs_repr
            and canonical_repr(s.rhs) == target_rhs_repr
        ):
            return True
        if not s.is_canonical_target():
            return False
        if target_solset is None:
            return True
        try:
            return s.solution_set() == target_solset
        except Exception:
            return False

    return is_target


def pairs_from_trace(trace, target: EqState) -> list[dict]:
    """Convert a BFS (state, action) path into N+1 partial-record dicts.

    Records 0..N-1 use trace states with remaining_steps = N - i.
    Record N uses `target` with remaining_steps = 0.
    """
    n = len(trace)
    records: list[dict] = []
    for i, (state, _action) in enumerate(trace):
        rec = _state_to_record_partial(state)
        rec["remaining_steps"] = n - i
        records.append(rec)
    final_rec = _state_to_record_partial(target)
    final_rec["remaining_steps"] = 0
    records.append(final_rec)
    return records


def extract_training_pairs(
    initial: EqState,
    target: EqState,
    *,
    max_nodes: int = 5000,
    max_depth: int = 40,
    training_only: bool = False,
    is_target: Optional[callable] = None,
) -> Optional[list[dict]]:
    """BFS-solve initial → target with check_soundness=False, then walk path.

    `training_only=True` excludes oracle rules (training_safe=False) from
    enumeration — required for trig trace generation (Marcus Constraint 1).

    `is_target` overrides the default target predicate. The default builds
    a per-target canonical-repr matcher; trig trace generation passes a
    domain-specific predicate (canonical_repr(lhs) == canonical_repr(rhs)).
    """
    if is_target is None:
        is_target = _build_is_target(target)
    result = bfs(
        initial,
        is_target,
        max_nodes=max_nodes,
        max_depth=max_depth,
        check_soundness=False,
        problem_id="<training>",
        training_only=training_only,
    )
    if not result.found:
        return None
    return pairs_from_trace(result.path, target)
