"""YAML problem loaders for Phase 2 evaluation.

Loads:
- `ggmr/problems/hard_evaluation_set_v2.yaml` (50 motif-template problems)
- `phase0/problems/problems.yaml` (20 phase0 problems, regression set)

Each Problem carries an `is_target` predicate built via
`ggmr/training/extract_pairs.py:_build_is_target` so the target-matching
semantics agree with how the training data was generated -- this guards
against `state.is_canonical_target()` false-positives on multi-root
canonical targets like `(x-3)(x-1)(x+4) = 0`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml

from ggmr.state import EqState
from ggmr.training.extract_pairs import _build_is_target
from ggmr.training.srepr_parse import parse_srepr

HARD_EVAL_V2_PATH = Path(__file__).parent / "hard_evaluation_set_v2.yaml"
PHASE0_PROBLEMS_PATH = Path(__file__).resolve().parents[2] / "phase0" / "problems" / "problems.yaml"


@dataclass
class Problem:
    id: str
    family: str
    initial: EqState
    target: EqState
    is_target: Callable[[EqState], bool]
    baseline_astar_nodes: int  # YAML-stored A* node count (may be stale)
    source: str  # "hard" or "phase0"


def _hard_family(problem_id: str) -> str:
    """hard_motif_v1_ex1_000 -> v1_ex1; hard_motif_L1_017 -> L1."""
    if problem_id.startswith("hard_motif_"):
        rest = problem_id[len("hard_motif_"):]
        m = re.match(r"^(.+?)_\d+$", rest)
        return m.group(1) if m else rest
    return "unknown"


def _build_problem(entry: dict, *, source: str) -> Optional[Problem]:
    """Convert one YAML entry to a Problem. Returns None on parse failure."""
    try:
        var_name = entry["variable"]
        initial = EqState.from_strings(
            entry["initial"]["lhs"],
            entry["initial"]["rhs"],
            var_name=var_name,
        )
        excluded_srepr = entry.get("excluded_srepr") or []
        if excluded_srepr:
            excluded = [parse_srepr(s) for s in excluded_srepr]
            initial = initial.with_excluded(*excluded)
        target = EqState.from_strings(
            entry["canonical_target"]["lhs"],
            entry["canonical_target"]["rhs"],
            var_name=var_name,
        )
        family = (
            _hard_family(entry["id"]) if source == "hard"
            else entry.get("category", "unknown")
        )
        return Problem(
            id=entry["id"],
            family=family,
            initial=initial,
            target=target,
            is_target=_build_is_target(target),
            baseline_astar_nodes=int(entry.get("astar_nodes_expanded", 0)),
            source=source,
        )
    except Exception:
        return None


def load_hard_evaluation_set(path: Path | str | None = None) -> list[Problem]:
    p = Path(path) if path else HARD_EVAL_V2_PATH
    with open(p, "r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    return [pr for pr in (_build_problem(e, source="hard") for e in entries) if pr is not None]


def load_phase0_problems(path: Path | str | None = None) -> list[Problem]:
    p = Path(path) if path else PHASE0_PROBLEMS_PATH
    with open(p, "r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    return [pr for pr in (_build_problem(e, source="phase0") for e in entries) if pr is not None]
