"""YAML emission/loading for hard evaluation set.

Uses `sp.srepr` as the authoritative form of `initial.lhs` / `initial.rhs`
because `parse_expr(..., evaluate=False)` does NOT round-trip disguised
expressions like `Add(x, 3, -3, evaluate=False)` (the parser collapses
numeric subtrees back to `x`, undoing the disguise).

Schema (additive over Phase 0 schema):

    - id: hard_<recipe>_<NNN>
      category: <template-name>
      recipe: <recipe-name>
      difficulty: hard
      variable: x
      source: ggmr-hard-generator-v1
      seed: <int>
      depth: <int>
      astar_nodes_expanded: <int>
      bfs_nodes_expanded: <int>
      applied_inverses: [<inverse-name>, ...]
      initial_srepr: "<sp.srepr of full Eq>"
      initial:               # pretty-print, human-readable only
        lhs: "..."
        rhs: "..."
      canonical_target:
        lhs: "..."
        rhs: "..."
      excluded: [<srepr>, ...]  # values forbidden by inverse-rule guards
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp
import yaml

from ..state import EqState
from .generator import GeneratedProblem


@dataclass
class HardProblemRecord:
    """Decoded form of one entry in hard_evaluation_set.yaml."""

    id: str
    category: str
    recipe: str
    seed: int
    depth: int
    astar_nodes_expanded: int
    bfs_nodes_expanded: int
    applied_inverses: list
    initial: EqState
    canonical_target: EqState


def _expr_to_str(expr) -> str:
    return str(expr)


def _expr_to_srepr(expr) -> str:
    return sp.srepr(expr)


def hard_problem_to_dict(
    problem: GeneratedProblem,
    problem_id: str,
    recipe: str,
) -> dict:
    initial = problem.initial
    target = problem.target
    return {
        "id": problem_id,
        "category": problem.template,
        "recipe": recipe,
        "difficulty": "hard",
        "variable": initial.var.name,
        "source": "ggmr-hard-generator-v1",
        "seed": problem.seed,
        "depth": problem.depth,
        "astar_nodes_expanded": int(problem.astar_nodes_expanded),
        "bfs_nodes_expanded": int(problem.bfs_stats.get("nodes_expanded", 0)),
        "applied_inverses": [a.inverse_name for a in problem.applied_inverses],
        "initial_srepr_lhs": _expr_to_srepr(initial.lhs),
        "initial_srepr_rhs": _expr_to_srepr(initial.rhs),
        "excluded_srepr": sorted(_expr_to_srepr(e) for e in initial.excluded),
        "initial": {
            "lhs": _expr_to_str(initial.lhs),
            "rhs": _expr_to_str(initial.rhs),
        },
        "canonical_target": {
            "lhs": _expr_to_str(target.lhs),
            "rhs": _expr_to_str(target.rhs),
        },
    }


def emit_hard_problems_yaml(records: Sequence[dict], file_path: str) -> None:
    """Write a list of pre-serialized hard-problem dicts to YAML."""
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(list(records), f, sort_keys=False, allow_unicode=True)


def _sympify_srepr(srepr_str: str):
    """Reconstruct a sympy expression from its srepr form."""
    return sp.sympify(srepr_str)


def load_hard_problems_yaml(file_path: str) -> list[HardProblemRecord]:
    """Load and decode a hard-problems YAML file into HardProblemRecord list."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    out: list[HardProblemRecord] = []
    for entry in raw:
        var = sp.Symbol(entry.get("variable", "x"))
        lhs = _sympify_srepr(entry["initial_srepr_lhs"])
        rhs = _sympify_srepr(entry["initial_srepr_rhs"])
        excluded = frozenset(
            _sympify_srepr(s) for s in entry.get("excluded_srepr", [])
        )
        initial = EqState(lhs=lhs, rhs=rhs, var=var, excluded=excluded)
        tgt = entry["canonical_target"]
        # canonical_target is pretty-printed; parse_expr is safe here because
        # canonical targets are simple (e.g., "x" and a constant)
        from sympy.parsing.sympy_parser import parse_expr

        target_lhs = parse_expr(tgt["lhs"], local_dict={var.name: var}, evaluate=False)
        target_rhs = parse_expr(tgt["rhs"], local_dict={var.name: var}, evaluate=False)
        target = EqState(lhs=target_lhs, rhs=target_rhs, var=var)
        out.append(
            HardProblemRecord(
                id=entry["id"],
                category=entry["category"],
                recipe=entry.get("recipe", "unknown"),
                seed=int(entry.get("seed", 0)),
                depth=int(entry.get("depth", 0)),
                astar_nodes_expanded=int(entry.get("astar_nodes_expanded", 0)),
                bfs_nodes_expanded=int(entry.get("bfs_nodes_expanded", 0)),
                applied_inverses=list(entry.get("applied_inverses", [])),
                initial=initial,
                canonical_target=target,
            )
        )
    return out
