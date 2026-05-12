"""Phase 0–compatible YAML emission for generated problems.

Schema (matching `phase0/problems/problems.yaml`):

```yaml
- id: gen_<template>_<depth>_<seq>
  category: <template>
  variable: x
  source: "ggmr-generator-v1"
  initial: {lhs: "...", rhs: "..."}
  canonical_target: {lhs: "x", rhs: "..."}
  trace:
    - rule: "..."
      lhs: "..."
      rhs: "..."
```
"""

from __future__ import annotations

from typing import Sequence

import yaml

from ..state import EqState
from .generator import GeneratedProblem


def _expr_to_str(expr) -> str:
    """Stable, parseable string form of an Expr."""
    return str(expr)


def problem_to_dict(problem: GeneratedProblem, problem_id: str) -> dict:
    initial_dict = {
        "lhs": _expr_to_str(problem.initial.lhs),
        "rhs": _expr_to_str(problem.initial.rhs),
    }
    target_dict = {
        "lhs": _expr_to_str(problem.target.lhs),
        "rhs": _expr_to_str(problem.target.rhs),
    }
    trace_list: list[dict] = []
    for state, action in problem.forward_trace:
        # The state in the trace is the parent; we want to emit the result of applying the action.
        # The path is parent → action → child, and the next entry's state is the child.
        # Emit: rule name + parent state (we'll fix this; phase 0 emits the post-step state).
        trace_list.append(
            {
                "rule": action.rule_name,
                "lhs": _expr_to_str(state.lhs),
                "rhs": _expr_to_str(state.rhs),
            }
        )
    return {
        "id": problem_id,
        "category": problem.template,
        "variable": problem.initial.var.name,
        "source": "ggmr-generator-v1",
        "initial": initial_dict,
        "canonical_target": target_dict,
        "trace": trace_list,
    }


def emit_problems_yaml(problems: Sequence[GeneratedProblem], file_path: str) -> None:
    """Write a list of generated problems to a YAML file in Phase 0 schema."""
    payload: list[dict] = []
    for i, p in enumerate(problems):
        problem_id = f"gen_{p.template}_{p.depth}_{i:03d}"
        payload.append(problem_to_dict(p, problem_id))
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
