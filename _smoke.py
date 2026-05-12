"""Smoke test: run BFS on each Phase 0 problem and report outcome."""

from __future__ import annotations

import sys

import sympy as sp
import yaml

from ggmr.rules.core import *  # noqa: F403,F401  (registers rules)
from ggmr.search.bfs import bfs
from ggmr.state import EqState


def _load_phase0() -> list[dict]:
    with open("phase0/problems/problems.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_state(entry: dict) -> EqState:
    return EqState.from_strings(
        entry["initial"]["lhs"], entry["initial"]["rhs"], var_name=entry["variable"]
    )


def _target_state(entry: dict) -> EqState:
    return EqState.from_strings(
        entry["canonical_target"]["lhs"],
        entry["canonical_target"]["rhs"],
        var_name=entry["variable"],
    )


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # flush per print
    only = sys.argv[1] if len(sys.argv) > 1 else None
    problems = _load_phase0()
    if only:
        problems = [p for p in problems if p["id"] == only]
    from ggmr.expr.tree import canonical_repr

    n_pass = 0
    for entry in problems:
        pid = entry["id"]
        initial = _make_state(entry)
        target = _target_state(entry)
        target_lhs_key = canonical_repr(target.lhs)
        target_rhs_key = canonical_repr(target.rhs)
        target_solset = target.solution_set()

        def is_target(
            s: EqState,
            _l: str = target_lhs_key,
            _r: str = target_rhs_key,
            _ss: frozenset = target_solset,
        ) -> bool:
            # Primary: structural canonical_repr match with the YAML target.
            if canonical_repr(s.lhs) == _l and canonical_repr(s.rhs) == _r:
                return True
            # Secondary: any canonical end-state with the same effective solution set.
            if s.is_canonical_target() and s.solution_set() == _ss:
                return True
            return False

        try:
            result = bfs(
                initial, is_target, max_nodes=5_000, max_depth=12, problem_id=pid
            )
        except Exception as e:
            print(f"  {pid:8s}  EXCEPTION: {type(e).__name__}: {e}")
            continue
        if result.found:
            n_pass += 1
            print(
                f"  {pid:8s}  PASS  steps={result.num_steps}  "
                f"expanded={result.stats.nodes_expanded}  "
                f"time={result.stats.time_ms:.0f}ms"
            )
            if only:
                for i, (s, a) in enumerate(result.path):
                    label = f"{a.rule_name}({a.params})" if a.params else a.rule_name
                    print(f"    step {i}: {s.lhs} = {s.rhs}  --[{label}]-->")
                print(
                    f"    final: {result.final_state.lhs} = {result.final_state.rhs}"
                )
        else:
            print(
                f"  {pid:8s}  FAIL  expanded={result.stats.nodes_expanded}  "
                f"depth={result.stats.max_depth_reached}  "
                f"time={result.stats.time_ms:.0f}ms"
            )
    print(f"\n{n_pass}/{len(problems)} solved")


if __name__ == "__main__":
    main()
