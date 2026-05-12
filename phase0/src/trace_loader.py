"""Load Phase 0 problems from YAML and parse expressions with SymPy.

Each problem is `{id, category, variable, source, initial, canonical_target, trace}`.
Expressions are stored as separate `lhs` / `rhs` strings to avoid `=` parsing issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sympy as sp
import yaml
from sympy import Eq, Symbol
from sympy.parsing.sympy_parser import parse_expr


@dataclass(frozen=True)
class Step:
    rule: str
    eq: Eq
    guard: str | None = None


@dataclass(frozen=True)
class Problem:
    id: str
    category: str
    variable: Symbol
    source: str
    initial: Eq
    canonical_target: Eq
    trace: tuple[Step, ...]
    notes: str = ""

    @property
    def states(self) -> list[Eq]:
        """All states along the trace, including the initial state."""
        return [self.initial, *(s.eq for s in self.trace)]


def _parse_eq(d: dict, local_dict: dict) -> Eq:
    """Parse a {lhs, rhs} dict into a SymPy Eq.

    Uses evaluate=False to retain structural form for feature analysis.
    """
    lhs = parse_expr(d["lhs"], local_dict=local_dict, evaluate=False)
    rhs = parse_expr(d["rhs"], local_dict=local_dict, evaluate=False)
    return Eq(lhs, rhs, evaluate=False)


def _final_target_match(eq: Eq, target: Eq, var: Symbol) -> bool:
    """Check the final state matches the canonical target by solution-set equality.

    Uses sympy.solve to compare; falls back to direct simplification equality
    if solve returns inconsistent results across the two equations.
    """
    try:
        s_eq = set(map(sp.simplify, sp.solve(eq, var)))
        s_tg = set(map(sp.simplify, sp.solve(target, var)))
        if s_eq == s_tg:
            return True
    except (NotImplementedError, ValueError):
        pass
    # Fallback: structural simplification
    return sp.simplify((eq.lhs - eq.rhs) - (target.lhs - target.rhs)) == 0 \
        or sp.simplify((eq.lhs - eq.rhs) + (target.lhs - target.rhs)) == 0


def load_problems(path: str | Path) -> list[Problem]:
    """Load and parse the problems YAML, returning typed Problem objects."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    problems: list[Problem] = []
    for entry in raw:
        var_name = entry["variable"]
        var = sp.Symbol(var_name)
        local = {var_name: var}
        initial = _parse_eq(entry["initial"], local)
        canonical = _parse_eq(entry["canonical_target"], local)
        steps: list[Step] = []
        for step_entry in entry["trace"]:
            steps.append(
                Step(
                    rule=step_entry["rule"],
                    eq=_parse_eq(step_entry, local),
                    guard=step_entry.get("guard"),
                )
            )
        problems.append(
            Problem(
                id=entry["id"],
                category=entry["category"],
                variable=var,
                source=entry.get("source", ""),
                initial=initial,
                canonical_target=canonical,
                trace=tuple(steps),
                notes=entry.get("notes", ""),
            )
        )

    return problems


def validate_canonical_targets(problems: list[Problem]) -> list[str]:
    """Check that each trace ends at the canonical target. Returns failure messages."""
    failures: list[str] = []
    for p in problems:
        if not p.trace:
            failures.append(f"{p.id}: empty trace")
            continue
        final = p.trace[-1].eq
        if not _final_target_match(final, p.canonical_target, p.variable):
            failures.append(
                f"{p.id}: final state {final} does not match "
                f"canonical_target {p.canonical_target} by solution-set equality"
            )
    return failures
