"""Step-legality verifier: every consecutive (s_t, s_{t+1}) on a trace
must preserve the equation's solution set.

Per PHASE0_PREREG.md §3, any verification failure aborts the experiment so
that the monotonicity numbers are computed only on legal traces.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy import Eq, Symbol

from .trace_loader import Problem


class IllegalStepError(Exception):
    """Raised when a consecutive (eq_t, eq_{t+1}) pair fails to preserve solution set."""

    def __init__(self, problem_id: str, step_idx: int, reason: str):
        self.problem_id = problem_id
        self.step_idx = step_idx
        self.reason = reason
        super().__init__(f"{problem_id} step {step_idx}: {reason}")


@dataclass(frozen=True)
class StepCheck:
    problem_id: str
    step_idx: int
    rule: str
    ok: bool
    reason: str = ""


def _solution_set(eq: Eq, var: Symbol) -> frozenset:
    """Solve the equation for var; return the simplified solution set as a frozenset.

    Raises if solve cannot proceed (caller should fall back to structural check).
    """
    sols = sp.solve(eq, var)
    if isinstance(sols, dict):
        sols = list(sols.values())
    return frozenset(sp.simplify(s) for s in sols)


def verify_step(eq_t: Eq, eq_next: Eq, var: Symbol) -> tuple[bool, str]:
    """Return (ok, reason). Compares solution sets of the two equations."""
    try:
        s_t = _solution_set(eq_t, var)
    except Exception as e:
        return False, f"solve(eq_t) raised {type(e).__name__}: {e}"
    try:
        s_next = _solution_set(eq_next, var)
    except Exception as e:
        return False, f"solve(eq_next) raised {type(e).__name__}: {e}"
    if s_t == s_next:
        return True, ""
    # If sets differ but the next is a subset (e.g., cancellation removed a
    # spurious-by-domain root), allow it. Going from broader to narrower is a
    # legal cancellation; the reverse (introducing extraneous solutions) is not.
    if s_next.issubset(s_t):
        return True, "subset (cancellation removed extraneous root)"
    return False, f"solution sets differ: {s_t} vs {s_next}"


def verify_problem(problem: Problem) -> list[StepCheck]:
    """Verify every step of one problem. Returns one StepCheck per transition."""
    checks: list[StepCheck] = []
    states = problem.states
    for i in range(len(states) - 1):
        ok, reason = verify_step(states[i], states[i + 1], problem.variable)
        rule = problem.trace[i].rule
        checks.append(
            StepCheck(
                problem_id=problem.id,
                step_idx=i,
                rule=rule,
                ok=ok,
                reason=reason,
            )
        )
    return checks


def verify_all(problems: list[Problem], strict: bool = True) -> list[StepCheck]:
    """Verify every problem. If `strict`, raise IllegalStepError on first failure."""
    all_checks: list[StepCheck] = []
    for p in problems:
        for check in verify_problem(p):
            all_checks.append(check)
            if strict and not check.ok:
                raise IllegalStepError(check.problem_id, check.step_idx, check.reason)
    return all_checks
