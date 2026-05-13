"""Solution-set soundness predicates. Graduates `phase0/src/verifier.py`
into the production package without modification of phase0.

A transition (s_t -> s_{t+1}) is sound iff:
    solution_set(s_{t+1}) ⊆ solution_set(s_t)

Equality is the dominant case. Strict subset is allowed when a rule legitimately
removes an extraneous root (e.g., CANCEL_COMMON_FACTOR removing a factor
at a value already excluded by a guard).
"""

from __future__ import annotations

import sympy as sp
from sympy import Eq, Symbol


class IllegalStepError(Exception):
    """Raised when a (state, action, next_state) triple violates soundness."""

    def __init__(self, problem_id: str, step_idx: int, reason: str):
        self.problem_id = problem_id
        self.step_idx = step_idx
        self.reason = reason
        super().__init__(f"{problem_id} step {step_idx}: {reason}")


def safe_solve(expr, var: Symbol, **kwargs) -> list:
    """Solve `expr = 0` (or `expr` directly) for `var`. Falls back to
    `solveset(..., domain=Reals)` on NotImplementedError. Returns a list of
    solutions (possibly empty). Forwards kwargs (e.g., `rational=True`) to
    `sp.solve`; kwargs are dropped for the solveset fallback path.
    """
    try:
        sols = sp.solve(expr, var, **kwargs)
        if isinstance(sols, dict):
            sols = list(sols.values())
        return sols
    except NotImplementedError:
        ss = sp.solveset(expr, var, domain=sp.S.Reals)
        if ss == sp.S.EmptySet:
            return []
        if hasattr(ss, "is_FiniteSet") and ss.is_FiniteSet:
            return list(ss)
        raise


def solution_set(lhs: sp.Expr, rhs: sp.Expr, var: Symbol) -> frozenset:
    """Solve lhs = rhs for var; return the simplified solution set as a frozenset.

    Falls back to `sp.solveset(..., domain=Reals)` when `solve()` raises
    NotImplementedError. This covers absolute-value equations like `|x| = 5`,
    where solve() refuses on the un-declared real-domain symbol.
    """
    eq = Eq(lhs, rhs, evaluate=False)
    try:
        sols = sp.solve(eq, var)
        if isinstance(sols, dict):
            sols = list(sols.values())
    except NotImplementedError:
        ss = sp.solveset(eq, var, domain=sp.S.Reals)
        if ss == sp.S.EmptySet:
            sols = []
        elif hasattr(ss, "is_FiniteSet") and ss.is_FiniteSet:
            sols = list(ss)
        else:
            raise
    return frozenset(sp.simplify(s) for s in sols)


VERIFY_PASS = "pass"
VERIFY_UNVERIFIABLE = "unverifiable"
VERIFY_UNSOUND = "unsound"


def _effective_set(lhs: sp.Expr, rhs: sp.Expr, var: Symbol, excluded: frozenset) -> frozenset:
    raw = solution_set(lhs, rhs, var)
    if not excluded:
        return raw
    excluded_simplified = frozenset(sp.simplify(e) for e in excluded)
    return frozenset(s for s in raw if s not in excluded_simplified)


def verify_transition(
    parent_lhs: sp.Expr,
    parent_rhs: sp.Expr,
    child_lhs: sp.Expr,
    child_rhs: sp.Expr,
    var: Symbol,
    parent_excluded: frozenset = frozenset(),
    child_excluded: frozenset = frozenset(),
) -> tuple[str, str]:
    """Three-state verifier:
        VERIFY_PASS         — confirmed sound (effective sets equal, or child ⊊ parent)
        VERIFY_UNVERIFIABLE — solve raised; caller should skip
        VERIFY_UNSOUND      — confirmed unsound

    `excluded` arguments are sets of values forbidden by guards on each state.
    The effective solution set is `solve(eq) − excluded`.
    """
    try:
        s_parent = _effective_set(parent_lhs, parent_rhs, var, parent_excluded)
    except Exception as e:
        return VERIFY_UNVERIFIABLE, f"solve(parent) raised {type(e).__name__}: {e}"
    try:
        s_child = _effective_set(child_lhs, child_rhs, var, child_excluded)
    except Exception as e:
        return VERIFY_UNVERIFIABLE, f"solve(child) raised {type(e).__name__}: {e}"
    if s_parent == s_child:
        return VERIFY_PASS, ""
    if s_child.issubset(s_parent):
        return VERIFY_PASS, "subset (cancellation removed extraneous root)"
    return VERIFY_UNSOUND, f"solution sets diverge: parent={s_parent} child={s_child}"
