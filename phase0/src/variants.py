"""AC-equivalent variant generator for Phase 0 fragility test.

For each problem, generate 3 deterministic variants:
  v1: permute Add args (depth ≤ 2)
  v2: permute Mul args (depth ≤ 2)
  v3: rename target variable (x → y)

Each variant must preserve the solution set; this is asserted at generation time.
"""

from __future__ import annotations

import random
from dataclasses import replace

import sympy as sp
from sympy import Add, Eq, Expr, Mul, Symbol

from .trace_loader import Problem, Step

_SEED = 20260510  # deterministic per PHASE0_PREREG.md §10


def _permute_args(expr: Expr, op_type: type, max_depth: int, rng: random.Random) -> Expr:
    """Recursively walk `expr`; for nodes of type `op_type` at depth ≤ max_depth,
    randomly permute their args. Other nodes are unchanged."""

    def walk(node: Expr, depth: int) -> Expr:
        if not node.args:
            return node
        new_args = tuple(walk(a, depth + 1) for a in node.args)
        if isinstance(node, op_type) and depth <= max_depth and len(new_args) >= 2:
            shuffled = list(new_args)
            rng.shuffle(shuffled)
            return node.func(*shuffled, evaluate=False)
        return node.func(*new_args, evaluate=False) if new_args != node.args else node

    return walk(expr, 0)


def _rename_var(expr: Expr, old: Symbol, new: Symbol) -> Expr:
    """Substitute `old` with `new`. SymPy's xreplace preserves structure better
    than subs for our purposes (no auto-simplification)."""
    return expr.xreplace({old: new})


def _variant_eq(eq: Eq, transform) -> Eq:
    return Eq(transform(eq.lhs), transform(eq.rhs), evaluate=False)


def _check_equivalent(orig: Eq, variant: Eq, var_orig: Symbol, var_new: Symbol) -> bool:
    """Both equations should have equal solution sets (after variable rename)."""
    try:
        sols_orig = set(map(sp.simplify, sp.solve(orig, var_orig)))
        sols_var = set(map(sp.simplify, sp.solve(variant, var_new)))
        return sols_orig == sols_var
    except Exception:
        return False


def make_add_permuted(problem: Problem) -> Problem:
    rng = random.Random(_SEED + hash(problem.id) % 10000)
    transform = lambda e: _permute_args(e, Add, max_depth=2, rng=rng)
    initial = _variant_eq(problem.initial, transform)
    target = _variant_eq(problem.canonical_target, transform)
    new_steps = tuple(
        Step(rule=s.rule, eq=_variant_eq(s.eq, transform), guard=s.guard)
        for s in problem.trace
    )
    return replace(
        problem,
        id=f"{problem.id}_var1_addperm",
        initial=initial,
        canonical_target=target,
        trace=new_steps,
    )


def make_mul_permuted(problem: Problem) -> Problem:
    rng = random.Random(_SEED + 1 + hash(problem.id) % 10000)
    transform = lambda e: _permute_args(e, Mul, max_depth=2, rng=rng)
    initial = _variant_eq(problem.initial, transform)
    target = _variant_eq(problem.canonical_target, transform)
    new_steps = tuple(
        Step(rule=s.rule, eq=_variant_eq(s.eq, transform), guard=s.guard)
        for s in problem.trace
    )
    return replace(
        problem,
        id=f"{problem.id}_var2_mulperm",
        initial=initial,
        canonical_target=target,
        trace=new_steps,
    )


def make_var_renamed(problem: Problem) -> Problem:
    """Rename the target variable. If it's already `y`, use `z`; otherwise `y`."""
    new_var = sp.Symbol("y" if problem.variable.name != "y" else "z")
    transform = lambda e: _rename_var(e, problem.variable, new_var)
    initial = _variant_eq(problem.initial, transform)
    target = _variant_eq(problem.canonical_target, transform)
    new_steps = tuple(
        Step(rule=s.rule, eq=_variant_eq(s.eq, transform), guard=s.guard)
        for s in problem.trace
    )
    return replace(
        problem,
        id=f"{problem.id}_var3_rename",
        variable=new_var,
        initial=initial,
        canonical_target=target,
        trace=new_steps,
    )


def make_variants(problem: Problem) -> list[Problem]:
    """Return the 3 AC-equivalent variants of a problem."""
    return [
        make_add_permuted(problem),
        make_mul_permuted(problem),
        make_var_renamed(problem),
    ]


def assert_variants_equivalent(problem: Problem, variants: list[Problem]) -> list[str]:
    """Verify each variant has the same solution set as the original.
    Returns a list of failure messages."""
    failures: list[str] = []
    for v in variants:
        if not _check_equivalent(problem.initial, v.initial, problem.variable, v.variable):
            failures.append(
                f"{v.id}: variant initial state has different solutions from original"
            )
    return failures
