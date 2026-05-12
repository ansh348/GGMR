"""Tests for AC-equivalent variant generator."""

import sympy as sp
from sympy import Eq, Symbol
from sympy.parsing.sympy_parser import parse_expr

from phase0.src.trace_loader import Problem, Step
from phase0.src.variants import (
    assert_variants_equivalent,
    make_variants,
)


x = Symbol("x")


def _p(s: str):
    return parse_expr(s, local_dict={"x": x}, evaluate=False)


def _eq(lhs: str, rhs: str) -> Eq:
    return Eq(_p(lhs), _p(rhs), evaluate=False)


def _sample_problem() -> Problem:
    return Problem(
        id="testprob",
        category="linear",
        variable=x,
        source="test",
        initial=_eq("2*x + 3", "7"),
        canonical_target=_eq("x", "2"),
        trace=(
            Step(rule="SUB3", eq=_eq("2*x", "4")),
            Step(rule="DIV2", eq=_eq("x", "2")),
        ),
    )


def test_make_variants_returns_three():
    p = _sample_problem()
    variants = make_variants(p)
    assert len(variants) == 3
    ids = {v.id for v in variants}
    assert any("var1_addperm" in i for i in ids)
    assert any("var2_mulperm" in i for i in ids)
    assert any("var3_rename" in i for i in ids)


def test_variants_preserve_solution_set():
    p = _sample_problem()
    variants = make_variants(p)
    failures = assert_variants_equivalent(p, variants)
    assert failures == []


def test_variant_rename_changes_variable():
    p = _sample_problem()
    variants = make_variants(p)
    rename_variant = next(v for v in variants if "rename" in v.id)
    assert rename_variant.variable.name != p.variable.name


def test_variants_complex_expression():
    p = Problem(
        id="quad",
        category="quadratic",
        variable=x,
        source="test",
        initial=_eq("x**2 - 5*x + 6", "0"),
        canonical_target=_eq("(x - 2)*(x - 3)", "0"),
        trace=(Step(rule="FACTOR", eq=_eq("(x - 2)*(x - 3)", "0")),),
    )
    variants = make_variants(p)
    failures = assert_variants_equivalent(p, variants)
    assert failures == []
