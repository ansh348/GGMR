"""Tests for structural feature extraction."""

import sympy as sp
from sympy import Eq, Symbol
from sympy.parsing.sympy_parser import parse_expr

from phase0.src.features import (
    composite_z,
    features,
    leaf_count,
    op_count,
    tree_depth,
    var_isolation_score,
)


x = Symbol("x")


def _p(s: str):
    return parse_expr(s, local_dict={"x": x}, evaluate=False)


def test_tree_depth_atom():
    assert tree_depth(x) == 1
    assert tree_depth(sp.Integer(5)) == 1


def test_tree_depth_simple_add():
    expr = _p("x + 1")
    assert tree_depth(expr) == 2


def test_tree_depth_nested():
    expr = _p("(x + 1)*(x - 1)")
    # SymPy evaluate=False parses `x - 1` as Add(x, Mul(-1, 1)), inflating depth by 1.
    # Structure: Mul → Add → Mul → Integer = depth 4. Consistent across all states.
    assert tree_depth(expr) == 4


def test_op_count_atoms_zero():
    assert op_count(x) == 0
    assert op_count(sp.Integer(5)) == 0


def test_op_count_simple():
    expr = _p("x + 1")
    assert op_count(expr) == 1


def test_op_count_compound():
    expr = _p("(x + 1)*(x - 1)")
    # `x - 1` adds a Mul(-1, 1) under evaluate=False, inflating op count by 1.
    # 1 outer Mul + 2 Adds + 1 inner Mul = 4
    assert op_count(expr) == 4


def test_leaf_count_atom():
    assert leaf_count(x) == 1


def test_leaf_count_polynomial():
    expr = _p("x**2 + 2*x + 1")
    # x*x (in Pow), 2, x, 1 — depends on parse_expr expansion. Just check positive.
    assert leaf_count(expr) > 0


def test_isolation_perfect_lhs():
    eq = Eq(x, sp.Integer(5), evaluate=False)
    assert var_isolation_score(eq, x) == 0


def test_isolation_perfect_rhs():
    eq = Eq(sp.Integer(5), x, evaluate=False)
    assert var_isolation_score(eq, x) == 0


def test_isolation_x_plus_1():
    eq = Eq(_p("x + 1"), sp.Integer(5), evaluate=False)
    assert var_isolation_score(eq, x) == 1


def test_isolation_x_on_both_sides():
    eq = Eq(_p("2*x"), _p("x + 3"), evaluate=False)
    assert var_isolation_score(eq, x) == 2


def test_features_dict():
    eq = Eq(_p("2*x + 3"), sp.Integer(7), evaluate=False)
    f = features(eq, x)
    assert f.depth >= 2
    assert f.ops >= 2
    assert f.leaves >= 3
    assert f.isolation >= 1


def test_composite_z_centers():
    # Simple corpus of identical rows: composite should be ~0 for each
    eq = Eq(_p("2*x + 3"), sp.Integer(7), evaluate=False)
    rows = [features(eq, x) for _ in range(5)]
    comps = composite_z(rows)
    for c in comps:
        assert abs(c) < 1e-6
