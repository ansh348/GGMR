"""Tests for canonical_repr, normalize, traversal utilities, and serializers."""

from __future__ import annotations

import sympy as sp

from ggmr.expr.tree import (
    canonical_repr,
    iter_subtrees,
    leaf_count,
    normalize,
    op_count,
    replace_at_path,
    tree_depth,
)
from ggmr.expr.serialize import (
    from_prefix_notation,
    parse_equation,
    to_prefix_notation,
)


x, y = sp.symbols("x y")


# --- canonical_repr -------------------------------------------------------


def test_canonical_repr_ac_invariance_add():
    """Add args in different orders produce the same canonical_repr."""
    a = sp.Add(x, sp.Integer(1), evaluate=False)
    b = sp.Add(sp.Integer(1), x, evaluate=False)
    assert canonical_repr(a) == canonical_repr(b)


def test_canonical_repr_ac_invariance_mul():
    a = sp.Mul(sp.Integer(2), x, evaluate=False)
    b = sp.Mul(x, sp.Integer(2), evaluate=False)
    assert canonical_repr(a) == canonical_repr(b)


def test_canonical_repr_flattens_nested_add():
    """Add(Add(2x, 3), -7) and Add(2x, 3, -7) produce the same canonical_repr."""
    inner = sp.Add(sp.Integer(2) * x, sp.Integer(3), evaluate=False)
    nested = sp.Add(inner, sp.Integer(-7), evaluate=False)
    flat = sp.Add(sp.Integer(2) * x, sp.Integer(3), sp.Integer(-7), evaluate=False)
    assert canonical_repr(nested) == canonical_repr(flat)


def test_canonical_repr_folds_pure_numerics():
    """Add(3, -7) and Integer(-4) produce the same canonical_repr (numeric folding)."""
    folded = sp.Integer(-4)
    unfolded = sp.Add(sp.Integer(3), sp.Integer(-7), evaluate=False)
    assert canonical_repr(folded) == canonical_repr(unfolded)


def test_canonical_repr_distinguishes_structurally_different():
    a = sp.Add(x, sp.Integer(1), evaluate=False)
    b = sp.Mul(x, sp.Integer(1), evaluate=False)
    assert canonical_repr(a) != canonical_repr(b)


def test_canonical_repr_preserves_pow_structure():
    """(x + 1)^2 should NOT be equal to x^2 + 2x + 1 under canonical_repr;
    expansion is a structural change reserved for explicit rules."""
    pow_form = sp.Pow(sp.Add(x, 1, evaluate=False), 2, evaluate=False)
    expanded = sp.expand(pow_form)
    assert canonical_repr(pow_form) != canonical_repr(expanded)


# --- normalize ------------------------------------------------------------


def test_normalize_flattens_nested_add():
    nested = sp.Add(sp.Add(x, sp.Integer(3), evaluate=False), sp.Integer(-7), evaluate=False)
    flat = normalize(nested)
    assert canonical_repr(flat) == canonical_repr(sp.Add(x, sp.Integer(-4), evaluate=False))


def test_normalize_preserves_pow():
    """Pow(Add(x, 1), 2) is preserved (no expansion)."""
    expr = sp.Pow(sp.Add(x, 1, evaluate=False), 2, evaluate=False)
    n = normalize(expr)
    assert isinstance(n, sp.Pow)


# --- tree_depth / op_count / leaf_count (Phase 0 parity) ------------------


def test_phase0_metric_parity_simple():
    expr = sp.Add(sp.Mul(sp.Integer(2), x, evaluate=False), sp.Integer(3), evaluate=False)
    assert tree_depth(expr) == 3
    assert op_count(expr) == 2
    assert leaf_count(expr) == 3


# --- iter_subtrees / replace_at_path -------------------------------------


def test_iter_subtrees_yields_root():
    expr = sp.Add(x, sp.Integer(1), evaluate=False)
    paths = [p for p, _ in iter_subtrees(expr)]
    assert () in paths


def test_iter_subtrees_yields_children():
    expr = sp.Add(sp.Mul(sp.Integer(2), x, evaluate=False), sp.Integer(3), evaluate=False)
    paths = [p for p, _ in iter_subtrees(expr)]
    # root, two args, x and 2 inside the Mul, integer 3
    assert (0,) in paths and (1,) in paths


def test_replace_at_path_root():
    expr = sp.Add(x, sp.Integer(1), evaluate=False)
    new = replace_at_path(expr, (), sp.Integer(0))
    assert new == sp.Integer(0)


def test_replace_at_path_subtree():
    expr = sp.Add(sp.Mul(sp.Integer(2), x, evaluate=False), sp.Integer(3), evaluate=False)
    new = replace_at_path(expr, (0,), sp.Integer(7))
    # Replaced the Mul(2, x) with 7; result should be Add(7, 3)
    assert canonical_repr(new) == canonical_repr(sp.Add(sp.Integer(7), sp.Integer(3), evaluate=False))


# --- prefix-notation roundtrip -------------------------------------------


def test_prefix_roundtrip_simple():
    expr = sp.Add(sp.Mul(sp.Integer(2), x, evaluate=False), sp.Integer(3), evaluate=False)
    tokens = to_prefix_notation(expr)
    rebuilt = from_prefix_notation(tokens)
    assert canonical_repr(rebuilt) == canonical_repr(expr)


def test_prefix_roundtrip_pow():
    expr = sp.Pow(sp.Add(x, 1, evaluate=False), 2, evaluate=False)
    tokens = to_prefix_notation(expr)
    rebuilt = from_prefix_notation(tokens)
    assert canonical_repr(rebuilt) == canonical_repr(expr)


def test_prefix_roundtrip_rational():
    expr = sp.Rational(2, 3)
    tokens = to_prefix_notation(expr)
    rebuilt = from_prefix_notation(tokens)
    assert rebuilt == expr


def test_parse_equation():
    lhs, rhs = parse_equation("2*x + 3 = 7")
    assert canonical_repr(lhs) == canonical_repr(
        sp.Add(sp.Mul(sp.Integer(2), x, evaluate=False), sp.Integer(3), evaluate=False)
    )
    assert rhs == sp.Integer(7)
