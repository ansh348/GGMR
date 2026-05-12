"""Expression tree primitives over SymPy: canonical hashing, structural metrics,
preorder traversal with paths, immutable replacement at a path.

`canonical_repr` is the load-bearing dedup hash for BFS. It must be:
- Deterministic
- AC-canonical (Add/Mul args sorted by srepr of children, recursively)
- Stable across runs and across SymPy auto-canonicalization variations

Per `ggmr_v10.pdf` §2.2, structural parameterization for rules uses index paths
into the expression tree. `iter_subtrees` and `replace_at_path` provide the
walk and rewrite primitives.
"""

from __future__ import annotations

from typing import Iterator

import sympy as sp
from sympy import Add, Expr, Mul


def _flatten_add_mul(expr: Expr) -> Expr:
    """Recursively flatten nested Add/Mul of same class, preserving evaluate=False."""
    if not expr.args:
        return expr
    new_args = tuple(_flatten_add_mul(a) for a in expr.args)
    if isinstance(expr, Add):
        flat: list = []
        for a in new_args:
            if isinstance(a, Add):
                flat.extend(a.args)
            else:
                flat.append(a)
        return Add(*flat, evaluate=False) if len(flat) >= 2 else (flat[0] if flat else sp.Integer(0))
    if isinstance(expr, Mul):
        flat = []
        for a in new_args:
            if isinstance(a, Mul):
                flat.extend(a.args)
            else:
                flat.append(a)
        return Mul(*flat, evaluate=False) if len(flat) >= 2 else (flat[0] if flat else sp.Integer(1))
    # Some sympy types (e.g. ComplexRootOf) reject evaluate kwarg; fall back gracefully.
    try:
        return expr.func(*new_args, evaluate=False)
    except TypeError:
        try:
            return expr.func(*new_args)
        except TypeError:
            return expr


def _fold_numeric_subtrees(expr: Expr) -> Expr:
    """Fold pure-numeric Add/Mul subtrees to single numbers, recursively.

    `Add(Integer(3), Integer(-7)) -> Integer(-4)`. Symbolic terms are
    preserved. Does not expand: `Mul(Integer(2), Add(x, 1))` stays Mul.
    """
    if not expr.args:
        return expr
    new_args = tuple(_fold_numeric_subtrees(a) for a in expr.args)
    if isinstance(expr, Add):
        numeric_sum = sp.Integer(0)
        non_numeric: list = []
        for a in new_args:
            if a.is_number:
                numeric_sum = numeric_sum + a
            else:
                non_numeric.append(a)
        if numeric_sum != 0:
            non_numeric.append(numeric_sum)
        if not non_numeric:
            return sp.Integer(0)
        if len(non_numeric) == 1:
            return non_numeric[0]
        return Add(*non_numeric, evaluate=False)
    if isinstance(expr, Mul):
        numeric_prod = sp.Integer(1)
        non_numeric = []
        for a in new_args:
            if a.is_number:
                numeric_prod = numeric_prod * a
            else:
                non_numeric.append(a)
        if numeric_prod == 0:
            return sp.Integer(0)
        if numeric_prod != 1:
            non_numeric.append(numeric_prod)
        if not non_numeric:
            return sp.Integer(1)
        if len(non_numeric) == 1:
            return non_numeric[0]
        return Mul(*non_numeric, evaluate=False)
    # Some sympy types (e.g. ComplexRootOf) reject evaluate kwarg; fall back gracefully.
    try:
        return expr.func(*new_args, evaluate=False)
    except TypeError:
        try:
            return expr.func(*new_args)
        except TypeError:
            return expr


def normalize(expr: Expr) -> Expr:
    """Flatten nested Add/Mul and fold pure-numeric subtrees.

    This is the canonical-form pass applied by rule applies (to keep
    intermediate states clean) and by `canonical_repr` (so dedup catches
    `Add(Add(2x, 3), -7)` and `Add(2x, -4)` as equal).
    """
    return _fold_numeric_subtrees(_flatten_add_mul(expr))


def _ac_sorted_args(expr: Expr) -> tuple:
    """Return AC-canonical args for Add/Mul: sorted by recursive structural repr.

    For non-AC ops (Pow, Function, Symbol, Rational), preserve original arg order.
    """
    args = expr.args
    if isinstance(expr, (Add, Mul)) and len(args) > 1:
        return tuple(sorted(args, key=_structural_repr))
    return args


def _structural_repr(expr: Expr) -> str:
    """Recursive AC-sorted srepr without normalization. Internal helper."""
    if not getattr(expr, "args", None):
        return sp.srepr(expr)
    sorted_args = _ac_sorted_args(expr)
    return f"{type(expr).__name__}({', '.join(_structural_repr(a) for a in sorted_args)})"


def canonical_repr(expr: Expr) -> str:
    """Deterministic AC-canonical structural fingerprint for an expression.

    Two expressions that are structurally equivalent under (a) Add/Mul arg
    permutation, (b) nested Add/Mul flattening, and (c) pure-numeric folding
    produce the same string. Used as the dedup hash for BFS.
    """
    return _structural_repr(normalize(expr))


def tree_depth(expr: Expr) -> int:
    """Maximum AST depth: 1 for atoms, 1 + max(child depths) for internal nodes.

    Re-exports phase0/src/features.py:tree_depth for parity.
    """
    if not expr.args:
        return 1
    return 1 + max(tree_depth(a) for a in expr.args)


def op_count(expr: Expr) -> int:
    """Count of internal (non-atomic) AST nodes. Phase 0 parity."""
    if not expr.args:
        return 0
    return 1 + sum(op_count(a) for a in expr.args)


def leaf_count(expr: Expr) -> int:
    """Count of atomic nodes (Symbol, Number). Phase 0 parity."""
    if not expr.args:
        return 1
    return sum(leaf_count(a) for a in expr.args)


def iter_subtrees(expr: Expr) -> Iterator[tuple[tuple[int, ...], Expr]]:
    """Preorder traversal yielding (path, node) pairs.

    `path` is a tuple of `args` indices from the root. The empty tuple denotes
    the root itself. Used as structural parameterization for rule actions.
    """
    yield ((), expr)
    for i, child in enumerate(expr.args):
        for subpath, node in iter_subtrees(child):
            yield ((i,) + subpath, node)


def replace_at_path(expr: Expr, path: tuple[int, ...], replacement: Expr) -> Expr:
    """Return a new expression with the subtree at `path` replaced by `replacement`.

    Uses `func(*new_args, evaluate=False)` to avoid SymPy auto-canonicalization
    of the parent. Paths must be valid (raises IndexError otherwise).
    """
    if not path:
        return replacement
    head, *rest = path
    args = list(expr.args)
    args[head] = replace_at_path(args[head], tuple(rest), replacement)
    return expr.func(*args, evaluate=False)
