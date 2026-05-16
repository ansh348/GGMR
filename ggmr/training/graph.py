"""SymPy expression tree -> PyG Data conversion.

Each (lhs, rhs) pair becomes a single graph with a virtual `Eq` super-root
connecting LHS and RHS subtrees. Node features are a fixed 30-dim vector;
edges are bidirectional (parent<->child) plus self-loops.

The same conversion is used by the JSONL dataset (for training) and by
LearnedHeuristic.evaluate (at A* runtime), so the feature definition must
be deterministic and import-stable.

Feature width was extended from 24 to 30 to support trig and calculus
domains (Phase 0.2). The new columns [24:30] mark domain-specific node
content: has_trig, has_exp, has_log, is_derivative, is_integral, is_limit.
Algebra-only states have zeros in those columns, so a 24-dim checkpoint
zero-padded to 30-dim is mathematically equivalent on algebra inputs
(see `scripts/migrate_24_to_30.py`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import sympy as sp
import torch
from sympy import Add, Expr, Integer, Mul, Pow, Rational, Symbol
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from torch_geometric.data import Data

from ggmr.expr.tree import leaf_count, op_count, tree_depth

from .srepr_parse import parse_srepr

NODE_TYPE_VOCAB: tuple[str, ...] = (
    "Add", "Mul", "Pow", "Symbol", "Integer", "Rational", "UNK",
)
NODE_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(NODE_TYPE_VOCAB)}
NUM_TYPES: int = len(NODE_TYPE_VOCAB)

# Feature layout (30 dims total):
#   [0:7]   op-type one-hot
#   [7:10]  symbol role: is_target_var, is_other_symbol, is_atom
#   [10:16] numeric: sign_pos, sign_neg, sign_zero, log1p(|v|)/10, is_integer, is_minus_one
#   [16:20] structural role: is_lhs_root, is_rhs_root, is_pow_base, is_pow_exponent
#   [20:24] subtree shape: depth/10, log1p(op_count), log1p(leaf_count), contains_target_var
#   [24:30] domain flags: has_trig, has_exp, has_log, is_derivative, is_integral, is_limit
FEATURE_DIM: int = 30
# Width of the legacy (pre-trig) feature vector — used by the migration script
# and by inference paths that need to detect old checkpoints.
LEGACY_FEATURE_DIM: int = 24


def _classify(expr: Optional[Expr]) -> str:
    """Map a SymPy expr to a NODE_TYPE_VOCAB tag. Order matters: Integer before Rational."""
    if expr is None:
        return "UNK"
    if isinstance(expr, Add):
        return "Add"
    if isinstance(expr, Mul):
        return "Mul"
    if isinstance(expr, Pow):
        return "Pow"
    if isinstance(expr, Symbol):
        return "Symbol"
    if isinstance(expr, Integer):
        return "Integer"
    if isinstance(expr, Rational):
        return "Rational"
    return "UNK"


def _symbol_role(expr: Optional[Expr], var: Symbol) -> tuple[float, float, float]:
    """(is_target_var, is_other_symbol, is_atom)."""
    if expr is None:
        return (0.0, 0.0, 0.0)
    if isinstance(expr, Symbol):
        if expr.name == var.name:
            return (1.0, 0.0, 1.0)
        return (0.0, 1.0, 1.0)
    if not getattr(expr, "args", ()):
        return (0.0, 0.0, 1.0)
    return (0.0, 0.0, 0.0)


def _numeric_features(expr: Optional[Expr]) -> tuple[float, float, float, float, float, float]:
    """(sign_pos, sign_neg, sign_zero, log1p(|v|)/10, is_integer, is_minus_one)."""
    if expr is None or not getattr(expr, "is_number", False):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    try:
        v = float(expr)
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if v > 0:
        sp_pos, sp_neg, sp_zero = 1.0, 0.0, 0.0
    elif v < 0:
        sp_pos, sp_neg, sp_zero = 0.0, 1.0, 0.0
    else:
        sp_pos, sp_neg, sp_zero = 0.0, 0.0, 1.0
    log_abs = float(np.clip(np.log1p(abs(v)), 0.0, 10.0)) / 10.0
    is_int = 1.0 if isinstance(expr, Integer) else 0.0
    is_m1 = 1.0 if (isinstance(expr, Integer) and int(expr) == -1) else 0.0
    return (sp_pos, sp_neg, sp_zero, log_abs, is_int, is_m1)


def _shape_features(expr: Optional[Expr], var: Symbol) -> tuple[float, float, float, float]:
    """(depth/10, log1p(op_count), log1p(leaf_count), contains_target_var)."""
    if expr is None:
        return (0.0, 0.0, 0.0, 0.0)
    d = min(10, tree_depth(expr)) / 10.0
    return (
        float(d),
        float(np.log1p(op_count(expr))),
        float(np.log1p(leaf_count(expr))),
        1.0 if expr.has(var) else 0.0,
    )


def _domain_flags(expr: Optional[Expr]) -> tuple[float, float, float, float, float, float]:
    """Subtree presence flags: (has_trig, has_exp, has_log, is_derivative, is_integral, is_limit).

    has_* checks whether the subtree contains the relevant function class anywhere;
    is_derivative/is_integral/is_limit checks whether THIS node is the operator
    (the unevaluated SymPy wrapper). For algebra-only states all six are zero.
    """
    if expr is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    has_trig = 1.0 if expr.has(TrigonometricFunction) else 0.0
    has_exp = 1.0 if expr.has(sp.exp) else 0.0
    has_log = 1.0 if expr.has(sp.log) else 0.0
    is_deriv = 1.0 if isinstance(expr, sp.Derivative) else 0.0
    is_intg = 1.0 if isinstance(expr, sp.Integral) else 0.0
    is_lim = 1.0 if isinstance(expr, sp.Limit) else 0.0
    return (has_trig, has_exp, has_log, is_deriv, is_intg, is_lim)


def _build_nodes(
    expr: Expr,
    parent_idx: int,
    parent_type: Optional[str],
    position: Optional[int],
    nodes: list[dict],
    edges: list[tuple[int, int]],
) -> int:
    """DFS-add expr and descendants to (nodes, edges). Return root index of this subtree."""
    idx = len(nodes)
    classification = _classify(expr)
    nodes.append({
        "type": classification,
        "expr": expr,
        "is_pow_exponent": (parent_type == "Pow" and position == 1),
        "is_pow_base": (parent_type == "Pow" and position == 0),
        "is_lhs_root": False,
        "is_rhs_root": False,
    })
    edges.append((parent_idx, idx))
    if hasattr(expr, "args"):
        for i, child in enumerate(expr.args):
            _build_nodes(child, idx, classification, i, nodes, edges)
    return idx


def _canonical(expr: Expr) -> Expr:
    """Apply srepr roundtrip so the graph structure is consistent regardless of
    whether the expression came from A* (direct SymPy) or from JSONL (parsed srepr).
    SymPy's Mul constructor splits Integer(-N) into Integer(-1)*Integer(N) inside
    parse_srepr; we apply that same canonicalization here so both code paths agree.
    """
    try:
        return parse_srepr(sp.srepr(expr))
    except Exception:
        return expr


def sympy_to_pyg(lhs: Expr, rhs: Expr, var: Symbol) -> Data:
    """Build a single PyG Data graph from (lhs, rhs) with a virtual Eq super-root.

    Edge convention: bidirectional (parent <-> child) plus self-loops on every node.
    Eq super-root is at index 0 with type UNK; LHS/RHS root nodes carry is_lhs/rhs_root flags.
    """
    lhs = _canonical(lhs)
    rhs = _canonical(rhs)
    nodes: list[dict] = [{
        "type": "UNK",
        "expr": None,
        "is_pow_exponent": False,
        "is_pow_base": False,
        "is_lhs_root": False,
        "is_rhs_root": False,
    }]
    edges: list[tuple[int, int]] = []

    lhs_root = _build_nodes(lhs, parent_idx=0, parent_type=None, position=None,
                            nodes=nodes, edges=edges)
    rhs_root = _build_nodes(rhs, parent_idx=0, parent_type=None, position=None,
                            nodes=nodes, edges=edges)
    nodes[lhs_root]["is_lhs_root"] = True
    nodes[rhs_root]["is_rhs_root"] = True

    n = len(nodes)
    x = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    for i, info in enumerate(nodes):
        type_idx = NODE_TYPE_TO_IDX.get(info["type"], NODE_TYPE_TO_IDX["UNK"])
        x[i, type_idx] = 1.0
        x[i, 7:10] = _symbol_role(info["expr"], var)
        x[i, 10:16] = _numeric_features(info["expr"])
        x[i, 16] = 1.0 if info["is_lhs_root"] else 0.0
        x[i, 17] = 1.0 if info["is_rhs_root"] else 0.0
        x[i, 18] = 1.0 if info["is_pow_base"] else 0.0
        x[i, 19] = 1.0 if info["is_pow_exponent"] else 0.0
        x[i, 20:24] = _shape_features(info["expr"], var)
        x[i, 24:30] = _domain_flags(info["expr"])

    src: list[int] = []
    dst: list[int] = []
    for s_, d_ in edges:
        src.append(s_); dst.append(d_)
        src.append(d_); dst.append(s_)
    for i in range(n):
        src.append(i); dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=torch.from_numpy(x), edge_index=edge_index)
