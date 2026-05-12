"""Serialization primitives: prefix-notation tokens (for Phase 2 Transformer
input) and `lhs = rhs` string parsing (convenience for tests).
"""

from __future__ import annotations

import sympy as sp
from sympy import Expr, Symbol
from sympy.parsing.sympy_parser import parse_expr


def to_prefix_notation(expr: Expr) -> list[str]:
    """Serialize an expression to a flat list of prefix-notation tokens.

    Tokens:
        ADD, MUL, POW                          -- internal op
        FN_<name>                              -- generic function
        SYMBOL_<name>                          -- variable
        INT_<n>                                -- integer literal
        RATIONAL_<n>_<d>                       -- p/q rational (canonicalized)
        FLOAT_<repr>                           -- float literal (rare in algebra)
        ATOM_<srepr>                           -- fallback for unrecognized atoms

    The token stream is unambiguous because each internal op is followed
    immediately by its arity-many serialized children.
    """
    if isinstance(expr, sp.Add):
        out = ["ADD", f"ARITY_{len(expr.args)}"]
        for a in expr.args:
            out.extend(to_prefix_notation(a))
        return out
    if isinstance(expr, sp.Mul):
        out = ["MUL", f"ARITY_{len(expr.args)}"]
        for a in expr.args:
            out.extend(to_prefix_notation(a))
        return out
    if isinstance(expr, sp.Pow):
        out = ["POW"]
        for a in expr.args:
            out.extend(to_prefix_notation(a))
        return out
    if isinstance(expr, sp.Symbol):
        return [f"SYMBOL_{expr.name}"]
    if isinstance(expr, sp.Integer):
        return [f"INT_{int(expr)}"]
    if isinstance(expr, sp.Rational):
        return [f"RATIONAL_{expr.p}_{expr.q}"]
    if isinstance(expr, sp.Float):
        return [f"FLOAT_{float(expr)!r}"]
    if isinstance(expr, sp.Function):
        out = [f"FN_{type(expr).__name__}", f"ARITY_{len(expr.args)}"]
        for a in expr.args:
            out.extend(to_prefix_notation(a))
        return out
    return [f"ATOM_{sp.srepr(expr)}"]


def from_prefix_notation(tokens: list[str]) -> Expr:
    """Inverse of `to_prefix_notation`. Consumes the entire token list."""
    expr, idx = _parse_one(tokens, 0)
    if idx != len(tokens):
        raise ValueError(f"Trailing tokens after parse: {tokens[idx:]}")
    return expr


def _parse_one(tokens: list[str], idx: int) -> tuple[Expr, int]:
    if idx >= len(tokens):
        raise ValueError("Unexpected end of token stream")
    tok = tokens[idx]
    if tok == "ADD" or tok == "MUL":
        op_cls = sp.Add if tok == "ADD" else sp.Mul
        arity_tok = tokens[idx + 1]
        if not arity_tok.startswith("ARITY_"):
            raise ValueError(f"Expected ARITY_n after {tok}, got {arity_tok}")
        arity = int(arity_tok[len("ARITY_"):])
        args, j = [], idx + 2
        for _ in range(arity):
            child, j = _parse_one(tokens, j)
            args.append(child)
        return op_cls(*args, evaluate=False), j
    if tok == "POW":
        base, j = _parse_one(tokens, idx + 1)
        exp, j = _parse_one(tokens, j)
        return sp.Pow(base, exp, evaluate=False), j
    if tok.startswith("SYMBOL_"):
        return sp.Symbol(tok[len("SYMBOL_"):]), idx + 1
    if tok.startswith("INT_"):
        return sp.Integer(int(tok[len("INT_"):])), idx + 1
    if tok.startswith("RATIONAL_"):
        body = tok[len("RATIONAL_"):]
        p_str, q_str = body.split("_", 1)
        return sp.Rational(int(p_str), int(q_str)), idx + 1
    if tok.startswith("FLOAT_"):
        return sp.Float(float(tok[len("FLOAT_"):])), idx + 1
    raise ValueError(f"Unrecognized token: {tok}")


def parse_equation(s: str, var_name: str = "x") -> tuple[Expr, Expr]:
    """Parse an `lhs = rhs` string into two SymPy expressions, with evaluate=False.

    Convenience for tests; production code constructs equations via `EqState`.
    """
    if "=" not in s:
        raise ValueError(f"Expected '=' in {s!r}")
    lhs_str, rhs_str = s.split("=", 1)
    var = sp.Symbol(var_name)
    local: dict[str, Symbol] = {var_name: var}
    lhs = parse_expr(lhs_str.strip(), local_dict=local, evaluate=False)
    rhs = parse_expr(rhs_str.strip(), local_dict=local, evaluate=False)
    return lhs, rhs
