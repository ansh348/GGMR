"""Round-trip sympy.srepr() output back to a SymPy Expr.

sympy.srepr emits class-call syntax like `Add(Mul(Integer(2), Symbol('x')), Integer(3))`.
parse_srepr eval's this against the sympy namespace with builtins disabled so the
only callables available are SymPy classes.
"""

from __future__ import annotations

import sympy as sp

_SYMPY_NS: dict = {**sp.__dict__, "__builtins__": {}}


def parse_srepr(s: str) -> sp.Basic:
    """Parse a sympy.srepr() string back to the SymPy Basic it came from.

    Wraps eval in `sympy.evaluate(False)` so SymPy does not auto-canonicalize
    (re-sort Mul/Add args, fold numerics, simplify Pow) during reconstruction.
    Without this, srepr round-trip diverges on ~half the data.

    Raises ValueError on parse failure or if the result is not a SymPy Basic.
    """
    try:
        with sp.evaluate(False):
            result = eval(s, _SYMPY_NS)  # noqa: S307 -- builtins disabled, sympy-only namespace
    except Exception as e:
        raise ValueError(f"parse_srepr failed on {s!r}: {type(e).__name__}: {e}") from e
    if not isinstance(result, sp.Basic):
        raise ValueError(
            f"parse_srepr({s!r}) returned {type(result).__name__}, expected sympy.Basic"
        )
    return result
