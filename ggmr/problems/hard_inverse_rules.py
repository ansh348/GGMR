"""Hard-problem inverse rules.

Four additional inverse-rule primitives that create structural entanglement
(nested fractions, hidden quadratics, Möbius-style rational wraps) for the
Phase 2 hard evaluation set.

Kept separate from `default_inverse_registry` in `inverse_rules.py` so that
Phase 1b reproducibility (the 7-rule registry it was certified against)
remains intact. Use `hard_inverse_registry` for hard-problem generation.
"""

from __future__ import annotations

import random
from typing import Iterator

import sympy as sp
from sympy import Add, Integer, Mul, Pow

from ..expr.tree import canonical_repr
from ..state import EqState
from .inverse_rules import (
    InverseAction,
    InverseRegistry,
    default_inverse_registry,
)


# ---------------------------------------------------------------------------
# Rational inverses
# ---------------------------------------------------------------------------


class InvEmbedInFraction:
    """Multiply lhs by (x+k)/(x+k): introduces a single-side fraction wrap.

    Forward undo: CANCEL_COMMON_FACTOR(factor=(x+k)). Alone this is a 1-step
    undo, but stacked with other hard inverses it forces multi-step paths
    through CLEAR_FRACTIONS / CROSS_MULTIPLY.

    Adds excluded value `-k` (root of x+k).
    """

    name = "INV_EMBED_IN_FRACTION"

    def enumerate(self, state: EqState, rng: random.Random) -> Iterator[InverseAction]:
        k = rng.choice([-3, -2, -1, 1, 2, 3])
        yield InverseAction(self.name, params=(Integer(k),))

    def apply(self, state: EqState, action: InverseAction) -> EqState:
        (k,) = action.params
        denom = Add(state.var, k, evaluate=False)
        numer = Mul(state.lhs, denom, evaluate=False)
        inv = Pow(denom, Integer(-1), evaluate=False)
        new_lhs = Mul(numer, inv, evaluate=False)
        excluded_val = Integer(-int(k))
        return state.with_lhs_rhs(new_lhs, state.rhs).with_excluded(excluded_val)


class InvSplitAcrossSides:
    """Split an Add term from lhs across the equation, with shared-denom dummy.

    Take state.lhs = T + R (Add with len(args) >= 2). Move T to rhs (as -T),
    then add the dummy `T/(x+k)` to BOTH sides (preserves equality, sound).
    The result has var on both sides through a shared (x+k) denominator —
    a state the structural heuristic strongly disprefers.

    Adds excluded value `-k`.
    """

    name = "INV_SPLIT_ACROSS_SIDES"

    def enumerate(self, state: EqState, rng: random.Random) -> Iterator[InverseAction]:
        if not isinstance(state.lhs, Add) or len(state.lhs.args) < 2:
            return
        k = rng.choice([1, 2, 3, -1, -2])
        yield InverseAction(self.name, params=(Integer(k),))

    def apply(self, state: EqState, action: InverseAction) -> EqState:
        (k,) = action.params
        terms = list(state.lhs.args)
        # deterministic choice: first term by srepr ordering (stable across runs)
        terms_sorted = sorted(terms, key=sp.srepr)
        T = terms_sorted[0]
        rest = [t for t in terms if t is not T] or [a for a in terms if sp.srepr(a) != sp.srepr(T)]
        # Defensive: if all terms had same srepr, fall back to using the first
        if not rest:
            rest = terms[1:]
        if len(rest) == 1:
            new_lhs_main = rest[0]
        else:
            new_lhs_main = Add(*rest, evaluate=False)
        neg_T = Mul(Integer(-1), T, evaluate=False)
        new_rhs_main = Add(state.rhs, neg_T, evaluate=False)
        denom = Add(state.var, k, evaluate=False)
        inv_d = Pow(denom, Integer(-1), evaluate=False)
        dummy = Mul(T, inv_d, evaluate=False)
        final_lhs = Add(new_lhs_main, dummy, evaluate=False)
        final_rhs = Add(new_rhs_main, dummy, evaluate=False)
        excluded_val = Integer(-int(k))
        return state.with_lhs_rhs(final_lhs, final_rhs).with_excluded(excluded_val)


# ---------------------------------------------------------------------------
# Polynomial / quadratic inverses
# ---------------------------------------------------------------------------


class InvDisguiseByExpansion:
    """Expand a Mul-of-Add lhs into polynomial form AND add `+kx -kx` dummy.

    Forward undo: COMBINE_LIKE_TERMS_AT collapses `+kx -kx` to zero, then
    IDENTITY_ADD_ZERO_AT removes it, then FACTOR_POLYNOMIAL re-factors.

    No excluded values (no division).
    """

    name = "INV_DISGUISE_BY_EXPANSION"

    def enumerate(self, state: EqState, rng: random.Random) -> Iterator[InverseAction]:
        if not isinstance(state.lhs, Mul):
            return
        if not any(isinstance(a, Add) for a in state.lhs.args):
            return
        k = rng.choice([1, 2, 3])
        yield InverseAction(self.name, params=(Integer(k),))

    def apply(self, state: EqState, action: InverseAction) -> EqState:
        (k,) = action.params
        expanded = sp.expand(state.lhs)
        plus_kx = Mul(k, state.var, evaluate=False)
        minus_kx = Mul(Integer(-int(k)), state.var, evaluate=False)
        new_lhs = Add(expanded, plus_kx, minus_kx, evaluate=False)
        return state.with_lhs_rhs(new_lhs, state.rhs)


class InvNestInRational:
    """Möbius-style rational wrap: replace `e = c` with `(e*q + p*x)/q = (c*q + p*x)/q`.

    Both sides wrapped in `(... + p*x) / (x+q)`. The forward undo is multi-
    step: CROSS_MULTIPLY -> DISTRIBUTE -> COMBINE_LIKE_TERMS -> ISOLATE.
    This rule explodes all four structural features simultaneously.

    Adds excluded value `-q`.
    """

    name = "INV_NEST_IN_RATIONAL"

    def enumerate(self, state: EqState, rng: random.Random) -> Iterator[InverseAction]:
        if canonical_repr(state.lhs) == canonical_repr(state.rhs):
            return
        p = rng.choice([1, 2, -1])
        q = rng.choice([1, 2, 3, -1])
        yield InverseAction(self.name, params=(Integer(p), Integer(q)))

    def apply(self, state: EqState, action: InverseAction) -> EqState:
        p, q = action.params
        q_sym = Add(state.var, q, evaluate=False)
        r_sym = Mul(p, state.var, evaluate=False)
        top_l = Add(Mul(state.lhs, q_sym, evaluate=False), r_sym, evaluate=False)
        top_r = Add(Mul(state.rhs, q_sym, evaluate=False), r_sym, evaluate=False)
        inv_q = Pow(q_sym, Integer(-1), evaluate=False)
        new_lhs = Mul(top_l, inv_q, evaluate=False)
        new_rhs = Mul(top_r, inv_q, evaluate=False)
        excluded_val = Integer(-int(q))
        return state.with_lhs_rhs(new_lhs, new_rhs).with_excluded(excluded_val)


# ---------------------------------------------------------------------------
# Registry: hard inverse rules (existing 7 + 4 new)
# ---------------------------------------------------------------------------


hard_inverse_registry = InverseRegistry()
for _r in default_inverse_registry.all_rules():
    hard_inverse_registry.register(_r)
hard_inverse_registry.register(InvEmbedInFraction())
hard_inverse_registry.register(InvSplitAcrossSides())
hard_inverse_registry.register(InvDisguiseByExpansion())
hard_inverse_registry.register(InvNestInRational())
