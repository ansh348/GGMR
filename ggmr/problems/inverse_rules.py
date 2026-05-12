"""Inverse rules: starting from a canonical-target state, apply inverse
operations to manufacture problems of controlled depth.

Per `ggmr/PHASE1B_PREREG.md` §3.4 (problem generator) and the inverse map in
the Phase 1b plan, each forward rule has an inverse step that, when applied to
a canonical target, produces a state from which forward BFS can solve back.

The InverseAction is intentionally separate from the forward Action (different
semantics: inverse application is for problem GENERATION, not solving).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

import sympy as sp
from sympy import Add, Expr, Integer, Mul, Pow, Symbol

from ..state import EqState


@dataclass(frozen=True)
class InverseAction:
    """An inverse-rule application: parameters that, applied via the corresponding
    InverseRule.apply, transform a canonical target state into a "harder" state.
    """

    inverse_name: str
    params: tuple = ()


@runtime_checkable
class InverseRule(Protocol):
    """Inverse-rule contract. `enumerate` yields candidate inverse actions
    (with random small parameters). `apply` is the structural transformation."""

    name: str

    def enumerate(self, state: EqState, rng: random.Random) -> Iterator[InverseAction]: ...
    def apply(self, state: EqState, action: InverseAction) -> EqState: ...


# ---------------------------------------------------------------------------
# Arithmetic inverses
# ---------------------------------------------------------------------------


class InvAddToBothSides:
    """Inverse of ADD_TO_BOTH_SIDES: add a non-zero constant c to both sides.

    Forward solve: ADD_TO_BOTH_SIDES(-c) undoes this.
    """

    name = "INV_ADD_TO_BOTH_SIDES"

    def enumerate(self, state, rng):
        c = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5])
        yield InverseAction(self.name, params=(Integer(c),))

    def apply(self, state, action):
        (c,) = action.params
        return state.with_lhs_rhs(
            Add(state.lhs, c, evaluate=False),
            Add(state.rhs, c, evaluate=False),
        )


class InvMultiplyBothSides:
    """Inverse of DIVIDE_BOTH_SIDES_BY(c): multiply both sides by non-zero c."""

    name = "INV_MULTIPLY_BOTH_SIDES"

    def enumerate(self, state, rng):
        c = rng.choice([2, 3, 4, 5, -2, -3])
        yield InverseAction(self.name, params=(Integer(c),))

    def apply(self, state, action):
        (c,) = action.params
        return state.with_lhs_rhs(
            Mul(c, state.lhs, evaluate=False),
            Mul(c, state.rhs, evaluate=False),
        )


class InvFlipSides:
    """Inverse of FLIP_SIDES: swap lhs and rhs (self-inverse)."""

    name = "INV_FLIP_SIDES"

    def enumerate(self, state, rng):
        yield InverseAction(self.name)

    def apply(self, state, action):
        return state.with_lhs_rhs(state.rhs, state.lhs)


# ---------------------------------------------------------------------------
# Algebra inverses
# ---------------------------------------------------------------------------


class InvDistributeOverSum:
    """Inverse of FACTOR_OUT_GCF_AT / DISTRIBUTE_OVER_SUBTREE: multiply lhs by
    a binomial expansion form. We expand a "factor" that's already implicit:
    take the lhs, multiply it by 1 = (a+b) * (1/(a+b)), simplifying gives
    nothing useful — instead, we reverse via "distribute" on the lhs.

    Concrete inverse: when lhs is a single term, multiply by (1+x)/(1+x). This
    introduces structural complexity that forward CANCEL_COMMON_FACTOR / SIMPLIFY
    can undo.

    For simplicity in Phase 1b, we use a different approach: introduce an
    expanded-product structure. Pick a small integer k and replace lhs `e` with
    `e*1` rewritten as `e * (k+1) - e*k`.

    This is sound (value-preserving) and adds 2 levels of structure.
    """

    name = "INV_DISTRIBUTE_OVER_SUM"

    def enumerate(self, state, rng):
        k = rng.choice([1, 2, 3])
        yield InverseAction(self.name, params=(Integer(k),))

    def apply(self, state, action):
        (k,) = action.params
        # lhs = e → lhs = e*(k+1) + e*(-k)
        new_lhs = Add(
            Mul(state.lhs, Integer(int(k) + 1), evaluate=False),
            Mul(state.lhs, Integer(-int(k)), evaluate=False),
            evaluate=False,
        )
        return state.with_lhs_rhs(new_lhs, state.rhs)


class InvCombineLikeTerms:
    """Inverse of COMBINE_LIKE_TERMS_AT: introduce a `+x - x` dummy term.

    Concrete: rewrite lhs as lhs + (k*var - k*var) for a small k.
    """

    name = "INV_COMBINE_LIKE_TERMS"

    def enumerate(self, state, rng):
        k = rng.choice([1, 2, 3])
        yield InverseAction(self.name, params=(Integer(k),))

    def apply(self, state, action):
        (k,) = action.params
        plus_term = Mul(k, state.var, evaluate=False)
        minus_term = Mul(Integer(-int(k)), state.var, evaluate=False)
        new_lhs = Add(state.lhs, plus_term, minus_term, evaluate=False)
        return state.with_lhs_rhs(new_lhs, state.rhs)


# ---------------------------------------------------------------------------
# Rational inverses
# ---------------------------------------------------------------------------


class InvClearFractions:
    """Inverse of CLEAR_FRACTIONS_BY_LCD: divide both sides by a small LCD.

    Concrete: divide both sides by (var + k) for small k, i.e., introduce a
    fraction that forward CLEAR_FRACTIONS will undo.
    """

    name = "INV_CLEAR_FRACTIONS"

    def enumerate(self, state, rng):
        k = rng.choice([1, 2, 3, -1, -2])
        yield InverseAction(self.name, params=(Integer(k),))

    def apply(self, state, action):
        (k,) = action.params
        denom = Add(state.var, k, evaluate=False)
        inv_denom = Pow(denom, Integer(-1), evaluate=False)
        new_lhs = Mul(state.lhs, inv_denom, evaluate=False)
        new_rhs = Mul(state.rhs, inv_denom, evaluate=False)
        # Add the corresponding excluded value
        excluded_val = Integer(-int(k))
        return state.with_lhs_rhs(new_lhs, new_rhs).with_excluded(excluded_val)


# ---------------------------------------------------------------------------
# Polynomial / quadratic inverses
# ---------------------------------------------------------------------------


class InvExpandProduct:
    """Inverse of FACTOR_POLYNOMIAL: expand the lhs.

    Concrete: when lhs is a Mul of factors, replace it with the expanded form.
    For canonical-target seeds like `(x-2)*(x-3) = 0`, this makes lhs the
    expanded polynomial `x² - 5x + 6`.
    """

    name = "INV_EXPAND_PRODUCT"

    def enumerate(self, state, rng):
        if not isinstance(state.lhs, Mul):
            return
        if not any(isinstance(a, Add) for a in state.lhs.args):
            return
        yield InverseAction(self.name)

    def apply(self, state, action):
        new_lhs = sp.expand(state.lhs)
        return state.with_lhs_rhs(new_lhs, state.rhs)


# ---------------------------------------------------------------------------
# Registry: inverse rules available for generation
# ---------------------------------------------------------------------------


@dataclass
class InverseRegistry:
    rules: list = field(default_factory=list)

    def register(self, rule):
        self.rules.append(rule)
        return rule

    def all_rules(self) -> list:
        return list(self.rules)


default_inverse_registry = InverseRegistry()
default_inverse_registry.register(InvAddToBothSides())
default_inverse_registry.register(InvMultiplyBothSides())
default_inverse_registry.register(InvFlipSides())
default_inverse_registry.register(InvDistributeOverSum())
default_inverse_registry.register(InvCombineLikeTerms())
default_inverse_registry.register(InvClearFractions())
default_inverse_registry.register(InvExpandProduct())
