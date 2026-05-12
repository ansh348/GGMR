"""Arithmetic rules: add/multiply/divide both sides, negate, flip.

Per `ggmr_v10.pdf` §2.2, parameterizations are structurally motivated — we
enumerate over additive terms / multiplicative factors already present in
`lhs` or `rhs`, not arbitrary constants.
"""

from __future__ import annotations

from typing import Iterator

import sympy as sp
from sympy import Add, Expr, Mul, Pow

from ...expr.tree import canonical_repr
from ...state import EqState
from ..base import Action, GuardResult
from ..registry import default_registry


def _additive_terms(expr: Expr) -> tuple[Expr, ...]:
    if isinstance(expr, Add):
        return expr.args
    return (expr,)


def _multiplicative_factors(expr: Expr) -> tuple[Expr, ...]:
    if isinstance(expr, Mul):
        return expr.args
    return (expr,)


def _is_symbolically_zero(expr: Expr) -> bool:
    """True iff `expr` simplifies to 0. Catches structural-but-zero forms like
    `Add(7, -7, evaluate=False)` that pass `expr == sp.Integer(0)` literal check."""
    if expr == sp.Integer(0):
        return True
    try:
        return bool(sp.simplify(expr) == 0)
    except Exception:
        return False


def _denominators(expr: Expr) -> list[Expr]:
    """Return the symbolic denominators occurring as Pow(_, -1) factors anywhere
    in `expr`. Used to enumerate multiplication candidates that clear fractions."""
    out: list[Expr] = []

    def walk(e: Expr) -> None:
        if isinstance(e, Pow) and e.args[1] == sp.Integer(-1):
            out.append(e.args[0])
            return
        if isinstance(e, Pow) and isinstance(e.args[1], sp.Integer) and int(e.args[1]) < 0:
            out.append(e.args[0])
            return
        for a in e.args:
            walk(a)

    walk(expr)
    return out


# ---------------------------------------------------------------------------
# 1. ADD_TO_BOTH_SIDES(c) — add a structurally-motivated constant/term to both sides
# ---------------------------------------------------------------------------


class AddToBothSides:
    name = "ADD_TO_BOTH_SIDES"
    arity = 1

    def enumerate(self, state: EqState) -> Iterator[Action]:
        seen: set[str] = set()
        for side in (state.lhs, state.rhs):
            for term in _additive_terms(side):
                if term == sp.Integer(0):
                    continue
                c = sp.Mul(sp.Integer(-1), term, evaluate=False) if term.could_extract_minus_sign() is False else -term
                # Use simplest sign for c to make canonical
                c = -term
                key = canonical_repr(c)
                if key in seen:
                    continue
                seen.add(key)
                yield Action(self.name, params=(c,))

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        (c,) = action.params
        return state.with_lhs_rhs(
            Add(state.lhs, c, evaluate=False),
            Add(state.rhs, c, evaluate=False),
        )


default_registry.register(AddToBothSides())


# ---------------------------------------------------------------------------
# 2. MULTIPLY_BOTH_SIDES_BY(c) — c ≠ 0
# ---------------------------------------------------------------------------


class MultiplyBothSidesBy:
    name = "MULTIPLY_BOTH_SIDES_BY"
    arity = 1

    def enumerate(self, state: EqState) -> Iterator[Action]:
        seen: set[str] = set()
        # Candidates: denominators present in lhs/rhs (clear fractions selectively)
        # and constant multiplicative factors of lhs/rhs.
        candidates: list[Expr] = []
        for side in (state.lhs, state.rhs):
            candidates.extend(_denominators(side))
            for f in _multiplicative_factors(side):
                if f == sp.Integer(1) or f == sp.Integer(-1):
                    continue
                if state.var not in f.free_symbols:
                    candidates.append(f)
        for c in candidates:
            if c == sp.Integer(0):
                continue
            key = canonical_repr(c)
            if key in seen:
                continue
            seen.add(key)
            yield Action(self.name, params=(c,))

    def guard(self, state: EqState, action: Action) -> GuardResult:
        (c,) = action.params
        # Require c != 0 symbolically (not just literally).
        if _is_symbolically_zero(c):
            return GuardResult.failing("multiplier simplifies to zero")
        if state.var in c.free_symbols:
            roots_of_c = sp.solve(c, state.var)
            return GuardResult.passing(new_excluded=roots_of_c)
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        (c,) = action.params
        return state.with_lhs_rhs(
            Mul(state.lhs, c, evaluate=False),
            Mul(state.rhs, c, evaluate=False),
        )


default_registry.register(MultiplyBothSidesBy())


# ---------------------------------------------------------------------------
# 3. DIVIDE_BOTH_SIDES_BY(c) — c ≠ 0
# ---------------------------------------------------------------------------


class DivideBothSidesBy:
    name = "DIVIDE_BOTH_SIDES_BY"
    arity = 1

    def enumerate(self, state: EqState) -> Iterator[Action]:
        seen: set[str] = set()
        candidates: list[Expr] = []
        # Constant multiplicative factors of lhs/rhs (for cancellation of leading coefficient)
        for side in (state.lhs, state.rhs):
            for f in _multiplicative_factors(side):
                if f == sp.Integer(1) or f == sp.Integer(-1):
                    continue
                if state.var not in f.free_symbols:
                    candidates.append(f)
        for c in candidates:
            if c == sp.Integer(0):
                continue
            key = canonical_repr(c)
            if key in seen:
                continue
            seen.add(key)
            yield Action(self.name, params=(c,))

    def guard(self, state: EqState, action: Action) -> GuardResult:
        (c,) = action.params
        if _is_symbolically_zero(c):
            return GuardResult.failing("divisor simplifies to zero")
        if state.var in c.free_symbols:
            roots_of_c = sp.solve(c, state.var)
            return GuardResult.passing(new_excluded=roots_of_c)
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        (c,) = action.params
        inv = Pow(c, sp.Integer(-1), evaluate=False)
        return state.with_lhs_rhs(
            Mul(state.lhs, inv, evaluate=False),
            Mul(state.rhs, inv, evaluate=False),
        )


default_registry.register(DivideBothSidesBy())


# ---------------------------------------------------------------------------
# 4. NEGATE_BOTH_SIDES — multiply both sides by -1
# ---------------------------------------------------------------------------


class NegateBothSides:
    name = "NEGATE_BOTH_SIDES"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        yield Action(self.name)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        neg = sp.Integer(-1)
        return state.with_lhs_rhs(
            Mul(neg, state.lhs, evaluate=False),
            Mul(neg, state.rhs, evaluate=False),
        )


default_registry.register(NegateBothSides())


# ---------------------------------------------------------------------------
# 5. FLIP_SIDES — swap lhs and rhs
# ---------------------------------------------------------------------------


class FlipSides:
    name = "FLIP_SIDES"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        yield Action(self.name)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        return state.with_lhs_rhs(state.rhs, state.lhs)


default_registry.register(FlipSides())


# ===========================================================================
# Phase 1b additions (rules 16-20)
# ===========================================================================


# ---------------------------------------------------------------------------
# 16. MOVE_ALL_TO_LHS — subtract rhs from both sides → lhs - rhs = 0
# ---------------------------------------------------------------------------


class MoveAllToLhs:
    name = "MOVE_ALL_TO_LHS"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        if _is_symbolically_zero(state.rhs):
            return
        yield Action(self.name)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        neg_rhs = Mul(sp.Integer(-1), state.rhs, evaluate=False)
        new_lhs = Add(state.lhs, neg_rhs, evaluate=False)
        return state.with_lhs_rhs(new_lhs, sp.Integer(0))


default_registry.register(MoveAllToLhs())


# ---------------------------------------------------------------------------
# 17. MOVE_ALL_TO_RHS — subtract lhs from both sides → 0 = rhs - lhs
# ---------------------------------------------------------------------------


class MoveAllToRhs:
    name = "MOVE_ALL_TO_RHS"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        if _is_symbolically_zero(state.lhs):
            return
        yield Action(self.name)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        neg_lhs = Mul(sp.Integer(-1), state.lhs, evaluate=False)
        new_rhs = Add(state.rhs, neg_lhs, evaluate=False)
        return state.with_lhs_rhs(sp.Integer(0), new_rhs)


default_registry.register(MoveAllToRhs())


# ---------------------------------------------------------------------------
# 18. ISOLATE_VARIABLE — macro: a*var + b = c → var = (c - b) / a in one apply
#
# Compresses Phase 1a's `lin02` 338-node BFS expansion to ≤ 5 nodes. The macro
# is sound because verify_transition compares effective solution sets, not steps.
# Fires only when lhs is structurally `a*var + b` (or `var + b` or just `a*var`)
# and rhs is var-free; this leaves complex cases for finer-grained rules.
# ---------------------------------------------------------------------------


def _linear_in_var(expr: Expr, var: sp.Symbol) -> tuple[Expr, Expr] | None:
    """Return (a, b) if expr is a*var + b with constant a, b in var.
    Returns None if not linear in var.
    """
    try:
        poly = sp.Poly(sp.expand(expr), var)
    except sp.PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    a = poly.coeff_monomial(var)
    b = poly.coeff_monomial(1)
    return a, b


class IsolateVariable:
    name = "ISOLATE_VARIABLE"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        # Fire only if lhs is linear in var (a*var + b) and rhs is var-free.
        # The other-side (rhs linear, lhs var-free) is symmetric — we handle it
        # via FLIP_SIDES + ISOLATE_VARIABLE in two steps.
        if state.var in state.rhs.free_symbols:
            return
        if state.var not in state.lhs.free_symbols:
            return
        coeffs = _linear_in_var(state.lhs, state.var)
        if coeffs is None:
            return
        a, _ = coeffs
        if _is_symbolically_zero(a):
            return
        # Skip if already isolated (lhs == var)
        if state.lhs == state.var:
            return
        yield Action(self.name)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        coeffs = _linear_in_var(state.lhs, state.var)
        if coeffs is None:
            return GuardResult.failing("lhs is not linear in var")
        a, _ = coeffs
        if _is_symbolically_zero(a):
            return GuardResult.failing("leading coefficient simplifies to zero")
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        coeffs = _linear_in_var(state.lhs, state.var)
        if coeffs is None:
            return state
        a, b = coeffs
        # Build (rhs - b) / a structurally (with default-evaluate so the macro
        # presents the cleanly-isolated form, since the canonical_repr layer
        # handles AC normalization for dedup).
        new_rhs = (state.rhs - b) / a
        return state.with_lhs_rhs(state.var, new_rhs)


default_registry.register(IsolateVariable())


# ---------------------------------------------------------------------------
# 19. SQUARE_BOTH_SIDES — lhs² = rhs². Records sign relation as side condition.
#
# Squaring can introduce extraneous roots (any value where lhs = -rhs becomes
# a solution to lhs² = rhs²). Phase 1a's verifier compares effective sets and
# allows strict-subset child solution sets, so this rule is sound provided we
# add a side condition Eq(lhs - rhs, 0) capturing the original equality sign.
# In Phase 1b we record the side condition; downstream guards may use it.
# ---------------------------------------------------------------------------


class SquareBothSides:
    """Square both sides: lhs = rhs → lhs² = rhs².

    Squaring is sound only when the sign relation is tracked. To avoid
    introducing extraneous roots in arbitrary contexts, this rule fires only
    when one side contains a sqrt — i.e., the natural use is undoing a sqrt.
    Otherwise, the rule's apply would produce a child solution-set that is a
    SUPERSET of the parent's, violating the verifier's subset rule.
    """

    name = "SQUARE_BOTH_SIDES"
    arity = 0

    @staticmethod
    def _contains_sqrt(expr: Expr) -> bool:
        if expr.func == sp.sqrt or (isinstance(expr, Pow) and expr.args[1] == sp.Rational(1, 2)):
            return True
        return any(SquareBothSides._contains_sqrt(a) for a in expr.args)

    def enumerate(self, state: EqState) -> Iterator[Action]:
        # Only fire when one side has a sqrt (so squaring undoes it). On the
        # Phase 0 problem set there are no sqrts, so this rule never fires there.
        if not (self._contains_sqrt(state.lhs) or self._contains_sqrt(state.rhs)):
            return
        if canonical_repr(state.lhs) == canonical_repr(state.rhs):
            return
        yield Action(self.name)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        new_lhs = Pow(state.lhs, sp.Integer(2), evaluate=False)
        new_rhs = Pow(state.rhs, sp.Integer(2), evaluate=False)
        # Record the original sign relation as a side condition for downstream
        # domain-aware checks (Phase 1c verifier extension).
        sign_cond = sp.Eq(state.lhs - state.rhs, 0)
        return state.with_lhs_rhs(new_lhs, new_rhs).with_side_conditions(sign_cond)


default_registry.register(SquareBothSides())


# ---------------------------------------------------------------------------
# 20. RECIPROCATE_BOTH_SIDES — lhs = rhs → 1/lhs = 1/rhs (both sides non-zero)
# ---------------------------------------------------------------------------


class ReciprocateBothSides:
    name = "RECIPROCATE_BOTH_SIDES"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        # Skip if either side is literally zero (guard would reject).
        if _is_symbolically_zero(state.lhs) or _is_symbolically_zero(state.rhs):
            return
        yield Action(self.name)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        if _is_symbolically_zero(state.lhs):
            return GuardResult.failing("lhs simplifies to zero")
        if _is_symbolically_zero(state.rhs):
            return GuardResult.failing("rhs simplifies to zero")
        # Propagate var-roots of lhs/rhs as excluded.
        new_excluded: list[Expr] = []
        if state.var in state.lhs.free_symbols:
            new_excluded.extend(sp.solve(state.lhs, state.var))
        if state.var in state.rhs.free_symbols:
            new_excluded.extend(sp.solve(state.rhs, state.var))
        return GuardResult.passing(new_excluded=new_excluded)

    def apply(self, state: EqState, action: Action) -> EqState:
        inv_lhs = Pow(state.lhs, sp.Integer(-1), evaluate=False)
        inv_rhs = Pow(state.rhs, sp.Integer(-1), evaluate=False)
        return state.with_lhs_rhs(inv_lhs, inv_rhs)


default_registry.register(ReciprocateBothSides())
