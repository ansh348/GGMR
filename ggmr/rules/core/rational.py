"""Rational rules: cancel common factor, clear fractions by LCD.

Both rules require care: cancellation can legitimately remove an extraneous
root iff the factor's zero is recorded in `excluded`. Clearing fractions
multiplies through by an LCD, which propagates the LCD's zeros as `excluded`.
"""

from __future__ import annotations

from typing import Iterator

import sympy as sp
from sympy import Add, Expr, Integer, Mul, Pow, Symbol

from ...expr.tree import canonical_repr, iter_subtrees
from ...soundness import safe_solve
from ...state import EqState
from ..base import Action, GuardResult
from ..registry import default_registry
from .algebra import DistributeOverSubtree, _replace_in_state, _walk_with_side


def _flatten_mul_args(expr: Expr) -> list[Expr]:
    """Flatten nested Mul into a single arg list (recursively)."""
    if isinstance(expr, Mul):
        out: list[Expr] = []
        for a in expr.args:
            out.extend(_flatten_mul_args(a))
        return out
    return [expr]


def _denominator_factors(expr: Expr) -> list[Expr]:
    """Pow(_, -k) factors. Flattens nested Muls so `Mul(Mul(A, B), C^-1)` finds C."""
    out: list[Expr] = []
    flat = _flatten_mul_args(expr) if isinstance(expr, Mul) else [expr]
    for f in flat:
        if isinstance(f, Pow) and isinstance(f.args[1], Integer) and int(f.args[1]) < 0:
            out.append(f.args[0])
    return out


def _numerator_factors(expr: Expr) -> list[Expr]:
    """Positive-power factors. Flattens nested Muls."""
    out: list[Expr] = []
    if isinstance(expr, Mul):
        flat = _flatten_mul_args(expr)
        for f in flat:
            if isinstance(f, Pow) and isinstance(f.args[1], Integer) and int(f.args[1]) < 0:
                continue
            out.append(f)
    else:
        out.append(expr)
    return out


def _all_denominators_in(expr: Expr) -> list[Expr]:
    """Recursively gather all symbolic denominators (as bases of negative-exponent Pows)."""
    out: list[Expr] = []
    for _, sub in iter_subtrees(expr):
        if isinstance(sub, Pow) and isinstance(sub.args[1], Integer) and int(sub.args[1]) < 0:
            out.append(sub.args[0])
    return out


# ---------------------------------------------------------------------------
# 10. CANCEL_COMMON_FACTOR(path) — cancel a factor `f` appearing in numerator
#     and denominator of a Mul subtree.
# ---------------------------------------------------------------------------


class CancelCommonFactor:
    name = "CANCEL_COMMON_FACTOR"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        seen: set[tuple[str, tuple[int, ...], str]] = set()
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            num = _numerator_factors(sub)
            den = _denominator_factors(sub)
            if not num or not den:
                continue
            num_keys = {canonical_repr(n): n for n in num}
            for d in den:
                d_key = canonical_repr(d)
                if d_key in num_keys:
                    sig = (side, path, d_key)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    yield Action(self.name, params=(d,), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        (factor,) = action.params
        # Cancellation removes the factor from both num and den. The values where
        # `factor == 0` would otherwise be excluded from the domain — we add them
        # to `excluded` so the next-state's solution set is `parent ∩ {factor != 0}`.
        if state.var in factor.free_symbols:
            roots = safe_solve(factor, state.var)
            return GuardResult.passing(new_excluded=roots)
        if factor == sp.Integer(0):
            return GuardResult.failing("cannot cancel a constantly-zero factor")
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        (factor,) = action.params
        side = action.target_side
        path = action.target_path
        # Rebuild the targeted Mul without one numerator copy and one denominator copy of `factor`.
        from .algebra import DistributeOverSubtree

        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Mul):
            return state
        f_key = canonical_repr(factor)
        new_args = list(sub.args)
        # remove first numerator occurrence
        for i, a in enumerate(new_args):
            if (
                not (isinstance(a, Pow) and isinstance(a.args[1], Integer) and int(a.args[1]) < 0)
                and canonical_repr(a) == f_key
            ):
                new_args.pop(i)
                break
        # remove first denominator occurrence
        for i, a in enumerate(new_args):
            if isinstance(a, Pow) and isinstance(a.args[1], Integer) and int(a.args[1]) < 0:
                if canonical_repr(a.args[0]) == f_key:
                    if int(a.args[1]) == -1:
                        new_args.pop(i)
                    else:
                        new_args[i] = Pow(a.args[0], Integer(int(a.args[1]) + 1), evaluate=False)
                    break
        if not new_args:
            new_sub: Expr = Integer(1)
        elif len(new_args) == 1:
            new_sub = new_args[0]
        else:
            new_sub = Mul(*new_args, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(CancelCommonFactor())


# ---------------------------------------------------------------------------
# 11. CLEAR_FRACTIONS_BY_LCD — multiply both sides by lcm of all denominators.
# ---------------------------------------------------------------------------


class ClearFractionsByLCD:
    name = "CLEAR_FRACTIONS_BY_LCD"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        denoms = list({canonical_repr(d): d for d in _all_denominators_in(state.lhs) + _all_denominators_in(state.rhs)}.values())
        if not denoms:
            return
        # Compute LCM symbolically. Use sympy.lcm pairwise.
        lcd: Expr = denoms[0]
        for d in denoms[1:]:
            try:
                lcd = sp.lcm(lcd, d)
            except Exception:
                lcd = Mul(lcd, d, evaluate=False)
        if lcd == sp.Integer(1):
            return
        yield Action(self.name, params=(lcd,))

    def guard(self, state: EqState, action: Action) -> GuardResult:
        (lcd,) = action.params
        try:
            if bool(sp.simplify(lcd) == 0):
                return GuardResult.failing("LCD simplifies to zero")
        except Exception:
            pass
        if state.var in lcd.free_symbols:
            roots = safe_solve(lcd, state.var)
            return GuardResult.passing(new_excluded=roots)
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        (lcd,) = action.params
        # Multiply, then immediately simplify cancellations within each side.
        new_lhs = sp.together(state.lhs * lcd)
        new_rhs = sp.together(state.rhs * lcd)
        new_lhs = sp.cancel(new_lhs)
        new_rhs = sp.cancel(new_rhs)
        return state.with_lhs_rhs(new_lhs, new_rhs)


default_registry.register(ClearFractionsByLCD())


# ===========================================================================
# Phase 1b additions (rules 28-33)
# ===========================================================================


def _split_num_den(expr: Expr) -> tuple[Expr, Expr]:
    """Return (numerator, denominator) for an Expr. Uses sp.fraction.

    Returns (expr, 1) if expr has no denominator structure.
    """
    n, d = sp.fraction(expr)
    return n, d


# ---------------------------------------------------------------------------
# 28. CROSS_MULTIPLY — a/b = c/d → a*d = b*c. Both sides must have nontrivial denominators.
# ---------------------------------------------------------------------------


class CrossMultiply:
    name = "CROSS_MULTIPLY"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        l_num, l_den = _split_num_den(state.lhs)
        r_num, r_den = _split_num_den(state.rhs)
        if l_den == Integer(1) and r_den == Integer(1):
            return
        yield Action(self.name)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        _, l_den = _split_num_den(state.lhs)
        _, r_den = _split_num_den(state.rhs)
        new_excluded: list[Expr] = []
        if state.var in l_den.free_symbols:
            new_excluded.extend(safe_solve(l_den, state.var))
        if state.var in r_den.free_symbols:
            new_excluded.extend(safe_solve(r_den, state.var))
        # Check denominators non-zero literally
        if l_den == Integer(0) or r_den == Integer(0):
            return GuardResult.failing("denominator is zero")
        return GuardResult.passing(new_excluded=new_excluded)

    def apply(self, state: EqState, action: Action) -> EqState:
        l_num, l_den = _split_num_den(state.lhs)
        r_num, r_den = _split_num_den(state.rhs)
        new_lhs = Mul(l_num, r_den, evaluate=False)
        new_rhs = Mul(r_num, l_den, evaluate=False)
        return state.with_lhs_rhs(new_lhs, new_rhs)


default_registry.register(CrossMultiply())


# ---------------------------------------------------------------------------
# 29. COMBINE_FRACTIONS_AT(path) — a/b + c/d → (a*d + b*c)/(b*d). Targets an Add
#     whose terms include at least one fraction.
# ---------------------------------------------------------------------------


def _has_denominator(expr: Expr) -> bool:
    _, d = _split_num_den(expr)
    return d != Integer(1)


class CombineFractionsAt:
    name = "COMBINE_FRACTIONS_AT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add):
                continue
            # At least one term must be a fraction
            if not any(_has_denominator(t) for t in sub.args):
                continue
            # Combined form must be structurally different
            try:
                combined = sp.together(sub)
            except Exception:
                continue
            if sp.srepr(combined) == sp.srepr(sub):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Add):
            return GuardResult.failing("subtree is not an Add")
        new_excluded: list[Expr] = []
        for t in sub.args:
            _, d = _split_num_den(t)
            if d == Integer(1):
                continue
            if d == Integer(0):
                return GuardResult.failing("denominator is zero")
            if state.var in d.free_symbols:
                new_excluded.extend(safe_solve(d, state.var))
        return GuardResult.passing(new_excluded=new_excluded)

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        try:
            combined = sp.together(sub)
        except Exception:
            return state
        return _replace_in_state(state, side, path, combined)


default_registry.register(CombineFractionsAt())


# ---------------------------------------------------------------------------
# 30. SPLIT_FRACTION_AT(path) — (a+b)/c → a/c + b/c. Targets a Mul of Add * Pow(_, -1).
# ---------------------------------------------------------------------------


class SplitFractionAt:
    name = "SPLIT_FRACTION_AT"
    arity = 1  # path

    @staticmethod
    def _classify(sub: Expr) -> tuple[Expr, Expr] | None:
        """If sub is (Add) * Pow(c, -1), return (Add, c). Else None."""
        if not isinstance(sub, Mul):
            return None
        the_add: Expr | None = None
        the_inv_base: Expr | None = None
        for f in sub.args:
            if isinstance(f, Add) and the_add is None:
                the_add = f
                continue
            if (
                isinstance(f, Pow)
                and isinstance(f.args[1], Integer)
                and int(f.args[1]) == -1
                and the_inv_base is None
            ):
                the_inv_base = f.args[0]
                continue
            return None  # Other factors not allowed for this simple split
        if the_add is None or the_inv_base is None:
            return None
        return the_add, the_inv_base

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if self._classify(sub) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        cls = self._classify(sub)
        if cls is None:
            return GuardResult.failing("subtree is not (Add)/c")
        _, c = cls
        if c == Integer(0):
            return GuardResult.failing("denominator is zero")
        new_excluded: list[Expr] = []
        if state.var in c.free_symbols:
            new_excluded.extend(safe_solve(c, state.var))
        return GuardResult.passing(new_excluded=new_excluded)

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        cls = self._classify(sub)
        if cls is None:
            return state
        the_add, c = cls
        inv_c = Pow(c, Integer(-1), evaluate=False)
        new_terms = [Mul(t, inv_c, evaluate=False) for t in the_add.args]
        new_sub = Add(*new_terms, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(SplitFractionAt())


# ---------------------------------------------------------------------------
# 31. COMMON_DENOMINATOR_AT(path) — for Add of two fractions a/b + c/d, rewrite
#     each term over the common denominator b*d without combining numerators yet.
#     Result: (a*d)/(b*d) + (c*b)/(b*d). Useful for proof granularity.
# ---------------------------------------------------------------------------


class CommonDenominatorAt:
    name = "COMMON_DENOMINATOR_AT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add) or len(sub.args) < 2:
                continue
            denoms = [_split_num_den(t)[1] for t in sub.args]
            distinct = {canonical_repr(d): d for d in denoms if d != Integer(1)}
            if len(distinct) < 2:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Add):
            return GuardResult.failing("subtree is not an Add")
        new_excluded: list[Expr] = []
        for t in sub.args:
            _, d = _split_num_den(t)
            if d == Integer(0):
                return GuardResult.failing("denominator is zero")
            if state.var in d.free_symbols:
                new_excluded.extend(safe_solve(d, state.var))
        return GuardResult.passing(new_excluded=new_excluded)

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Add):
            return state
        # Common denominator = product of distinct denominators (not LCM, simpler)
        denoms = [_split_num_den(t)[1] for t in sub.args]
        distinct = list({canonical_repr(d): d for d in denoms if d != Integer(1)}.values())
        if len(distinct) < 2:
            return state
        common = distinct[0]
        for d in distinct[1:]:
            common = Mul(common, d, evaluate=False)
        new_terms: list[Expr] = []
        for t in sub.args:
            n, d = _split_num_den(t)
            if d == Integer(1):
                # Multiply n by common to bring over the shared denominator
                new_n = Mul(n, common, evaluate=False)
                new_terms.append(Mul(new_n, Pow(common, Integer(-1), evaluate=False), evaluate=False))
            else:
                # Multiply n by (common/d) and put over common
                missing_factors = [
                    df for df in distinct if canonical_repr(df) != canonical_repr(d)
                ]
                if missing_factors:
                    extra = missing_factors[0]
                    for mf in missing_factors[1:]:
                        extra = Mul(extra, mf, evaluate=False)
                    new_n = Mul(n, extra, evaluate=False)
                else:
                    new_n = n
                new_terms.append(Mul(new_n, Pow(common, Integer(-1), evaluate=False), evaluate=False))
        new_sub = Add(*new_terms, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(CommonDenominatorAt())


# ---------------------------------------------------------------------------
# 32. SIMPLIFY_AT(path) — apply sp.simplify on a subtree. Replaces the
#     Phase 1b draft INVERT_FRACTION_AT, which was fundamentally unsound
#     when applied to a single side of an equation (inverting (a/b) to (b/a)
#     in just one side does not preserve the equation's solution set).
#
#     SIMPLIFY_AT is sound because sp.simplify is value-preserving on the
#     subtree, so substituting the simplified form for the original at the
#     same path leaves the surrounding expression's value unchanged.
# ---------------------------------------------------------------------------


class SimplifyAt:
    name = "SIMPLIFY_AT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not sub.args:
                continue
            try:
                simplified = sp.simplify(sub)
            except Exception:
                continue
            if sp.srepr(simplified) == sp.srepr(sub):
                continue
            if canonical_repr(simplified) == canonical_repr(sub):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        try:
            simplified = sp.simplify(sub)
        except Exception:
            return state
        return _replace_in_state(state, side, path, simplified)


default_registry.register(SimplifyAt())


# ---------------------------------------------------------------------------
# 33. PARTIAL_FRACTIONS — apply sp.apart to lhs (or rhs) when it is a rational
#     function with a factorable denominator.
# ---------------------------------------------------------------------------


class PartialFractions:
    name = "PARTIAL_FRACTIONS"
    arity = 0

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side in ("lhs", "rhs"):
            expr = state.lhs if side == "lhs" else state.rhs
            if state.var not in expr.free_symbols:
                continue
            _, d = _split_num_den(expr)
            if d == Integer(1):
                continue
            try:
                aparted = sp.apart(expr, state.var)
            except Exception:
                continue
            if sp.srepr(aparted) == sp.srepr(expr):
                continue
            yield Action(self.name, params=(), target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        side = action.target_side
        expr = state.lhs if side == "lhs" else state.rhs
        _, d = _split_num_den(expr)
        new_excluded: list[Expr] = []
        if state.var in d.free_symbols:
            new_excluded.extend(safe_solve(d, state.var))
        return GuardResult.passing(new_excluded=new_excluded)

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        expr = state.lhs if side == "lhs" else state.rhs
        try:
            aparted = sp.apart(expr, state.var)
        except Exception:
            return state
        if side == "lhs":
            return state.with_lhs_rhs(aparted, state.rhs)
        return state.with_lhs_rhs(state.lhs, aparted)


default_registry.register(PartialFractions())
