"""Algebra rules: distribute, expand product, expand power, combine like terms.

Rules operate on a target subtree path. Enumeration scans `iter_subtrees` for
subtrees of the appropriate shape (Mul over Add for distribute/expand_product;
Pow with integer exponent for expand_power; Add for combine_like_terms).
"""

from __future__ import annotations

from itertools import product
from typing import Iterator

import sympy as sp
from sympy import Add, Expr, Integer, Mul, Pow

from ...expr.tree import canonical_repr, iter_subtrees, replace_at_path
from ...state import EqState
from ..base import Action, GuardResult
from ..registry import default_registry


def _walk_with_side(state: EqState) -> Iterator[tuple[str, tuple[int, ...], Expr]]:
    """Yield (side, path, subtree) for every subtree on lhs and rhs."""
    for path, sub in iter_subtrees(state.lhs):
        yield ("lhs", path, sub)
    for path, sub in iter_subtrees(state.rhs):
        yield ("rhs", path, sub)


def _replace_in_state(state: EqState, side: str, path: tuple[int, ...], rep: Expr) -> EqState:
    if side == "lhs":
        return state.with_lhs_rhs(replace_at_path(state.lhs, path, rep), state.rhs)
    return state.with_lhs_rhs(state.lhs, replace_at_path(state.rhs, path, rep))


# ---------------------------------------------------------------------------
# 6. DISTRIBUTE_OVER_SUBTREE(path) — a * (b + c) -> a*b + a*c, applied to one Mul-of-Add node
# ---------------------------------------------------------------------------


class DistributeOverSubtree:
    name = "DISTRIBUTE_OVER_SUBTREE"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            # find one Add factor among args
            add_idxs = [i for i, a in enumerate(sub.args) if isinstance(a, Add)]
            if not add_idxs:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = self._fetch(state, side, path)
        # find first Add factor; multiply the others through it
        add_idx = None
        for i, a in enumerate(sub.args):
            if isinstance(a, Add):
                add_idx = i
                break
        if add_idx is None:
            return state  # no-op (shouldn't happen if enumerate is correct)
        coeffs = [a for i, a in enumerate(sub.args) if i != add_idx]
        terms = sub.args[add_idx].args
        new_terms = []
        for t in terms:
            new_factors = list(coeffs) + [t]
            new_terms.append(Mul(*new_factors, evaluate=False))
        new_sub = Add(*new_terms, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)

    @staticmethod
    def _fetch(state: EqState, side: str, path: tuple[int, ...]) -> Expr:
        root = state.lhs if side == "lhs" else state.rhs
        for i in path:
            root = root.args[i]
        return root


default_registry.register(DistributeOverSubtree())


# ---------------------------------------------------------------------------
# 7. EXPAND_PRODUCT(path) — (a1+a2)(b1+b2) -> a1*b1 + a1*b2 + a2*b1 + a2*b2
#    Targets a Mul whose args contain ≥2 Add nodes (general distributive expansion).
# ---------------------------------------------------------------------------


class ExpandProduct:
    name = "EXPAND_PRODUCT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            n_add = sum(1 for a in sub.args if isinstance(a, Add))
            if n_add < 2:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Mul):
            return state
        # Cartesian product: each factor is either an Add (yielding its arg list)
        # or a non-Add (yielding [itself]).
        groups = []
        for f in sub.args:
            if isinstance(f, Add):
                groups.append(list(f.args))
            else:
                groups.append([f])
        new_terms = []
        for combo in product(*groups):
            new_terms.append(Mul(*combo, evaluate=False))
        new_sub = Add(*new_terms, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(ExpandProduct())


# ---------------------------------------------------------------------------
# 8. EXPAND_POWER(path) — (a + b)^n -> binomial expansion, n ∈ {2, 3}
# ---------------------------------------------------------------------------


class ExpandPower:
    name = "EXPAND_POWER"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Pow):
                continue
            base, exp = sub.args
            if not isinstance(exp, Integer):
                continue
            n = int(exp)
            if n not in (2, 3):
                continue
            if not isinstance(base, Add):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Pow):
            return state
        base, exp = sub.args
        n = int(exp)
        # Use SymPy's `expand` on the literal Pow rebuilt with evaluate=True
        rebuilt = sp.Pow(base, n)
        expanded = sp.expand(rebuilt)
        return _replace_in_state(state, side, path, expanded)


default_registry.register(ExpandPower())


# ---------------------------------------------------------------------------
# 9. COMBINE_LIKE_TERMS_AT(path) — folds an Add into its canonical sympy form
#    (this collapses literal-numeric arithmetic and merges like-coefficient terms).
# ---------------------------------------------------------------------------


class CombineLikeTermsAt:
    name = "COMBINE_LIKE_TERMS_AT"
    arity = 1  # path

    @staticmethod
    def _combine(sub: Expr) -> Expr:
        """Force SymPy's default Add canonicalization (which combines like terms)."""
        return Add(*sub.args)  # default evaluate=True

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add):
                continue
            combined = self._combine(sub)
            # Use sp.srepr (not canonical_repr) to detect structural change
            # without running normalize, which would mask the difference.
            if sp.srepr(combined) == sp.srepr(sub):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        return _replace_in_state(state, side, path, self._combine(sub))


default_registry.register(CombineLikeTermsAt())


# ===========================================================================
# Phase 1b additions (rules 21-27)
# ===========================================================================


# ---------------------------------------------------------------------------
# 21. FACTOR_OUT_GCF_AT(path) — `2x + 4` -> `2*(x+2)`; pulls integer GCF from Add args
# ---------------------------------------------------------------------------


class FactorOutGcfAt:
    name = "FACTOR_OUT_GCF_AT"
    arity = 1  # path

    @staticmethod
    def _integer_gcd(args: tuple[Expr, ...]) -> Integer | None:
        """Return integer GCF of leading numeric coefficients in args, or None
        if the gcd is 1 or all leading coefficients are 1.
        """
        coeffs: list[int] = []
        for a in args:
            if isinstance(a, Integer):
                coeffs.append(int(a))
                continue
            if isinstance(a, Mul):
                # Find first integer factor
                num = 1
                found = False
                for f in a.args:
                    if isinstance(f, Integer):
                        num = int(f)
                        found = True
                        break
                if not found:
                    return None  # Can't factor
                coeffs.append(num)
                continue
            return None  # Non-Mul, non-Integer atom — skip GCF factoring
        if not coeffs:
            return None
        from math import gcd
        from functools import reduce
        g = reduce(gcd, (abs(c) for c in coeffs))
        if g <= 1:
            return None
        return Integer(g)

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add) or len(sub.args) < 2:
                continue
            g = self._integer_gcd(sub.args)
            if g is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Add):
            return state
        g = self._integer_gcd(sub.args)
        if g is None:
            return state
        # Build factored form: g * (a1/g + a2/g + ...)
        new_terms: list[Expr] = []
        for a in sub.args:
            if isinstance(a, Integer):
                new_terms.append(Integer(int(a) // int(g)))
                continue
            if isinstance(a, Mul):
                new_factors = []
                divided = False
                for f in a.args:
                    if not divided and isinstance(f, Integer):
                        new_int = int(f) // int(g)
                        if new_int == 1 and len(a.args) > 1:
                            divided = True
                            continue  # drop *1
                        new_factors.append(Integer(new_int))
                        divided = True
                    else:
                        new_factors.append(f)
                if len(new_factors) == 1:
                    new_terms.append(new_factors[0])
                else:
                    new_terms.append(Mul(*new_factors, evaluate=False))
                continue
            new_terms.append(a)
        new_inner = Add(*new_terms, evaluate=False)
        new_sub = Mul(g, new_inner, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(FactorOutGcfAt())


# ---------------------------------------------------------------------------
# 22. COLLECT_LIKE_VARIABLE_TERMS_AT(path) — collect coefficients of var-power terms
# ---------------------------------------------------------------------------


class CollectLikeVariableTermsAt:
    name = "COLLECT_LIKE_VARIABLE_TERMS_AT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add):
                continue
            if state.var not in sub.free_symbols:
                continue
            try:
                collected = sp.collect(sub, state.var)
            except Exception:
                continue
            if sp.srepr(collected) == sp.srepr(sub):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        try:
            collected = sp.collect(sub, state.var)
        except Exception:
            return state
        return _replace_in_state(state, side, path, collected)


default_registry.register(CollectLikeVariableTermsAt())


# ---------------------------------------------------------------------------
# 23. DISTRIBUTE_NEGATIVE_AT(path) — -(a+b) -> -a - b. Targets Mul(-1, Add(...)).
# ---------------------------------------------------------------------------


class DistributeNegativeAt:
    name = "DISTRIBUTE_NEGATIVE_AT"
    arity = 1  # path

    @staticmethod
    def _is_negative_one_mul(sub: Expr) -> tuple[bool, Expr | None]:
        """Return (True, Add) if sub is Mul(-1, Add(...)) (in any arg order)."""
        if not isinstance(sub, Mul):
            return False, None
        has_neg = False
        the_add: Expr | None = None
        other_count = 0
        for a in sub.args:
            if a == Integer(-1):
                has_neg = True
                continue
            if isinstance(a, Add) and the_add is None:
                the_add = a
                continue
            other_count += 1
        # Must be exactly Mul(-1, Add) — no other multiplicands
        if has_neg and the_add is not None and other_count == 0:
            return True, the_add
        return False, None

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            ok, _ = self._is_negative_one_mul(sub)
            if not ok:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        ok, the_add = self._is_negative_one_mul(sub)
        if not ok or the_add is None:
            return state
        new_terms = [Mul(Integer(-1), t, evaluate=False) for t in the_add.args]
        new_sub = Add(*new_terms, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(DistributeNegativeAt())


# ---------------------------------------------------------------------------
# 24. IDENTITY_ADD_ZERO_AT(path) — drop +0 children of an Add
# ---------------------------------------------------------------------------


class IdentityAddZeroAt:
    name = "IDENTITY_ADD_ZERO_AT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add):
                continue
            if not any(a == Integer(0) for a in sub.args):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Add):
            return state
        new_args = [a for a in sub.args if a != Integer(0)]
        if not new_args:
            new_sub: Expr = Integer(0)
        elif len(new_args) == 1:
            new_sub = new_args[0]
        else:
            new_sub = Add(*new_args, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(IdentityAddZeroAt())


# ---------------------------------------------------------------------------
# 25. IDENTITY_MUL_ONE_AT(path) — drop *1 children of a Mul
# ---------------------------------------------------------------------------


class IdentityMulOneAt:
    name = "IDENTITY_MUL_ONE_AT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            if not any(a == Integer(1) for a in sub.args):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        if not isinstance(sub, Mul):
            return state
        new_args = [a for a in sub.args if a != Integer(1)]
        if not new_args:
            new_sub: Expr = Integer(1)
        elif len(new_args) == 1:
            new_sub = new_args[0]
        else:
            new_sub = Mul(*new_args, evaluate=False)
        return _replace_in_state(state, side, path, new_sub)


default_registry.register(IdentityMulOneAt())


# ---------------------------------------------------------------------------
# 26. ZERO_PROPERTY_AT(path) — Mul containing 0 collapses to 0
# ---------------------------------------------------------------------------


class ZeroPropertyAt:
    name = "ZERO_PROPERTY_AT"
    arity = 1  # path

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            if not any(a == Integer(0) for a in sub.args):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        return _replace_in_state(state, side, path, Integer(0))


default_registry.register(ZeroPropertyAt())


# ---------------------------------------------------------------------------
# 27. DOUBLE_NEGATION_AT(path) — Mul(-1, Mul(-1, x)) -> x; or Mul(-1, x) where x is
#     itself Mul(-1, ...). Targets the outer Mul.
# ---------------------------------------------------------------------------


class DoubleNegationAt:
    name = "DOUBLE_NEGATION_AT"
    arity = 1  # path

    @staticmethod
    def _is_neg(sub: Expr) -> bool:
        """True iff sub is Mul(-1, x) for some x (any arg order)."""
        if not isinstance(sub, Mul):
            return False
        if len(sub.args) != 2:
            return False
        return sub.args[0] == Integer(-1) or sub.args[1] == Integer(-1)

    @staticmethod
    def _strip_neg(sub: Expr) -> Expr | None:
        """Return x where sub == Mul(-1, x), else None."""
        if not isinstance(sub, Mul) or len(sub.args) != 2:
            return None
        if sub.args[0] == Integer(-1):
            return sub.args[1]
        if sub.args[1] == Integer(-1):
            return sub.args[0]
        return None

    def enumerate(self, state: EqState) -> Iterator[Action]:
        for side, path, sub in _walk_with_side(state):
            inner = self._strip_neg(sub)
            if inner is None:
                continue
            inner_inner = self._strip_neg(inner)
            if inner_inner is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state: EqState, action: Action) -> GuardResult:
        return GuardResult.passing()

    def apply(self, state: EqState, action: Action) -> EqState:
        side = action.target_side
        path = action.target_path
        sub = DistributeOverSubtree._fetch(state, side, path)
        inner = self._strip_neg(sub)
        if inner is None:
            return state
        inner_inner = self._strip_neg(inner)
        if inner_inner is None:
            return state
        return _replace_in_state(state, side, path, inner_inner)


default_registry.register(DoubleNegationAt())
