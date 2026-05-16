"""Trigonometric identity rules (Phase 1.1, Marcus).

41 primitive `training_safe=True` rules across 8 groups plus 2 oracle
shortcuts marked `training_safe=False` (excluded from BFS / SL / ExIt).

Mirrors the algebra-agent architecture: each rule is a class with
`name`, `arity`, `enumerate(state)`, `guard(state, action)`, `apply(state, action)`.
Self-registers with `default_registry` at import.

Each rule's `enumerate()` bails immediately when no trig atoms appear in
the state (the `_has_trig` short-circuit) — keeps trig rules from being
called 41 times on every algebra-only state expansion.

Groups
------
A. Pythagorean              (5)  Composite-Add matching: sin² + cos² → 1, etc.
B. Reciprocal               (6)  1/sin ↔ csc, 1/cos ↔ sec, 1/tan ↔ cot
C. Quotient                 (4)  sin/cos ↔ tan, cos/sin ↔ cot
D. Cofunction               (6)  sin(π/2 − u) → cos(u) and friends
E. Parity                   (4)  sin(−u) → −sin(u), cos(−u) → cos(u), …
F. Angle add/sub + double   (8)  sin(u+v), cos(u+v), tan(u±v), sin(2u), cos(2u)
G. Power reduction          (2)  sin²(u) → (1−cos(2u))/2, cos²(u) → (1+cos(2u))/2
H. Product/sum-to-product   (4)  sin(u)cos(v) → ½[sin(u+v)+sin(u−v)], etc.
I. Oracle (training_safe=F) (2)  TRIG_SIMPLIFY, TRIG_SOLVE — inference-only

Half-angle rules (sin(u/2), cos(u/2)) are deferred to v2 — they introduce a
sqrt(±) sign ambiguity that the current soundness machinery can't track.
"""

from __future__ import annotations

from typing import Iterator

import sympy as sp
from sympy import (
    Add,
    Expr,
    Integer,
    Mul,
    Pow,
    Rational,
    cos,
    cot,
    csc,
    pi,
    sec,
    sin,
    tan,
)
from sympy.functions.elementary.trigonometric import TrigonometricFunction

from ...expr.tree import canonical_repr
from ...state import EqState
from ..base import Action, GuardResult
from ..registry import default_registry
from .algebra import DistributeOverSubtree, _replace_in_state, _walk_with_side


# ============================================================================
# Helpers
# ============================================================================


def _has_trig_in_state(state: EqState) -> bool:
    """Fast-path check: skip trig rules entirely when neither side contains
    any TrigonometricFunction instance. Avoids 41 redundant guard calls per
    BFS expansion on algebra-only states.
    """
    return state.lhs.has(TrigonometricFunction) or state.rhs.has(TrigonometricFunction)


def _split_pow_squared(expr: Expr) -> Expr | None:
    """If `expr` is Pow(b, 2) with b a TrigonometricFunction, return b. Else None."""
    if isinstance(expr, Pow) and expr.args[1] == Integer(2):
        if isinstance(expr.args[0], TrigonometricFunction):
            return expr.args[0]
    return None


def _is_negative_arg(arg: Expr) -> Expr | None:
    """If `arg` is of the form `-u` (could_extract_minus_sign), return `u`.
    Else None. Pure structural — used by parity rules.
    """
    try:
        if arg.could_extract_minus_sign():
            return -arg
    except Exception:  # noqa: BLE001
        pass
    return None


def _complement_arg(arg: Expr) -> Expr | None:
    """If `arg` is structurally `pi/2 - u`, return `u`. Else None.

    Handles the most common shapes: Add(pi/2, -u), Add(-u, pi/2), Add(pi/2, Mul(-1, u)).
    """
    if not isinstance(arg, Add):
        return None
    half_pi = pi / 2
    others = []
    has_half_pi = False
    for term in arg.args:
        if term == half_pi:
            has_half_pi = True
            continue
        others.append(term)
    if not has_half_pi or not others:
        return None
    # Sum of `others` should be -u; we want u
    rest = others[0] if len(others) == 1 else Add(*others, evaluate=False)
    # Negate to recover u
    return Mul(Integer(-1), rest, evaluate=False)


def _is_double_arg(arg: Expr, var: sp.Symbol | None = None) -> Expr | None:
    """If `arg` is structurally `2*u`, return `u`. Else None."""
    if not isinstance(arg, Mul):
        return None
    if len(arg.args) < 2:
        return None
    if arg.args[0] != Integer(2):
        return None
    rest = arg.args[1:]
    if len(rest) == 1:
        return rest[0]
    return Mul(*rest, evaluate=False)


def _is_compound_arg(arg: Expr) -> bool:
    """True if `arg` is an Add of at least 2 terms (compound argument for SIN_SUM, etc.)."""
    return isinstance(arg, Add) and len(arg.args) >= 2


def _split_add_pair(arg: Add) -> tuple[Expr, Expr]:
    """Split an Add into two halves (first arg, remainder). Used by SIN_SUM/COS_SUM."""
    a = arg.args[0]
    rest = arg.args[1:]
    b = rest[0] if len(rest) == 1 else Add(*rest, evaluate=False)
    return a, b


def _trig_fn_at(expr: Expr, func) -> Expr | None:
    """If `expr` is `func(u)`, return u. Else None. `func` is sin/cos/tan/etc."""
    if isinstance(expr, func):
        return expr.args[0]
    return None


# ============================================================================
# Group A — Pythagorean identities
# ============================================================================


def _find_squared_trig_pair(args, func_a, func_b):
    """Find indices (i, j) in `args` such that args[i] = func_a(u)**2,
    args[j] = func_b(u)**2 for the SAME u. Returns (i, j, u) or None.
    """
    by_repr_a: dict[str, tuple[int, Expr]] = {}
    by_repr_b: dict[str, tuple[int, Expr]] = {}
    for idx, a in enumerate(args):
        base = _split_pow_squared(a)
        if base is None:
            continue
        if isinstance(base, func_a):
            by_repr_a[canonical_repr(base.args[0])] = (idx, base.args[0])
        if isinstance(base, func_b):
            by_repr_b[canonical_repr(base.args[0])] = (idx, base.args[0])
    for r, (i, u) in by_repr_a.items():
        if r in by_repr_b:
            return i, by_repr_b[r][0], u
    return None


def _find_squared_plus_one(args, func_sq):
    """Find indices (i, j) such that args[i] = func_sq(u)**2 and args[j] = Integer(1).
    Returns (i, j, u) or None.
    """
    sq_idx = None
    sq_u = None
    one_idx = None
    for idx, a in enumerate(args):
        base = _split_pow_squared(a)
        if base is not None and isinstance(base, func_sq) and sq_idx is None:
            sq_idx = idx
            sq_u = base.args[0]
        elif a == Integer(1) and one_idx is None:
            one_idx = idx
    if sq_idx is None or one_idx is None:
        return None
    return sq_idx, one_idx, sq_u


def _find_squared_minus_one(args, func_sq):
    """Find indices (i, j) such that args[i] = func_sq(u)**2 and args[j] = Integer(-1).
    Returns (i, j, u) or None.
    """
    sq_idx = None
    sq_u = None
    neg_one_idx = None
    for idx, a in enumerate(args):
        base = _split_pow_squared(a)
        if base is not None and isinstance(base, func_sq) and sq_idx is None:
            sq_idx = idx
            sq_u = base.args[0]
        elif a == Integer(-1) and neg_one_idx is None:
            neg_one_idx = idx
    if sq_idx is None or neg_one_idx is None:
        return None
    return sq_idx, neg_one_idx, sq_u


def _add_with_replacement(sub: Add, drop_idxs: set[int], add_term: Expr) -> Expr:
    """Build a new Add from `sub.args` excluding `drop_idxs` and appending `add_term`."""
    new_args = [a for k, a in enumerate(sub.args) if k not in drop_idxs]
    new_args.append(add_term)
    if len(new_args) == 1:
        return new_args[0]
    return Add(*new_args, evaluate=False)


class Sin2PlusCos2ToOne:
    name = "SIN2_PLUS_COS2_TO_ONE"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add):
                continue
            if _find_squared_trig_pair(sub.args, sin, cos) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, Add):
            return state
        match = _find_squared_trig_pair(sub.args, sin, cos)
        if match is None:
            return state
        i, j, _ = match
        new_sub = _add_with_replacement(sub, {i, j}, Integer(1))
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(Sin2PlusCos2ToOne())


class Tan2PlusOneToSec2:
    name = "TAN2_PLUS_ONE_TO_SEC2"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add):
                continue
            if _find_squared_plus_one(sub.args, tan) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, Add):
            return state
        match = _find_squared_plus_one(sub.args, tan)
        if match is None:
            return state
        i, j, u = match
        new_sub = _add_with_replacement(sub, {i, j}, Pow(sec(u), Integer(2), evaluate=False))
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(Tan2PlusOneToSec2())


class Cot2PlusOneToCsc2:
    name = "COT2_PLUS_ONE_TO_CSC2"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add):
                continue
            if _find_squared_plus_one(sub.args, cot) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, Add):
            return state
        match = _find_squared_plus_one(sub.args, cot)
        if match is None:
            return state
        i, j, u = match
        new_sub = _add_with_replacement(sub, {i, j}, Pow(csc(u), Integer(2), evaluate=False))
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(Cot2PlusOneToCsc2())


class Sec2MinusOneToTan2:
    name = "SEC2_MINUS_ONE_TO_TAN2"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add):
                continue
            if _find_squared_minus_one(sub.args, sec) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, Add):
            return state
        match = _find_squared_minus_one(sub.args, sec)
        if match is None:
            return state
        i, j, u = match
        new_sub = _add_with_replacement(sub, {i, j}, Pow(tan(u), Integer(2), evaluate=False))
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(Sec2MinusOneToTan2())


class OneToSin2PlusCos2:
    """Reverse Pythagorean: 1 → sin²(x) + cos²(x). Rarely useful at solve time
    but available for problem-generation completeness."""

    name = "ONE_TO_SIN2_PLUS_COS2"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        # Skip on algebra-only states: this rule injects trig into a non-trig
        # state, which is *almost always* a step backward in solving. Mark as
        # legal but rarely useful — the GIN learns to deprioritize via value.
        # We still gate on `_has_trig_in_state` to limit BFS branching factor.
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if sub == Integer(1):
                yield Action(self.name, params=(state.var,), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        u = action.params[0] if action.params else state.var
        new_sub = Add(Pow(sin(u), Integer(2), evaluate=False), Pow(cos(u), Integer(2), evaluate=False), evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(OneToSin2PlusCos2())


# ============================================================================
# Group B — Reciprocal identities
# ============================================================================


def _make_reciprocal_rule(rule_name, from_func, to_func, from_is_pow=True):
    """Build a class for: from_func(u) → 1/to_func(u) (or reverse, depending on from_is_pow).

    Generates the simple case: detect `from_func(u)` (as Pow with -1 exp or directly),
    rewrite using `to_func`. Boilerplate suffices for all 6 reciprocal rules.
    """

    class _Rule:
        name = rule_name
        arity = 1
        training_safe = True

        def enumerate(self, state):
            if not _has_trig_in_state(state):
                return
            for side, path, sub in _walk_with_side(state):
                u = self._detect(sub)
                if u is None:
                    continue
                yield Action(self.name, params=(), target_path=path, target_side=side)

        def guard(self, state, action):
            return GuardResult.passing()

        def apply(self, state, action):
            sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
            u = self._detect(sub)
            if u is None:
                return state
            new_sub = self._rewrite(u)
            return _replace_in_state(state, action.target_side, action.target_path, new_sub)

        @staticmethod
        def _detect(sub):
            if from_is_pow:
                # detect 1/from_func(u) i.e. Pow(from_func(u), -1)
                if isinstance(sub, Pow) and sub.args[1] == Integer(-1):
                    inner = sub.args[0]
                    if isinstance(inner, from_func):
                        return inner.args[0]
            else:
                # detect from_func(u) directly
                if isinstance(sub, from_func):
                    return sub.args[0]
            return None

        @staticmethod
        def _rewrite(u):
            if from_is_pow:
                # 1/from_func(u) → to_func(u)
                return to_func(u)
            else:
                # from_func(u) → 1/to_func(u) = Pow(to_func(u), -1)
                return Pow(to_func(u), Integer(-1), evaluate=False)

    _Rule.__name__ = f"Rule_{rule_name}"
    _Rule.__qualname__ = _Rule.__name__
    return _Rule


# B1-B3: 1/sin → csc, 1/cos → sec, 1/tan → cot
class ReciprocalCscToSin:
    name = "RECIPROCAL_1_OVER_SIN_TO_CSC"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, Pow) and sub.args[1] == Integer(-1) and isinstance(sub.args[0], sin):
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not (isinstance(sub, Pow) and sub.args[1] == Integer(-1) and isinstance(sub.args[0], sin)):
            return state
        u = sub.args[0].args[0]
        return _replace_in_state(state, action.target_side, action.target_path, csc(u))


default_registry.register(ReciprocalCscToSin())


class ReciprocalSecToCos:
    name = "RECIPROCAL_1_OVER_COS_TO_SEC"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, Pow) and sub.args[1] == Integer(-1) and isinstance(sub.args[0], cos):
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not (isinstance(sub, Pow) and sub.args[1] == Integer(-1) and isinstance(sub.args[0], cos)):
            return state
        u = sub.args[0].args[0]
        return _replace_in_state(state, action.target_side, action.target_path, sec(u))


default_registry.register(ReciprocalSecToCos())


class ReciprocalCotToTan:
    name = "RECIPROCAL_1_OVER_TAN_TO_COT"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, Pow) and sub.args[1] == Integer(-1) and isinstance(sub.args[0], tan):
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not (isinstance(sub, Pow) and sub.args[1] == Integer(-1) and isinstance(sub.args[0], tan)):
            return state
        u = sub.args[0].args[0]
        return _replace_in_state(state, action.target_side, action.target_path, cot(u))


default_registry.register(ReciprocalCotToTan())


# B4-B6: csc → 1/sin, sec → 1/cos, cot → 1/tan (reverse direction)
def _make_simple_trig_rewrite(name_str, from_func, to_expr_fn):
    """For rules like csc(u) → 1/sin(u): detect from_func(u), apply rewrite."""

    class _Rule:
        name = name_str
        arity = 1
        training_safe = True

        def enumerate(self, state):
            if not _has_trig_in_state(state):
                return
            for side, path, sub in _walk_with_side(state):
                if isinstance(sub, from_func):
                    yield Action(self.name, params=(), target_path=path, target_side=side)

        def guard(self, state, action):
            return GuardResult.passing()

        def apply(self, state, action):
            sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
            if not isinstance(sub, from_func):
                return state
            u = sub.args[0]
            return _replace_in_state(state, action.target_side, action.target_path, to_expr_fn(u))

    _Rule.__name__ = f"Rule_{name_str}"
    return _Rule


CscToReciprocalSin = _make_simple_trig_rewrite(
    "CSC_TO_1_OVER_SIN", csc, lambda u: Pow(sin(u), Integer(-1), evaluate=False)
)
SecToReciprocalCos = _make_simple_trig_rewrite(
    "SEC_TO_1_OVER_COS", sec, lambda u: Pow(cos(u), Integer(-1), evaluate=False)
)
CotToReciprocalTan = _make_simple_trig_rewrite(
    "COT_TO_1_OVER_TAN", cot, lambda u: Pow(tan(u), Integer(-1), evaluate=False)
)
default_registry.register(CscToReciprocalSin())
default_registry.register(SecToReciprocalCos())
default_registry.register(CotToReciprocalTan())


# ============================================================================
# Group C — Quotient identities
# ============================================================================


class TanToSinOverCos:
    """tan(u) → sin(u) * (1/cos(u))."""
    name = "TAN_TO_SIN_OVER_COS"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, tan):
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, tan):
            return GuardResult.failing("not tan")
        # cos(u) appearing in denominator means cos(u) != 0
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, tan):
            return state
        u = sub.args[0]
        new_sub = Mul(sin(u), Pow(cos(u), Integer(-1), evaluate=False), evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(TanToSinOverCos())


class CotToCosOverSin:
    name = "COT_TO_COS_OVER_SIN"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, cot):
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, cot):
            return state
        u = sub.args[0]
        new_sub = Mul(cos(u), Pow(sin(u), Integer(-1), evaluate=False), evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(CotToCosOverSin())


def _find_quotient_pair_in_mul(args, num_func, den_func):
    """Find indices (i, j) in `args` such that args[i] = num_func(u),
    args[j] = Pow(den_func(u), -1) for same u. Returns (i, j, u) or None.
    """
    nums: dict[str, tuple[int, Expr]] = {}
    dens: dict[str, tuple[int, Expr]] = {}
    for idx, a in enumerate(args):
        if isinstance(a, num_func):
            nums[canonical_repr(a.args[0])] = (idx, a.args[0])
        elif isinstance(a, Pow) and a.args[1] == Integer(-1) and isinstance(a.args[0], den_func):
            dens[canonical_repr(a.args[0].args[0])] = (idx, a.args[0].args[0])
    for r, (i, u) in nums.items():
        if r in dens:
            return i, dens[r][0], u
    return None


class SinOverCosToTan:
    name = "SIN_OVER_COS_TO_TAN"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            if _find_quotient_pair_in_mul(sub.args, sin, cos) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, Mul):
            return state
        match = _find_quotient_pair_in_mul(sub.args, sin, cos)
        if match is None:
            return state
        i, j, u = match
        new_args = [a for k, a in enumerate(sub.args) if k != i and k != j]
        new_args.append(tan(u))
        if len(new_args) == 1:
            new_sub = new_args[0]
        else:
            new_sub = Mul(*new_args, evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(SinOverCosToTan())


class CosOverSinToCot:
    name = "COS_OVER_SIN_TO_COT"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            if _find_quotient_pair_in_mul(sub.args, cos, sin) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, Mul):
            return state
        match = _find_quotient_pair_in_mul(sub.args, cos, sin)
        if match is None:
            return state
        i, j, u = match
        new_args = [a for k, a in enumerate(sub.args) if k != i and k != j]
        new_args.append(cot(u))
        if len(new_args) == 1:
            new_sub = new_args[0]
        else:
            new_sub = Mul(*new_args, evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(CosOverSinToCot())


# ============================================================================
# Group D — Cofunction identities: f(pi/2 - u) → cofunction(u)
# ============================================================================


def _make_complement_rule(name_str, from_func, to_func):
    class _Rule:
        name = name_str
        arity = 1
        training_safe = True

        def enumerate(self, state):
            if not _has_trig_in_state(state):
                return
            for side, path, sub in _walk_with_side(state):
                if not isinstance(sub, from_func):
                    continue
                if _complement_arg(sub.args[0]) is None:
                    continue
                yield Action(self.name, params=(), target_path=path, target_side=side)

        def guard(self, state, action):
            return GuardResult.passing()

        def apply(self, state, action):
            sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
            if not isinstance(sub, from_func):
                return state
            u = _complement_arg(sub.args[0])
            if u is None:
                return state
            return _replace_in_state(state, action.target_side, action.target_path, to_func(u))

    _Rule.__name__ = f"Rule_{name_str}"
    return _Rule


SinComplement = _make_complement_rule("SIN_COMPLEMENT", sin, cos)
CosComplement = _make_complement_rule("COS_COMPLEMENT", cos, sin)
TanComplement = _make_complement_rule("TAN_COMPLEMENT", tan, cot)
CotComplement = _make_complement_rule("COT_COMPLEMENT", cot, tan)
SecComplement = _make_complement_rule("SEC_COMPLEMENT", sec, csc)
CscComplement = _make_complement_rule("CSC_COMPLEMENT", csc, sec)
for cls in (SinComplement, CosComplement, TanComplement, CotComplement, SecComplement, CscComplement):
    default_registry.register(cls())


# ============================================================================
# Group E — Parity: sin(-u) → -sin(u), cos(-u) → cos(u), tan(-u) → -tan(u), cot(-u) → -cot(u)
# ============================================================================


class SinNeg:
    """sin(-u) → -sin(u)."""
    name = "SIN_NEG"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, sin) and _is_negative_arg(sub.args[0]) is not None:
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, sin):
            return state
        u = _is_negative_arg(sub.args[0])
        if u is None:
            return state
        new_sub = Mul(Integer(-1), sin(u), evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(SinNeg())


class CosNeg:
    """cos(-u) → cos(u). Even function — no sign change."""
    name = "COS_NEG"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, cos) and _is_negative_arg(sub.args[0]) is not None:
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, cos):
            return state
        u = _is_negative_arg(sub.args[0])
        if u is None:
            return state
        return _replace_in_state(state, action.target_side, action.target_path, cos(u))


default_registry.register(CosNeg())


class TanNeg:
    """tan(-u) → -tan(u)."""
    name = "TAN_NEG"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, tan) and _is_negative_arg(sub.args[0]) is not None:
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, tan):
            return state
        u = _is_negative_arg(sub.args[0])
        if u is None:
            return state
        new_sub = Mul(Integer(-1), tan(u), evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(TanNeg())


class CotNeg:
    """cot(-u) → -cot(u)."""
    name = "COT_NEG"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, cot) and _is_negative_arg(sub.args[0]) is not None:
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, cot):
            return state
        u = _is_negative_arg(sub.args[0])
        if u is None:
            return state
        new_sub = Mul(Integer(-1), cot(u), evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(CotNeg())


# ============================================================================
# Group F — Angle addition / subtraction / double
# ============================================================================


class SinSum:
    """sin(u + v) → sin(u)cos(v) + cos(u)sin(v)."""
    name = "SIN_SUM"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, sin) and _is_compound_arg(sub.args[0]):
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, sin):
            return state
        u, v = _split_add_pair(sub.args[0])
        new_sub = Add(
            Mul(sin(u), cos(v), evaluate=False),
            Mul(cos(u), sin(v), evaluate=False),
            evaluate=False,
        )
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(SinSum())


class CosSum:
    """cos(u + v) → cos(u)cos(v) - sin(u)sin(v)."""
    name = "COS_SUM"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, cos) and _is_compound_arg(sub.args[0]):
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, cos):
            return state
        u, v = _split_add_pair(sub.args[0])
        new_sub = Add(
            Mul(cos(u), cos(v), evaluate=False),
            Mul(Integer(-1), sin(u), sin(v), evaluate=False),
            evaluate=False,
        )
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(CosSum())


class TanSum:
    """tan(u + v) → (tan(u) + tan(v)) / (1 - tan(u)tan(v))."""
    name = "TAN_SUM"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, tan) and _is_compound_arg(sub.args[0]):
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, tan):
            return state
        u, v = _split_add_pair(sub.args[0])
        numerator = Add(tan(u), tan(v), evaluate=False)
        denominator = Add(
            Integer(1),
            Mul(Integer(-1), tan(u), tan(v), evaluate=False),
            evaluate=False,
        )
        new_sub = Mul(numerator, Pow(denominator, Integer(-1), evaluate=False), evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(TanSum())


# SIN_DIFF, COS_DIFF, TAN_DIFF: same as Sum but trigger only when the arg
# is structurally of the form `u + (-v)` — which is just any compound arg
# with a negative term. We treat them as a *separate* rule for the GIN to
# learn the sign distinction; they share the SinSum enumerator pattern but
# emit the difference form.
class SinDiff:
    """sin(u - v) → sin(u)cos(v) - cos(u)sin(v). Fires on compound args; the
    sign comes out naturally because v could be negative — we leave the apply
    identical to SinSum so the registered rule provides redundancy / alternative
    enumeration. This is intentional duplication for GIN robustness."""

    name = "SIN_DIFF"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        # Only fire when at least one term in the Add carries a minus sign.
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, sin):
                continue
            arg = sub.args[0]
            if not _is_compound_arg(arg):
                continue
            if not any(t.could_extract_minus_sign() for t in arg.args):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, sin):
            return state
        u, v = _split_add_pair(sub.args[0])
        new_sub = Add(
            Mul(sin(u), cos(v), evaluate=False),
            Mul(cos(u), sin(v), evaluate=False),
            evaluate=False,
        )
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(SinDiff())


class CosDiff:
    name = "COS_DIFF"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, cos):
                continue
            arg = sub.args[0]
            if not _is_compound_arg(arg):
                continue
            if not any(t.could_extract_minus_sign() for t in arg.args):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, cos):
            return state
        u, v = _split_add_pair(sub.args[0])
        new_sub = Add(
            Mul(cos(u), cos(v), evaluate=False),
            Mul(Integer(-1), sin(u), sin(v), evaluate=False),
            evaluate=False,
        )
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(CosDiff())


class TanDiff:
    name = "TAN_DIFF"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, tan):
                continue
            arg = sub.args[0]
            if not _is_compound_arg(arg):
                continue
            if not any(t.could_extract_minus_sign() for t in arg.args):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, tan):
            return state
        u, v = _split_add_pair(sub.args[0])
        numerator = Add(tan(u), tan(v), evaluate=False)
        denominator = Add(
            Integer(1),
            Mul(Integer(-1), tan(u), tan(v), evaluate=False),
            evaluate=False,
        )
        new_sub = Mul(numerator, Pow(denominator, Integer(-1), evaluate=False), evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(TanDiff())


class SinDouble:
    """sin(2u) → 2 sin(u) cos(u)."""
    name = "SIN_DOUBLE"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, sin) and _is_double_arg(sub.args[0]) is not None:
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, sin):
            return state
        u = _is_double_arg(sub.args[0])
        if u is None:
            return state
        new_sub = Mul(Integer(2), sin(u), cos(u), evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(SinDouble())


class CosDouble:
    """cos(2u) → cos²(u) - sin²(u). We pick one of three valid forms (Marcus's
    "double angle" group lists three variants — we ship the symmetric form,
    which can be combined with Pythagorean to yield the other two)."""

    name = "COS_DOUBLE"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if isinstance(sub, cos) and _is_double_arg(sub.args[0]) is not None:
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, cos):
            return state
        u = _is_double_arg(sub.args[0])
        if u is None:
            return state
        new_sub = Add(
            Pow(cos(u), Integer(2), evaluate=False),
            Mul(Integer(-1), Pow(sin(u), Integer(2), evaluate=False), evaluate=False),
            evaluate=False,
        )
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(CosDouble())


# ============================================================================
# Group G — Power reduction
# ============================================================================


class Sin2HalfAngle:
    """sin²(u) → (1 - cos(2u)) / 2."""
    name = "SIN_SQUARED_HALF_ANGLE"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            base = _split_pow_squared(sub)
            if base is not None and isinstance(base, sin):
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        base = _split_pow_squared(sub)
        if base is None or not isinstance(base, sin):
            return state
        u = base.args[0]
        numerator = Add(
            Integer(1),
            Mul(Integer(-1), cos(Mul(Integer(2), u, evaluate=False)), evaluate=False),
            evaluate=False,
        )
        new_sub = Mul(numerator, Pow(Integer(2), Integer(-1), evaluate=False), evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(Sin2HalfAngle())


class Cos2HalfAngle:
    """cos²(u) → (1 + cos(2u)) / 2."""
    name = "COS_SQUARED_HALF_ANGLE"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            base = _split_pow_squared(sub)
            if base is not None and isinstance(base, cos):
                yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        base = _split_pow_squared(sub)
        if base is None or not isinstance(base, cos):
            return state
        u = base.args[0]
        numerator = Add(
            Integer(1),
            cos(Mul(Integer(2), u, evaluate=False)),
            evaluate=False,
        )
        new_sub = Mul(numerator, Pow(Integer(2), Integer(-1), evaluate=False), evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(Cos2HalfAngle())


# ============================================================================
# Group H — Product/sum-to-product
# ============================================================================


def _find_prod_pair(args, func_a, func_b):
    """Find Mul-args indices (i, j) such that args[i] = func_a(u),
    args[j] = func_b(v), with u != v structurally. Returns (i, j, u, v) or None.
    """
    a_idxs: list[tuple[int, Expr]] = []
    b_idxs: list[tuple[int, Expr]] = []
    for idx, a in enumerate(args):
        if isinstance(a, func_a):
            a_idxs.append((idx, a.args[0]))
        if isinstance(a, func_b):
            b_idxs.append((idx, a.args[0]))
    for ia, ua in a_idxs:
        for ib, ub in b_idxs:
            if ia == ib:
                continue
            return ia, ib, ua, ub
    return None


class ProdSinCos:
    """sin(u) cos(v) → ½[sin(u+v) + sin(u-v)]."""
    name = "PROD_SIN_COS"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            if _find_prod_pair(sub.args, sin, cos) is None:
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, Mul):
            return state
        match = _find_prod_pair(sub.args, sin, cos)
        if match is None:
            return state
        ia, ib, u, v = match
        upv = Add(u, v, evaluate=False)
        umv = Add(u, Mul(Integer(-1), v, evaluate=False), evaluate=False)
        replacement = Mul(
            Pow(Integer(2), Integer(-1), evaluate=False),
            Add(sin(upv), sin(umv), evaluate=False),
            evaluate=False,
        )
        new_args = [a for k, a in enumerate(sub.args) if k != ia and k != ib]
        new_args.append(replacement)
        if len(new_args) == 1:
            new_sub = new_args[0]
        else:
            new_sub = Mul(*new_args, evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(ProdSinCos())


class ProdSinSin:
    """sin(u) sin(v) → ½[cos(u-v) - cos(u+v)]. Skips u == v (that's just sin²)."""
    name = "PROD_SIN_SIN"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            sins = [(i, a.args[0]) for i, a in enumerate(sub.args) if isinstance(a, sin)]
            if len(sins) < 2:
                continue
            (i, u), (j, v) = sins[0], sins[1]
            if canonical_repr(u) == canonical_repr(v):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, Mul):
            return state
        sins = [(i, a.args[0]) for i, a in enumerate(sub.args) if isinstance(a, sin)]
        if len(sins) < 2:
            return state
        (ia, u), (ib, v) = sins[0], sins[1]
        if canonical_repr(u) == canonical_repr(v):
            return state
        umv = Add(u, Mul(Integer(-1), v, evaluate=False), evaluate=False)
        upv = Add(u, v, evaluate=False)
        replacement = Mul(
            Pow(Integer(2), Integer(-1), evaluate=False),
            Add(cos(umv), Mul(Integer(-1), cos(upv), evaluate=False), evaluate=False),
            evaluate=False,
        )
        new_args = [a for k, a in enumerate(sub.args) if k != ia and k != ib]
        new_args.append(replacement)
        if len(new_args) == 1:
            new_sub = new_args[0]
        else:
            new_sub = Mul(*new_args, evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(ProdSinSin())


class ProdCosCos:
    """cos(u) cos(v) → ½[cos(u-v) + cos(u+v)]. Skips u == v."""
    name = "PROD_COS_COS"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Mul):
                continue
            coses = [(i, a.args[0]) for i, a in enumerate(sub.args) if isinstance(a, cos)]
            if len(coses) < 2:
                continue
            (i, u), (j, v) = coses[0], coses[1]
            if canonical_repr(u) == canonical_repr(v):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, Mul):
            return state
        coses = [(i, a.args[0]) for i, a in enumerate(sub.args) if isinstance(a, cos)]
        if len(coses) < 2:
            return state
        (ia, u), (ib, v) = coses[0], coses[1]
        if canonical_repr(u) == canonical_repr(v):
            return state
        umv = Add(u, Mul(Integer(-1), v, evaluate=False), evaluate=False)
        upv = Add(u, v, evaluate=False)
        replacement = Mul(
            Pow(Integer(2), Integer(-1), evaluate=False),
            Add(cos(umv), cos(upv), evaluate=False),
            evaluate=False,
        )
        new_args = [a for k, a in enumerate(sub.args) if k != ia and k != ib]
        new_args.append(replacement)
        if len(new_args) == 1:
            new_sub = new_args[0]
        else:
            new_sub = Mul(*new_args, evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(ProdCosCos())


class SumSinToProd:
    """sin(u) + sin(v) → 2 sin((u+v)/2) cos((u-v)/2). Skips u == v."""
    name = "SUM_SIN_TO_PROD"
    arity = 1
    training_safe = True

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        for side, path, sub in _walk_with_side(state):
            if not isinstance(sub, Add):
                continue
            sins = [(i, a.args[0]) for i, a in enumerate(sub.args) if isinstance(a, sin)]
            if len(sins) < 2:
                continue
            (i, u), (j, v) = sins[0], sins[1]
            if canonical_repr(u) == canonical_repr(v):
                continue
            yield Action(self.name, params=(), target_path=path, target_side=side)

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        sub = DistributeOverSubtree._fetch(state, action.target_side, action.target_path)
        if not isinstance(sub, Add):
            return state
        sins = [(i, a.args[0]) for i, a in enumerate(sub.args) if isinstance(a, sin)]
        if len(sins) < 2:
            return state
        (ia, u), (ib, v) = sins[0], sins[1]
        if canonical_repr(u) == canonical_repr(v):
            return state
        half = Pow(Integer(2), Integer(-1), evaluate=False)
        upv_half = Mul(Add(u, v, evaluate=False), half, evaluate=False)
        umv_half = Mul(Add(u, Mul(Integer(-1), v, evaluate=False), evaluate=False), half, evaluate=False)
        replacement = Mul(Integer(2), sin(upv_half), cos(umv_half), evaluate=False)
        new_args = [a for k, a in enumerate(sub.args) if k != ia and k != ib]
        new_args.append(replacement)
        if len(new_args) == 1:
            new_sub = new_args[0]
        else:
            new_sub = Add(*new_args, evaluate=False)
        return _replace_in_state(state, action.target_side, action.target_path, new_sub)


default_registry.register(SumSinToProd())


# ============================================================================
# Group I — Oracle shortcuts (training_safe = False)
# ============================================================================


class TrigSimplify:
    """Oracle: collapse via `sympy.trigsimp`. Single-shot. Excluded from training.

    Marcus Constraint 1: this rule must never enter BFS / SL / ExIt. The GIN
    must learn the underlying strategy via primitive rules, not learn to call
    the oracle.
    """

    name = "TRIG_SIMPLIFY"
    arity = 0
    training_safe = False

    def enumerate(self, state):
        # Cheap pre-check: only fire if at least one trig atom appears.
        if not _has_trig_in_state(state):
            return
        # Single canonical action — collapse both sides.
        yield Action(self.name, params=(), target_path=(), target_side="both")

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        try:
            new_lhs = sp.trigsimp(state.lhs)
            new_rhs = sp.trigsimp(state.rhs)
        except Exception:  # noqa: BLE001
            return state
        return state.with_lhs_rhs(new_lhs, new_rhs)


default_registry.register(TrigSimplify())


class TrigSolve:
    """Oracle: solve via `sympy.solveset` over the reals. Single-shot. Excluded from training."""

    name = "TRIG_SOLVE"
    arity = 0
    training_safe = False

    def enumerate(self, state):
        if not _has_trig_in_state(state):
            return
        yield Action(self.name, params=(), target_path=(), target_side="both")

    def guard(self, state, action):
        return GuardResult.passing()

    def apply(self, state, action):
        try:
            eq = sp.Eq(state.lhs, state.rhs)
            sols = sp.solveset(eq, state.var, domain=sp.S.Reals)
            # We need a concrete solution to put into canonical form. Solve()
            # returns more usable shapes; fall back to it.
            sols_list = sp.solve(eq, state.var)
        except Exception:  # noqa: BLE001
            return state
        if not sols_list:
            return state
        # Project to the canonical `x = solution` form using the first solution.
        new_lhs = state.var
        new_rhs = sols_list[0]
        return state.with_lhs_rhs(new_lhs, new_rhs)


default_registry.register(TrigSolve())
