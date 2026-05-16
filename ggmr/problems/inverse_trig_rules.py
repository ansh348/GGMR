"""Inverse trigonometric rules — used by `TrigReverseGenerator` (Phase 1.2a, v2).

v2 (after external review): adds 12 new explicit inverse rules and fixes
`InvPythagoreanIntroOne` to randomize the trig argument it introduces.
Before: every Pythagorean expansion used `state.var` (always `x`), producing
repetitive `sin²(x) + cos²(x)` subtrees. After: the introduced argument is
sampled from existing trig arguments in the state or from `random_angle(rng)`.

Strategy:
- 11 wrappers around expanding forward trig rules (each yields actions for
  every applicable position in the state — `_ForwardAsInverse`).
- 13 explicit inverse rules (this file's classes below): replacing trig
  fundamental identities in either direction at structural positions.

Total registry: 24 inverse rules (was 12 in v1).
"""

from __future__ import annotations

import random
from typing import Iterator

import sympy as sp
from sympy import Add, Integer, Mul, Pow, Rational, cos, cot, csc, pi, sec, sin, tan

from ..rules.registry import default_registry
from ..state import EqState
from .inverse_rules import InverseAction, InverseRegistry
from .trig_templates import random_angle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def trig_args(expr) -> list:
    """All distinct trig-function arguments anywhere in `expr`.

    Used by inverse rules that need to introduce a NEW trig-bearing expression
    and want to reuse an existing argument shape (so the disguised problem
    references familiar angles instead of always introducing fresh symbols).
    """
    args: list = []
    seen: set = set()
    for fn_cls in (sin, cos, tan, sec, csc, cot):
        for node in expr.atoms(fn_cls):
            arg = node.args[0]
            key = sp.srepr(arg)
            if key not in seen:
                seen.add(key)
                args.append(arg)
    return args


def _pick_arg(state: EqState, rng: random.Random):
    """Pick a trig argument: prefer one already present in the state, else
    fall back to `random_angle(rng)`. Mix both sources for diversity."""
    candidates = trig_args(state.lhs) + trig_args(state.rhs)
    candidates.append(random_angle(rng))
    return rng.choice(candidates)


def _Pow2_noeval(e):
    return Pow(e, Integer(2), evaluate=False)


def _neg(e):
    return Mul(Integer(-1), e, evaluate=False)


# ---------------------------------------------------------------------------
# Wrapper for expanding forward rules
# ---------------------------------------------------------------------------


class _ForwardAsInverse:
    """Wrap an expanding forward trig rule as an inverse-generation rule.

    The forward rule's `enumerate(state)` identifies positions where it can
    fire. For reverse-construction we want exactly that. The forward `Action`
    is carried verbatim in `InverseAction.params`.
    """

    def __init__(self, name: str, forward_rule_name: str):
        self.name = name
        self._forward_rule_name = forward_rule_name

    def _forward(self):
        return default_registry.get(self._forward_rule_name)

    def enumerate(self, state: EqState, rng: random.Random) -> Iterator[InverseAction]:
        forward = self._forward()
        actions = list(forward.enumerate(state))
        rng.shuffle(actions)
        for fwd_action in actions:
            if forward.guard(state, fwd_action).ok:
                yield InverseAction(self.name, params=(fwd_action,))

    def apply(self, state: EqState, action: InverseAction) -> EqState:
        forward = self._forward()
        (fwd_action,) = action.params
        return forward.apply(state, fwd_action)


# ---------------------------------------------------------------------------
# Explicit inverse rule: introduce sin²(u)+cos²(u) in place of `1`
# ---------------------------------------------------------------------------


class InvPythagoreanIntroOne:
    """Replace a `1` literal in LHS or RHS with `sin²(u) + cos²(u)`, where
    `u` is sampled from the trig-arguments already present in the state OR
    from `random_angle(rng)`. (v2 fix: was always `state.var`.)

    Yields up to 4 (side, u) action variants so the applicable-action sampler
    in the generator has multiple distinct choices.
    """

    name = "INV_PYTHAGOREAN_INTRO_ONE"

    def enumerate(self, state, rng):
        sides = []
        if Integer(1) in state.lhs.atoms():
            sides.append("lhs")
        if Integer(1) in state.rhs.atoms():
            sides.append("rhs")
        if not sides:
            return
        candidates = trig_args(state.lhs) + trig_args(state.rhs)
        if not candidates:
            candidates = [random_angle(rng)]
        else:
            candidates.append(random_angle(rng))
        # Yield up to 4 distinct (side, u) combinations
        seen = set()
        emitted = 0
        for _ in range(12):
            if emitted >= 4:
                break
            side = rng.choice(sides)
            u = rng.choice(candidates)
            key = (side, sp.srepr(u))
            if key in seen:
                continue
            seen.add(key)
            emitted += 1
            yield InverseAction(self.name, params=(side, u))

    def apply(self, state, action):
        (side, u) = action.params
        replacement = Add(_Pow2_noeval(sin(u)), _Pow2_noeval(cos(u)), evaluate=False)
        if side == "lhs":
            new_lhs = state.lhs.replace(Integer(1), replacement)
            return state.with_lhs_rhs(new_lhs, state.rhs)
        new_rhs = state.rhs.replace(Integer(1), replacement)
        return state.with_lhs_rhs(state.lhs, new_rhs)


# ---------------------------------------------------------------------------
# Explicit inverse rules — introduce 1 in disguised forms
# ---------------------------------------------------------------------------


class InvOneToSecTan:
    """Replace `1` (any side) with `sec²(u) - tan²(u)` for a chosen u."""

    name = "INV_ONE_TO_SEC_TAN"

    def enumerate(self, state, rng):
        sides = []
        if Integer(1) in state.lhs.atoms():
            sides.append("lhs")
        if Integer(1) in state.rhs.atoms():
            sides.append("rhs")
        if not sides:
            return
        candidates = trig_args(state.lhs) + trig_args(state.rhs) + [random_angle(rng)]
        seen = set()
        emitted = 0
        for _ in range(8):
            if emitted >= 3:
                break
            side = rng.choice(sides)
            u = rng.choice(candidates)
            key = (side, sp.srepr(u))
            if key in seen:
                continue
            seen.add(key)
            emitted += 1
            yield InverseAction(self.name, params=(side, u))

    def apply(self, state, action):
        (side, u) = action.params
        replacement = Add(_Pow2_noeval(sec(u)), _neg(_Pow2_noeval(tan(u))), evaluate=False)
        if side == "lhs":
            return state.with_lhs_rhs(state.lhs.replace(Integer(1), replacement), state.rhs)
        return state.with_lhs_rhs(state.lhs, state.rhs.replace(Integer(1), replacement))


class InvOneToCscCot:
    """Replace `1` (any side) with `csc²(u) - cot²(u)`."""

    name = "INV_ONE_TO_CSC_COT"

    def enumerate(self, state, rng):
        sides = []
        if Integer(1) in state.lhs.atoms():
            sides.append("lhs")
        if Integer(1) in state.rhs.atoms():
            sides.append("rhs")
        if not sides:
            return
        candidates = trig_args(state.lhs) + trig_args(state.rhs) + [random_angle(rng)]
        seen = set()
        emitted = 0
        for _ in range(8):
            if emitted >= 3:
                break
            side = rng.choice(sides)
            u = rng.choice(candidates)
            key = (side, sp.srepr(u))
            if key in seen:
                continue
            seen.add(key)
            emitted += 1
            yield InverseAction(self.name, params=(side, u))

    def apply(self, state, action):
        (side, u) = action.params
        replacement = Add(_Pow2_noeval(csc(u)), _neg(_Pow2_noeval(cot(u))), evaluate=False)
        if side == "lhs":
            return state.with_lhs_rhs(state.lhs.replace(Integer(1), replacement), state.rhs)
        return state.with_lhs_rhs(state.lhs, state.rhs.replace(Integer(1), replacement))


# ---------------------------------------------------------------------------
# Explicit inverse rules — rewrite trig function calls in disguised forms
# ---------------------------------------------------------------------------


def _find_function_calls(expr, fn_cls):
    """Return list of (subnode) instances of `fn_cls` in `expr`."""
    return list(expr.atoms(fn_cls))


def _yield_target_actions(state, fn_cls, rng, name, max_emit=3):
    """Generic helper for inverse rules that find a target trig-fn call
    on either side and yield (side, target_node) actions."""
    options = []
    for side, expr in (("lhs", state.lhs), ("rhs", state.rhs)):
        for node in _find_function_calls(expr, fn_cls):
            options.append((side, node))
    if not options:
        return
    rng.shuffle(options)
    for side, node in options[:max_emit]:
        yield InverseAction(name, params=(side, node))


def _apply_replacement(state, side, target, replacement):
    if side == "lhs":
        new_lhs = state.lhs.xreplace({target: replacement})
        return state.with_lhs_rhs(new_lhs, state.rhs)
    new_rhs = state.rhs.xreplace({target: replacement})
    return state.with_lhs_rhs(state.lhs, new_rhs)


class InvTanToSinCos:
    """Replace tan(u) → sin(u)/cos(u)."""
    name = "INV_TAN_TO_SIN_COS"

    def enumerate(self, state, rng):
        yield from _yield_target_actions(state, tan, rng, self.name)

    def apply(self, state, action):
        side, target = action.params
        u = target.args[0]
        replacement = Mul(sin(u), Pow(cos(u), Integer(-1), evaluate=False), evaluate=False)
        return _apply_replacement(state, side, target, replacement)


class InvCotToCosSin:
    """Replace cot(u) → cos(u)/sin(u)."""
    name = "INV_COT_TO_COS_SIN"

    def enumerate(self, state, rng):
        yield from _yield_target_actions(state, cot, rng, self.name)

    def apply(self, state, action):
        side, target = action.params
        u = target.args[0]
        replacement = Mul(cos(u), Pow(sin(u), Integer(-1), evaluate=False), evaluate=False)
        return _apply_replacement(state, side, target, replacement)


class InvSecToInvCos:
    """Replace sec(u) → 1/cos(u)."""
    name = "INV_SEC_TO_INV_COS"

    def enumerate(self, state, rng):
        yield from _yield_target_actions(state, sec, rng, self.name)

    def apply(self, state, action):
        side, target = action.params
        u = target.args[0]
        replacement = Pow(cos(u), Integer(-1), evaluate=False)
        return _apply_replacement(state, side, target, replacement)


class InvCscToInvSin:
    """Replace csc(u) → 1/sin(u)."""
    name = "INV_CSC_TO_INV_SIN"

    def enumerate(self, state, rng):
        yield from _yield_target_actions(state, csc, rng, self.name)

    def apply(self, state, action):
        side, target = action.params
        u = target.args[0]
        replacement = Pow(sin(u), Integer(-1), evaluate=False)
        return _apply_replacement(state, side, target, replacement)


class InvSin2ToOneMinusCos2:
    """Replace sin²(u) → 1 - cos²(u). Finds Pow(sin(u), 2) subtrees."""
    name = "INV_SIN2_TO_ONE_MINUS_COS2"

    def enumerate(self, state, rng):
        options = []
        for side, expr in (("lhs", state.lhs), ("rhs", state.rhs)):
            for sub in sp.preorder_traversal(expr):
                if isinstance(sub, Pow) and sub.args[1] == Integer(2):
                    if isinstance(sub.args[0], sin):
                        options.append((side, sub))
        if not options:
            return
        rng.shuffle(options)
        for side, node in options[:3]:
            yield InverseAction(self.name, params=(side, node))

    def apply(self, state, action):
        side, target = action.params
        u = target.args[0].args[0]
        replacement = Add(Integer(1), _neg(_Pow2_noeval(cos(u))), evaluate=False)
        return _apply_replacement(state, side, target, replacement)


class InvCos2ToOneMinusSin2:
    """Replace cos²(u) → 1 - sin²(u)."""
    name = "INV_COS2_TO_ONE_MINUS_SIN2"

    def enumerate(self, state, rng):
        options = []
        for side, expr in (("lhs", state.lhs), ("rhs", state.rhs)):
            for sub in sp.preorder_traversal(expr):
                if isinstance(sub, Pow) and sub.args[1] == Integer(2):
                    if isinstance(sub.args[0], cos):
                        options.append((side, sub))
        if not options:
            return
        rng.shuffle(options)
        for side, node in options[:3]:
            yield InverseAction(self.name, params=(side, node))

    def apply(self, state, action):
        side, target = action.params
        u = target.args[0].args[0]
        replacement = Add(Integer(1), _neg(_Pow2_noeval(sin(u))), evaluate=False)
        return _apply_replacement(state, side, target, replacement)


class InvSinToParity:
    """Replace sin(u) → -sin(-u). Only fires when u is NOT already a
    `_neg`-shaped expression (avoid -sin(-(-u))=-sin(u) no-ops)."""
    name = "INV_SIN_TO_PARITY"

    def enumerate(self, state, rng):
        options = []
        for side, expr in (("lhs", state.lhs), ("rhs", state.rhs)):
            for node in expr.atoms(sin):
                u = node.args[0]
                # Skip if u already looks like -v (avoid undoing immediately)
                if u.could_extract_minus_sign():
                    continue
                options.append((side, node))
        if not options:
            return
        rng.shuffle(options)
        for side, node in options[:3]:
            yield InverseAction(self.name, params=(side, node))

    def apply(self, state, action):
        side, target = action.params
        u = target.args[0]
        replacement = _neg(sin(_neg(u), evaluate=False))
        return _apply_replacement(state, side, target, replacement)


class InvCosToParity:
    """Replace cos(u) → cos(-u). Only fires when u is NOT already a -v form."""
    name = "INV_COS_TO_PARITY"

    def enumerate(self, state, rng):
        options = []
        for side, expr in (("lhs", state.lhs), ("rhs", state.rhs)):
            for node in expr.atoms(cos):
                u = node.args[0]
                if u.could_extract_minus_sign():
                    continue
                options.append((side, node))
        if not options:
            return
        rng.shuffle(options)
        for side, node in options[:3]:
            yield InverseAction(self.name, params=(side, node))

    def apply(self, state, action):
        side, target = action.params
        u = target.args[0]
        replacement = cos(_neg(u), evaluate=False)
        return _apply_replacement(state, side, target, replacement)


class InvSinToCofunction:
    """Replace sin(u) → cos(pi/2 - u)."""
    name = "INV_SIN_TO_COFUNCTION"

    def enumerate(self, state, rng):
        options = []
        for side, expr in (("lhs", state.lhs), ("rhs", state.rhs)):
            for node in expr.atoms(sin):
                options.append((side, node))
        if not options:
            return
        rng.shuffle(options)
        for side, node in options[:3]:
            yield InverseAction(self.name, params=(side, node))

    def apply(self, state, action):
        side, target = action.params
        u = target.args[0]
        arg = Add(Mul(Rational(1, 2), pi, evaluate=False), _neg(u), evaluate=False)
        replacement = cos(arg, evaluate=False)
        return _apply_replacement(state, side, target, replacement)


class InvCosToCofunction:
    """Replace cos(u) → sin(pi/2 - u)."""
    name = "INV_COS_TO_COFUNCTION"

    def enumerate(self, state, rng):
        options = []
        for side, expr in (("lhs", state.lhs), ("rhs", state.rhs)):
            for node in expr.atoms(cos):
                options.append((side, node))
        if not options:
            return
        rng.shuffle(options)
        for side, node in options[:3]:
            yield InverseAction(self.name, params=(side, node))

    def apply(self, state, action):
        side, target = action.params
        u = target.args[0]
        arg = Add(Mul(Rational(1, 2), pi, evaluate=False), _neg(u), evaluate=False)
        replacement = sin(arg, evaluate=False)
        return _apply_replacement(state, side, target, replacement)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_WRAPPED_FORWARDS: tuple[tuple[str, str], ...] = (
    ("INV_SIN_SUM",                "SIN_SUM"),
    ("INV_COS_SUM",                "COS_SUM"),
    ("INV_TAN_SUM",                "TAN_SUM"),
    ("INV_SIN_DIFF",               "SIN_DIFF"),
    ("INV_COS_DIFF",               "COS_DIFF"),
    ("INV_SIN_DOUBLE",             "SIN_DOUBLE"),
    ("INV_COS_DOUBLE",             "COS_DOUBLE"),
    ("INV_SIN_SQUARED_HALF_ANGLE", "SIN_SQUARED_HALF_ANGLE"),
    ("INV_COS_SQUARED_HALF_ANGLE", "COS_SQUARED_HALF_ANGLE"),
    ("INV_PROD_SIN_COS",           "PROD_SIN_COS"),
    ("INV_SUM_SIN_TO_PROD",        "SUM_SIN_TO_PROD"),
)


trig_inverse_registry = InverseRegistry()

# Explicit inverse rules
trig_inverse_registry.register(InvPythagoreanIntroOne())
trig_inverse_registry.register(InvOneToSecTan())
trig_inverse_registry.register(InvOneToCscCot())
trig_inverse_registry.register(InvTanToSinCos())
trig_inverse_registry.register(InvCotToCosSin())
trig_inverse_registry.register(InvSecToInvCos())
trig_inverse_registry.register(InvCscToInvSin())
trig_inverse_registry.register(InvSin2ToOneMinusCos2())
trig_inverse_registry.register(InvCos2ToOneMinusSin2())
trig_inverse_registry.register(InvSinToParity())
trig_inverse_registry.register(InvCosToParity())
trig_inverse_registry.register(InvSinToCofunction())
trig_inverse_registry.register(InvCosToCofunction())

# Wrapped expanding forward rules
for _inv_name, _fwd_name in _WRAPPED_FORWARDS:
    trig_inverse_registry.register(_ForwardAsInverse(_inv_name, _fwd_name))
