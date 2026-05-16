"""Inverse trigonometric rules — used by `TrigReverseGenerator` to manufacture
problems by reverse construction (Phase 1.2, Marcus).

Strategy: of the 39 forward training-safe trig rules, 9 EXPAND tree size in
the forward direction (sin(u+v) -> sin(u)cos(v)+cos(u)sin(v) adds nodes; the
power-reduction rules sin²(u) -> (1-cos(2u))/2 also add nodes). We wrap each
of these as an inverse-generation rule via `_ForwardAsInverse`.

For the Pythagorean expansion `1 -> sin²(u)+cos²(u)`, there is no expanding
forward rule (the forward is the simplifying SIN2_PLUS_COS2_TO_ONE). We
implement `InvPythagoreanIntroOne` explicitly.

Total: 12 inverse rules. Compares to algebra's 7 in `default_inverse_registry`.
"""

from __future__ import annotations

import random
from typing import Iterator

import sympy as sp
from sympy import Add, Integer, Pow

from ..rules.registry import default_registry
from ..state import EqState
from .inverse_rules import InverseAction, InverseRegistry


class _ForwardAsInverse:
    """Wrap an expanding forward rule as an inverse-generation rule.

    The forward rule's `enumerate(state)` already identifies positions where
    it can fire. For reverse-construction we want exactly that: find a spot
    where the forward expansion is legal, and apply it to grow tree size.

    The forward `Action` is carried verbatim in `InverseAction.params`.
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


class InvPythagoreanIntroOne:
    """Replace a `1` literal somewhere in lhs or rhs with `sin(u)**2 + cos(u)**2`.

    Inverse of SIN2_PLUS_COS2_TO_ONE: complexifies a state by expanding `1`.
    Uses `state.var` as the trig argument `u`.

    Operates via `Expr.replace(Integer(1), ...)` which replaces ALL occurrences
    of the literal `1` in the chosen side. For typical canonical seeds this is
    one position; if multiple it still produces a valid reverse step (forward
    can collapse each independently).
    """

    name = "INV_PYTHAGOREAN_INTRO_ONE"

    def enumerate(self, state: EqState, rng: random.Random) -> Iterator[InverseAction]:
        sides = []
        if sp.Integer(1) in state.lhs.atoms():
            sides.append("lhs")
        if sp.Integer(1) in state.rhs.atoms():
            sides.append("rhs")
        if not sides:
            return
        side = rng.choice(sides)
        yield InverseAction(self.name, params=(side,))

    def apply(self, state: EqState, action: InverseAction) -> EqState:
        (side,) = action.params
        var = state.var
        replacement = Add(
            Pow(sp.sin(var), Integer(2), evaluate=False),
            Pow(sp.cos(var), Integer(2), evaluate=False),
            evaluate=False,
        )
        if side == "lhs":
            new_lhs = state.lhs.replace(sp.Integer(1), replacement)
            return state.with_lhs_rhs(new_lhs, state.rhs)
        new_rhs = state.rhs.replace(sp.Integer(1), replacement)
        return state.with_lhs_rhs(state.lhs, new_rhs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Wrapped expanding forward rules. Each forward listed here is verified to
# strictly INCREASE tree size on every successful application.
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
trig_inverse_registry.register(InvPythagoreanIntroOne())
for _inv_name, _fwd_name in _WRAPPED_FORWARDS:
    trig_inverse_registry.register(_ForwardAsInverse(_inv_name, _fwd_name))
