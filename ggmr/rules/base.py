"""Rule architecture: Action, GuardResult, Rule Protocol, ApplyResult.

Designed for ~57 rules per `ggmr_v10.pdf` §3.2 (~25 guarded rewrite rules +
~12 guards + ~20 axioms). Phase 1a implements 15; Phase 1b extends.

Critical invariant: **guard precedes apply**. The BFS engine MUST call
`guard()` first; only on `GuardResult(ok=True)` is `apply()` legal. An unsound
application is structurally impossible because `apply()` may assume the guard
has already validated preconditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

import sympy as sp

from ..state import EqState


@dataclass(frozen=True)
class Action:
    """A concrete (rule, parameterization) pair.

    `params` is rule-specific (constants, signs, etc.). `target_path` is the
    structural index path into `lhs` or `rhs` per `ggmr_v10.pdf` §2.2.
    `target_side`: 'lhs', 'rhs', or 'both'.
    """

    rule_name: str
    params: tuple = ()
    target_path: tuple[int, ...] = ()
    target_side: str = "both"

    def canonical_key(self) -> str:
        """Deterministic ordering key for action enumeration."""
        return f"{self.rule_name}|{self.target_side}|{self.target_path}|{self.params}"


@dataclass(frozen=True)
class GuardResult:
    """Output of a rule's guard. `ok=False` rejects the action; `ok=True`
    accepts and may propagate new excluded values / side conditions."""

    ok: bool
    reason: str = ""
    new_excluded: frozenset = field(default_factory=frozenset)
    new_side_conditions: frozenset = field(default_factory=frozenset)

    @classmethod
    def passing(
        cls,
        new_excluded=None,
        new_side_conditions=None,
    ) -> "GuardResult":
        return cls(
            ok=True,
            new_excluded=frozenset(new_excluded or ()),
            new_side_conditions=frozenset(new_side_conditions or ()),
        )

    @classmethod
    def failing(cls, reason: str) -> "GuardResult":
        return cls(ok=False, reason=reason)


@runtime_checkable
class Rule(Protocol):
    """Guarded rewrite rule. Implementations register via `default_registry.register`.

    `training_safe` defaults to True (primitive rewrite). Oracle shortcuts
    (e.g. wrappers around `sympy.trigsimp` / `sympy.solveset`) set this to
    False so that BFS/SL/ExIt enumerate them only when `training_only=False`
    is passed to `Registry.enumerate_actions`. Inference paths leave the
    default, so oracles remain available for fast-mode evaluation.
    """

    name: str
    arity: int
    training_safe: bool

    def enumerate(self, state: EqState) -> Iterator[Action]: ...

    def guard(self, state: EqState, action: Action) -> GuardResult: ...

    def apply(self, state: EqState, action: Action) -> EqState: ...


def merge_guard_into_state(state: EqState, guard: GuardResult) -> EqState:
    """Apply a passing guard's side effects (new_excluded + new_side_conditions)
    to the state. The state's lhs/rhs are unchanged here — `Rule.apply()` is
    responsible for the structural rewrite.
    """
    if not guard.ok:
        raise ValueError(f"Cannot merge a failing guard: {guard.reason}")
    if not guard.new_excluded and not guard.new_side_conditions:
        return state
    return state.with_excluded(*guard.new_excluded).with_side_conditions(
        *guard.new_side_conditions
    )
