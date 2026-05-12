"""Rule registry: canonical iteration order for BFS determinism.

Rules are iterated in registration order. Within a rule, actions are iterated
in `Action.canonical_key()` lexicographic order. This guarantees byte-identical
BFS output across runs (Phase 1a pre-reg §3.4).
"""

from __future__ import annotations

from typing import Iterator

from ..state import EqState
from .base import Action, Rule


class Registry:
    """Ordered collection of rules. `register()` is idempotent on `name`."""

    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}
        self._order: list[str] = []

    def register(self, rule: Rule) -> Rule:
        if rule.name in self._rules:
            # Re-registering the same name is allowed (e.g., during test reloads),
            # but the rule object must be identical. Otherwise it's a programming error.
            existing = self._rules[rule.name]
            if existing is not rule:
                raise ValueError(
                    f"Rule {rule.name!r} already registered with a different object"
                )
            return rule
        self._rules[rule.name] = rule
        self._order.append(rule.name)
        return rule

    def get(self, name: str) -> Rule:
        return self._rules[name]

    def names(self) -> list[str]:
        return list(self._order)

    def rules(self) -> list[Rule]:
        return [self._rules[n] for n in self._order]

    def enumerate_actions(self, state: EqState) -> Iterator[tuple[Rule, Action]]:
        """Yield all (rule, action) pairs in canonical order."""
        for name in self._order:
            rule = self._rules[name]
            actions = list(rule.enumerate(state))
            actions.sort(key=lambda a: a.canonical_key())
            for a in actions:
                yield rule, a

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, name: object) -> bool:
        return name in self._rules


default_registry = Registry()
