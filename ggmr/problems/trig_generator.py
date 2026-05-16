"""Trig reverse-application problem generator (Phase 1.2b, Marcus).

Mirrors `ReverseGenerator` (generator.py) but for trigonometric identities.

Algorithm:
1. Sample a canonical-identity seed from `TRIG_TEMPLATES` (e.g. `sin²+cos²=1`).
2. Apply `depth` expansion-style inverse rules to the LHS (RHS fixed).
3. Run forward BFS with `training_only=True` (Marcus Constraint 1 — no oracle
   shortcuts in trace data) to verify the disguised LHS can be simplified back.
4. Accept if BFS solved within slack budget; reject and retry otherwise.

Termination predicate: `canonical_repr(lhs) == canonical_repr(rhs)` — both
sides have been reduced to the same syntactic canonical form. This is the
correct goal for identity verification (vs algebra's `x = k` predicate).

v1 supports `mode="verify_identity"` only. `solve_equation` and `simplify`
modes deferred.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import sympy as sp

from ..expr.tree import canonical_repr, op_count
from ..rules.core import *  # noqa: F401,F403  (registers forward rules)
from ..search.bfs import SearchResult, bfs
from ..state import EqState
from .inverse_rules import InverseAction, InverseRegistry
from .inverse_trig_rules import trig_inverse_registry
from .trig_templates import TRIG_TEMPLATES


@dataclass
class GeneratedTrigProblem:
    template: str
    depth: int
    seed: int
    initial: EqState
    target: EqState
    forward_trace: list  # list of (state, action) pairs from BFS
    bfs_stats: dict
    applied_inverses: list = field(default_factory=list)


def _identity_target_factory():
    """Predicate: True when BOTH sides canonicalize to the same string.

    This is the identity-verification termination criterion: any path that
    reduces LHS and RHS to a common canonical form counts as a solve.
    """

    def is_target(s: EqState) -> bool:
        return canonical_repr(s.lhs) == canonical_repr(s.rhs)

    return is_target


class TrigReverseGenerator:
    """Reverse-construction generator for trigonometric identity problems."""

    def __init__(
        self,
        seed: int = 0,
        depth: int = 3,
        template: str = "mixed",
        mode: str = "verify_identity",
        inverse_registry: Optional[InverseRegistry] = None,
        max_nodes: int = 5_000,
        max_depth_for_bfs: int = 20,
        accept_predicate: Optional[Callable[["GeneratedTrigProblem"], bool]] = None,
        pre_bfs_complexity_max: Optional[int] = None,
        require_nonzero_steps: bool = True,
    ):
        if mode != "verify_identity":
            raise ValueError(
                f"TrigReverseGenerator v1 only supports mode='verify_identity'; "
                f"got mode={mode!r}"
            )
        self.seed = seed
        self.depth = depth
        self.template = template
        self.mode = mode
        self.inverse_registry = inverse_registry or trig_inverse_registry
        self.max_nodes = max_nodes
        self.max_depth_for_bfs = max_depth_for_bfs
        self.accept_predicate = accept_predicate
        self.pre_bfs_complexity_max = pre_bfs_complexity_max
        self.require_nonzero_steps = require_nonzero_steps
        self.rng = random.Random(seed)

    def _generate_attempt(self) -> Optional[GeneratedTrigProblem]:
        if self.template not in TRIG_TEMPLATES:
            raise ValueError(f"Unknown trig template: {self.template}")
        target = TRIG_TEMPLATES[self.template](self.rng)

        state = target
        applied_inverses: list[InverseAction] = []
        rules = self.inverse_registry.all_rules()
        attempts_per_step = 5
        for _ in range(self.depth):
            picked = False
            for _try in range(attempts_per_step):
                rule = self.rng.choice(rules)
                actions = list(rule.enumerate(state, self.rng))
                if not actions:
                    continue
                action = actions[0]
                try:
                    new_state = rule.apply(state, action)
                except Exception:
                    continue
                # Discard if state didn't change structurally
                if (
                    canonical_repr(new_state.lhs) == canonical_repr(state.lhs)
                    and canonical_repr(new_state.rhs) == canonical_repr(state.rhs)
                ):
                    continue
                state = new_state
                applied_inverses.append(action)
                picked = True
                break
            if not picked:
                break

        if state == target:
            return None  # No structural change; reject

        # Bail early if state is pathologically complex (will exhaust BFS budget)
        if self.pre_bfs_complexity_max is not None:
            complexity = op_count(state.lhs) + op_count(state.rhs)
            if complexity > self.pre_bfs_complexity_max:
                return None

        initial = state
        is_target = _identity_target_factory()
        result: SearchResult = bfs(
            initial,
            is_target,
            max_nodes=self.max_nodes,
            max_depth=self.max_depth_for_bfs,
            check_soundness=False,
            problem_id=f"trig_{self.template}_d{self.depth}_s{self.seed}",
            training_only=True,
        )

        if not result.found:
            return None
        slack_limit = max(int(self.depth * 1.5), self.depth + 2)
        if result.num_steps > slack_limit:
            return None
        if self.require_nonzero_steps and result.num_steps == 0:
            return None

        problem = GeneratedTrigProblem(
            template=self.template,
            depth=self.depth,
            seed=self.seed,
            initial=initial,
            target=target,
            forward_trace=list(result.path),
            bfs_stats=result.stats.to_dict(),
            applied_inverses=list(applied_inverses),
        )
        if self.accept_predicate is not None and not self.accept_predicate(problem):
            return None
        return problem

    def generate_one(self, max_attempts: int = 5) -> Optional[GeneratedTrigProblem]:
        """Generate a single problem. Returns None if all attempts fail."""
        for _ in range(max_attempts):
            self.rng = random.Random(self.seed)
            self.seed += 1
            problem = self._generate_attempt()
            if problem is not None:
                return problem
        return None
