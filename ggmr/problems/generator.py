"""Reverse-application problem generator.

Algorithm (per `ggmr/PHASE1B_PREREG.md` §3.4):
1. Sample a canonical-target seed from a template.
2. Apply `depth` inverse rules from `default_inverse_registry`, biased by
   template-specific weights, to produce a "harder" state.
3. Run forward BFS with budget `max_nodes=5_000` to verify solvability.
4. If solved within `depth * 1.5` steps, accept; else discard and retry (max 3).
5. Return (initial_problem, original_target, forward_trace).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..expr.tree import canonical_repr, op_count
from ..rules.core import *  # noqa: F401,F403  (registers forward rules)
from ..search.bfs import SearchResult, bfs
from ..state import EqState
from .inverse_rules import InverseAction, InverseRegistry, default_inverse_registry
from .templates import TEMPLATES


@dataclass
class GeneratedProblem:
    template: str
    depth: int
    seed: int
    initial: EqState
    target: EqState
    forward_trace: list  # list of (state, action) pairs from BFS
    bfs_stats: dict
    applied_inverses: list = field(default_factory=list)
    astar_nodes_expanded: int = 0


class ReverseGenerator:
    def __init__(
        self,
        seed: int = 0,
        depth: int = 5,
        template: str = "linear",
        inverse_registry: Optional[InverseRegistry] = None,
        max_nodes: int = 5_000,
        max_depth_for_bfs: int = 30,
        accept_predicate: Optional[Callable[["GeneratedProblem"], bool]] = None,
        pre_bfs_complexity_max: Optional[int] = None,
    ):
        self.seed = seed
        self.depth = depth
        self.template = template
        self.inverse_registry = inverse_registry or default_inverse_registry
        self.max_nodes = max_nodes
        self.max_depth_for_bfs = max_depth_for_bfs
        self.accept_predicate = accept_predicate
        self.pre_bfs_complexity_max = pre_bfs_complexity_max
        self.rng = random.Random(seed)

    def _is_target_factory(self, target: EqState):
        target_lhs_repr = canonical_repr(target.lhs)
        target_rhs_repr = canonical_repr(target.rhs)

        def is_target(s: EqState) -> bool:
            return (
                canonical_repr(s.lhs) == target_lhs_repr
                and canonical_repr(s.rhs) == target_rhs_repr
            ) or s.is_canonical_target()

        return is_target

    def _generate_attempt(self) -> Optional[GeneratedProblem]:
        # 1. Sample seed
        if self.template not in TEMPLATES:
            raise ValueError(f"Unknown template: {self.template}")
        target = TEMPLATES[self.template](self.rng)

        # 2. Apply inverse rules
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
                if canonical_repr(new_state.lhs) == canonical_repr(state.lhs) and canonical_repr(
                    new_state.rhs
                ) == canonical_repr(state.rhs):
                    continue
                state = new_state
                applied_inverses.append(action)
                picked = True
                break
            if not picked:
                # Couldn't expand further at this step
                break

        if state == target:
            return None  # No structural change; reject

        initial = state

        # Optional structural-complexity gate: reject states whose op_count is so
        # high that BFS is virtually certain to exhaust its budget. This avoids
        # wasting 30-90s of search per pathological generation.
        if self.pre_bfs_complexity_max is not None:
            complexity = op_count(initial.lhs) + op_count(initial.rhs)
            if complexity > self.pre_bfs_complexity_max:
                return None

        # 3. Verify forward BFS solves it within slack
        is_target = self._is_target_factory(target)
        result: SearchResult = bfs(
            initial,
            is_target,
            max_nodes=self.max_nodes,
            max_depth=self.max_depth_for_bfs,
            check_soundness=False,
            problem_id=f"gen_{self.template}_{self.depth}_{self.seed}",
        )

        if not result.found:
            return None
        slack_limit = max(int(self.depth * 1.5), self.depth + 2)
        if result.num_steps > slack_limit:
            return None

        problem = GeneratedProblem(
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

    def generate_one(self, max_attempts: int = 5) -> Optional[GeneratedProblem]:
        """Generate a single problem. Returns None if all attempts fail."""
        for _ in range(max_attempts):
            self.rng = random.Random(self.seed)  # reset for reproducibility per attempt set
            self.seed += 1  # next attempt uses different seed
            problem = self._generate_attempt()
            if problem is not None:
                return problem
        return None
