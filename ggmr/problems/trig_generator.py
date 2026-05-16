"""Trig reverse-application problem generator (Phase 1.2b, Marcus, v2).

v2 (after external review) fixes the 95.6% duplication rate from v1 by
changing how inverse rules are sampled at each depth step.

v1 algorithm: at each depth step, pick a random RULE from the inverse
registry, try its enumerate, retry up to 5 times if no actions. With 24
inverse rules of which 1-3 are typically applicable to any given state,
the probability of picking the wrong rule 5 times is ~85% — so most
"depth 5" problems actually got 0-1 expansions. The effective depth
distribution was heavily biased toward zero.

v2 algorithm: at each depth step, collect ALL (rule, action) pairs that
are applicable to the current state ACROSS all rules, then sample
uniformly from that pool. With 24 rules each potentially yielding multiple
actions, a typical state has 5-30 candidate actions to choose from.
Combined with the parameterized angle space in `trig_templates.py`, this
should reduce the dedup rate to ~30-50% (target: ≥10k unique pairs).

Also adds:
- Effective-depth gate: reject problems where fewer than `depth - 1`
  inverse rules actually fired (early `break` on exhausted candidates).
- `NoveltyGate` utility: optional accept_predicate that tracks seen
  `(lhs_srepr, rhs_srepr, remaining_steps)` keys and rejects problems
  contributing fewer than `min_new_rows` new keys. Useful for
  single-process generation; not yet wired into the parallel pipeline.
- BFS budget raised from 5000 to 10000 by default. Trig has 90 rules
  in the active registry and verification on compound-angle seeds was
  hitting the old budget cap.
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
    """Predicate: True when BOTH sides canonicalize to the same string."""

    def is_target(s: EqState) -> bool:
        return canonical_repr(s.lhs) == canonical_repr(s.rhs)

    return is_target


def trace_dedupe_keys(problem: GeneratedTrigProblem) -> set:
    """Return the (lhs_srepr, rhs_srepr, remaining_steps) keys that
    `GGMRDataset.from_jsonl` will dedup against on load. Used by
    `NoveltyGate` to project the dataset-level dedup forward into
    the generator.
    """
    keys: set = set()
    path = problem.forward_trace
    n_steps = len(path)
    for i, item in enumerate(path):
        state = item[0] if isinstance(item, tuple) else item
        remaining = n_steps - i
        keys.add((sp.srepr(state.lhs), sp.srepr(state.rhs), remaining))
    # final canonical state (remaining=0)
    keys.add((sp.srepr(problem.target.lhs), sp.srepr(problem.target.rhs), 0))
    return keys


class NoveltyGate:
    """Stateful accept_predicate that rejects problems contributing fewer
    than `min_new_rows` new (lhs_srepr, rhs_srepr, remaining_steps) keys
    relative to all previously-accepted problems.

    Use in single-process generation loops:

        gate = NoveltyGate(min_new_rows=2)
        gen = TrigReverseGenerator(seed=..., depth=..., accept_predicate=gate)
        while len(gate.seen) < 10_000:
            problem = gen.generate_one()
            if problem is not None:
                emit(problem)

    Not yet wired into the parallel multiprocess pipeline — each worker
    process has its own NoveltyGate, so per-worker novelty doesn't compose.
    """

    def __init__(self, min_new_rows: int = 2):
        self.min_new_rows = min_new_rows
        self.seen: set = set()
        self.rejected: int = 0
        self.accepted: int = 0

    def __call__(self, problem: GeneratedTrigProblem) -> bool:
        keys = trace_dedupe_keys(problem)
        new_keys = keys - self.seen
        if len(new_keys) < self.min_new_rows:
            self.rejected += 1
            return False
        self.seen.update(new_keys)
        self.accepted += 1
        return True


class TrigReverseGenerator:
    """Reverse-construction generator for trigonometric identity problems."""

    def __init__(
        self,
        seed: int = 0,
        depth: int = 3,
        template: str = "mixed",
        mode: str = "verify_identity",
        inverse_registry: Optional[InverseRegistry] = None,
        max_nodes: int = 10_000,
        max_depth_for_bfs: int = 20,
        accept_predicate: Optional[Callable[["GeneratedTrigProblem"], bool]] = None,
        pre_bfs_complexity_max: Optional[int] = None,
        require_nonzero_steps: bool = True,
        effective_depth_tolerance: int = 1,
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
        # Reject problems with effective depth < depth - tolerance.
        # tolerance=1 means a requested depth=5 must produce >=4 expansions.
        self.effective_depth_tolerance = effective_depth_tolerance
        self.rng = random.Random(seed)

    def _applicable_inverse_actions(self, state: EqState) -> list:
        """Collect (rule, action) pairs across ALL inverse rules for `state`.

        v2's central fix: instead of picking a random rule and hoping it
        applies, enumerate every rule and gather every applicable action.
        Then the caller samples uniformly from the union.
        """
        candidates: list = []
        for rule in self.inverse_registry.all_rules():
            try:
                actions = list(rule.enumerate(state, self.rng))
            except Exception:
                continue
            for action in actions:
                candidates.append((rule, action))
        return candidates

    def _generate_attempt(self) -> Optional[GeneratedTrigProblem]:
        if self.template not in TRIG_TEMPLATES:
            raise ValueError(f"Unknown trig template: {self.template}")
        target = TRIG_TEMPLATES[self.template](self.rng)

        state = target
        applied_inverses: list[InverseAction] = []
        retries_per_step = 3

        for _depth_step in range(self.depth):
            candidates = self._applicable_inverse_actions(state)
            if not candidates:
                break  # No further expansion possible
            self.rng.shuffle(candidates)
            progressed = False
            # Try up to `retries_per_step` distinct candidates per step
            for rule, action in candidates[:retries_per_step]:
                try:
                    new_state = rule.apply(state, action)
                except Exception:
                    continue
                # Skip no-ops (SymPy canonicalized back to the same form)
                if (
                    canonical_repr(new_state.lhs) == canonical_repr(state.lhs)
                    and canonical_repr(new_state.rhs) == canonical_repr(state.rhs)
                ):
                    continue
                state = new_state
                applied_inverses.append(action)
                progressed = True
                break
            if not progressed:
                break  # Exhausted retries on this step → stop expanding

        if state == target:
            return None  # No structural change

        # Effective-depth gate: reject "depth=5" problems that only got 1
        # expansion to fire. The v1 dedup explosion was largely from these
        # near-trivial outputs collapsing onto a tiny set of shapes.
        min_effective_depth = max(1, self.depth - self.effective_depth_tolerance)
        if len(applied_inverses) < min_effective_depth:
            return None

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
