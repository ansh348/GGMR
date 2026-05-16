"""MCTS search engine (AlphaZero-style PUCT) for ExIt training and inference.

Each move runs `num_simulations` PUCT simulations from the current state, then
commits to the most-visited child. Per-move root-visit distributions are
collected and used as policy training targets in `ggmr.training.exit_loop`.

Value semantics: `value_fn(state) -> float in [0, 1]`, higher = closer to goal.
Convention: `Q = 1.0 / (1.0 + predicted_remaining_steps)`. This mapping puts
solved at 1.0 and 1-step-away at 0.5, giving PUCT a sharp gradient near the
goal (a flat `1 - steps/MAX_STEPS` mapping yields ~0.03 gap which the prior
term swamps for any reasonable c_puct).

Terminal preference: at SELECT time, if a child is solved-terminal and
unvisited, it's selected immediately. This is a standard "MCTS goal test"
shortcut and prevents the engine from spending sims exploring near-target
non-terminal subtrees while a 1-step-away terminal sits unvisited.

Policy semantics: `policy_fn(state, legal_rule_names) -> dict[str, float]` where
the returned dict gives `rule_name -> probability` summing to 1.0 over legal
rules. Per-instance priors are computed by MCTS (each rule's probability is
divided equally across legal instances of that rule).

The engine mirrors `ggmr.search.bfs.bfs` for expansion semantics: guard ->
apply -> merge_guard -> normalize, with check_soundness deferred to the final
trajectory verification step in `exit_loop`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..expr.tree import normalize
from ..rules.base import Action, merge_guard_into_state
from ..rules.registry import Registry, default_registry
from ..state import EqState

ValueFn = Callable[[EqState], float]
PolicyFn = Callable[[EqState, list[str]], dict[str, float]]

MAX_STEPS = 30
C_PUCT_DEFAULT = 1.5
SOLVED_VALUE = 1.0
DEAD_END_VALUE = 0.0


@dataclass
class MCTSNode:
    """One node of the MCTS tree. Children are created at expansion time as stubs;
    they are evaluated (and themselves expanded) by subsequent simulations."""

    state: EqState
    parent: Optional["MCTSNode"] = None
    incoming_action: Optional[Action] = None
    incoming_rule_name: Optional[str] = None
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    terminal_value: Optional[float] = None  # SOLVED_VALUE for target, DEAD_END_VALUE for no-moves
    children: list["MCTSNode"] = field(default_factory=list)
    expanded: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u


@dataclass
class MCTSStats:
    nodes_expanded: int = 0
    nodes_generated: int = 0
    value_evals: int = 0
    policy_evals: int = 0
    moves_taken: int = 0
    total_simulations: int = 0
    time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "value_evals": self.value_evals,
            "policy_evals": self.policy_evals,
            "moves_taken": self.moves_taken,
            "total_simulations": self.total_simulations,
            "time_ms": self.time_ms,
        }


@dataclass
class MCTSResult:
    found: bool
    final_state: Optional[EqState]
    path: list[tuple[EqState, Action]]
    visit_distributions: list[dict[str, float]]
    stats: MCTSStats

    @property
    def num_steps(self) -> int:
        return len(self.path)


def _enumerate_legal_with_apply(
    state: EqState,
    rules: Registry,
    *,
    training_only: bool = False,
) -> list[tuple[str, Action, EqState]]:
    """Run guard+apply+merge+normalize over every (rule, action) in canonical order.

    Mirrors A*/BFS expansion (cf. `ggmr/search/astar.py` lines 96-112). Actions
    that fail guard or raise during apply are silently skipped.
    """
    out: list[tuple[str, Action, EqState]] = []
    for rule, action in rules.enumerate_actions(state, training_only=training_only):
        guard = rule.guard(state, action)
        if not guard.ok:
            continue
        try:
            child = rule.apply(state, action)
        except Exception:
            continue
        if guard.new_excluded or guard.new_side_conditions:
            child = merge_guard_into_state(child, guard)
        child = child.with_lhs_rhs(normalize(child.lhs), normalize(child.rhs))
        out.append((rule.name, action, child))
    return out


def _select(node: MCTSNode, c_puct: float) -> MCTSNode:
    """Walk down following PUCT until reaching an unexpanded node, terminal node, or leaf.

    If any solved-terminal child is unvisited, prefer it (immediate goal test).
    Once a terminal child has been visited at least once, its Q=1.0 dominates
    PUCT naturally and the goal-test branch becomes a no-op.
    """
    while node.expanded and not node.is_terminal and node.children:
        unvisited_solved = next(
            (c for c in node.children
             if c.terminal_value == SOLVED_VALUE and c.visit_count == 0),
            None,
        )
        if unvisited_solved is not None:
            return unvisited_solved
        best = max(
            node.children,
            key=lambda c: c.puct_score(c_puct, node.visit_count),
        )
        node = best
    return node


def _expand(
    node: MCTSNode,
    rules: Registry,
    policy_fn: PolicyFn,
    is_target: Callable[[EqState], bool],
    stats: MCTSStats,
    *,
    training_only: bool = False,
) -> None:
    """Mark `node` as expanded; populate children with priors from `policy_fn`."""
    if node.expanded:
        return
    node.expanded = True

    legal = _enumerate_legal_with_apply(node.state, rules, training_only=training_only)
    stats.nodes_generated += len(legal)
    if not legal:
        node.terminal_value = DEAD_END_VALUE
        return

    legal_rule_names = sorted({rn for rn, _, _ in legal})
    rule_probs = policy_fn(node.state, legal_rule_names)
    stats.policy_evals += 1

    rule_counts: dict[str, int] = {}
    for rn, _, _ in legal:
        rule_counts[rn] = rule_counts.get(rn, 0) + 1

    for rule_name, action, child_state in legal:
        per_rule = rule_probs.get(rule_name, 0.0)
        instance_prior = per_rule / rule_counts[rule_name]
        is_solved = is_target(child_state)
        child = MCTSNode(
            state=child_state,
            parent=node,
            incoming_action=action,
            incoming_rule_name=rule_name,
            prior=instance_prior,
            terminal_value=(SOLVED_VALUE if is_solved else None),
        )
        node.children.append(child)


def _backprop(node: MCTSNode, value: float) -> None:
    cur: Optional[MCTSNode] = node
    while cur is not None:
        cur.visit_count += 1
        cur.value_sum += value
        cur = cur.parent


def _simulate(
    root: MCTSNode,
    rules: Registry,
    value_fn: ValueFn,
    policy_fn: PolicyFn,
    is_target: Callable[[EqState], bool],
    c_puct: float,
    stats: MCTSStats,
    *,
    training_only: bool = False,
) -> None:
    leaf = _select(root, c_puct)

    if leaf.is_terminal:
        _backprop(leaf, leaf.terminal_value)  # type: ignore[arg-type]
        return

    if not leaf.expanded:
        _expand(leaf, rules, policy_fn, is_target, stats, training_only=training_only)
        stats.nodes_expanded += 1
        if leaf.is_terminal:
            _backprop(leaf, leaf.terminal_value)  # type: ignore[arg-type]
            return

    v = float(value_fn(leaf.state))
    stats.value_evals += 1
    _backprop(leaf, v)


def _visit_distribution_by_rule(node: MCTSNode) -> dict[str, float]:
    """Aggregate child visit counts by rule_name and normalize to a probability distribution."""
    rule_visits: dict[str, int] = {}
    total = 0
    for child in node.children:
        if child.incoming_rule_name is None:
            continue
        rule_visits[child.incoming_rule_name] = (
            rule_visits.get(child.incoming_rule_name, 0) + child.visit_count
        )
        total += child.visit_count
    if total == 0:
        if not rule_visits:
            return {}
        u = 1.0 / len(rule_visits)
        return {r: u for r in rule_visits}
    return {r: v / total for r, v in rule_visits.items()}


def _best_child(node: MCTSNode) -> Optional[MCTSNode]:
    if not node.children:
        return None
    return max(node.children, key=lambda c: (c.visit_count, c.q_value))


def mcts_search(
    initial: EqState,
    is_target: Callable[[EqState], bool],
    *,
    value_fn: ValueFn,
    policy_fn: PolicyFn,
    num_simulations: int = 400,
    max_moves: int = 20,
    rules: Optional[Registry] = None,
    c_puct: float = C_PUCT_DEFAULT,
    training_only: bool = False,
) -> MCTSResult:
    """MCTS planning search.

    At each move, runs `num_simulations` PUCT simulations from the current state
    (fresh tree each move), commits to the child with highest visit count, and
    advances. Halts when `is_target(state)` is True or `max_moves` is reached.
    """
    if rules is None:
        rules = default_registry

    stats = MCTSStats()
    t0 = time.perf_counter()

    if is_target(initial):
        stats.time_ms = (time.perf_counter() - t0) * 1000
        return MCTSResult(found=True, final_state=initial, path=[], visit_distributions=[], stats=stats)

    path: list[tuple[EqState, Action]] = []
    visit_dists: list[dict[str, float]] = []
    current = initial

    for _ in range(max_moves):
        root = MCTSNode(state=current)
        for _ in range(num_simulations):
            _simulate(root, rules, value_fn, policy_fn, is_target, c_puct, stats, training_only=training_only)
            stats.total_simulations += 1

        visit_dists.append(_visit_distribution_by_rule(root))
        best = _best_child(root)
        if best is None or best.visit_count == 0 or best.incoming_action is None:
            break

        path.append((current, best.incoming_action))
        current = best.state
        stats.moves_taken += 1

        if is_target(current):
            stats.time_ms = (time.perf_counter() - t0) * 1000
            return MCTSResult(
                found=True,
                final_state=current,
                path=path,
                visit_distributions=visit_dists,
                stats=stats,
            )

    stats.time_ms = (time.perf_counter() - t0) * 1000
    return MCTSResult(
        found=False,
        final_state=current,
        path=path,
        visit_distributions=visit_dists,
        stats=stats,
    )


def uniform_policy(state: EqState, legal_rule_names: list[str]) -> dict[str, float]:
    """Uniform prior over legal rule names. Used for oracle / value-only sanity checks."""
    if not legal_rule_names:
        return {}
    p = 1.0 / len(legal_rule_names)
    return {name: p for name in legal_rule_names}


def steps_to_q(steps: float) -> float:
    """Convert predicted/true remaining_steps to a Q-value in [0, 1].

    Solved state (steps=0) -> 1.0. 1 step -> 0.5. 5 steps -> 0.167. Used by
    both the oracle and the PolicyAdvisor's value loader.
    """
    return 1.0 / (1.0 + max(float(steps), 0.0))


def oracle_value_factory(
    is_target: Callable[[EqState], bool],
    *,
    bfs_max_nodes: int = 5000,
    bfs_max_depth: int = 40,
) -> ValueFn:
    """Value oracle for sanity-checking the MCTS engine without trained networks.

    Runs BFS to compute true distance from each state to a goal-satisfying
    state, then returns `steps_to_q(true_steps)`. Cached.

    Used by the oracle sanity test in `__main__` and `ggmr/tests/test_mcts.py`.
    """
    from ..training.extract_pairs import _build_is_target  # noqa: F401  (sanity import)
    from .bfs import bfs

    cache: dict[EqState, float] = {}

    def value_fn(state: EqState) -> float:
        cached = cache.get(state)
        if cached is not None:
            return cached
        if is_target(state):
            v = SOLVED_VALUE
        else:
            result = bfs(
                state,
                is_target,
                max_nodes=bfs_max_nodes,
                max_depth=bfs_max_depth,
                check_soundness=False,
                problem_id="<oracle>",
            )
            if result.found:
                v = steps_to_q(len(result.path))
            else:
                v = 0.0
        cache[state] = v
        return v

    return value_fn


def _oracle_sanity_main() -> int:
    """Driver: `python -m ggmr.search.mcts --oracle-test`. Returns exit code (0=ok)."""
    import ggmr.rules.core  # noqa: F401  (register rules)
    from ggmr.training.extract_pairs import _build_is_target

    problems = [
        ("lin_already_solved", "x", "5", "x", "5"),
        ("lin_one_step", "x + 3", "5", "x", "2"),
        ("lin_two_step", "2*x + 3", "7", "x", "2"),
        ("quad_factor", "x**2 - 4", "0", "(x-2)*(x+2)", "0"),
        ("rat_basic", "(x - 1)/2", "3", "x", "7"),
    ]
    ok = 0
    for pid, lhs, rhs, tlhs, trhs in problems:
        initial = EqState.from_strings(lhs, rhs)
        target = EqState.from_strings(tlhs, trhs)
        is_target = _build_is_target(target)
        value_fn = oracle_value_factory(is_target)
        result = mcts_search(
            initial,
            is_target,
            value_fn=value_fn,
            policy_fn=uniform_policy,
            num_simulations=80,
            max_moves=10,
        )
        status = "OK" if result.found else "FAIL"
        print(
            f"[{status}] {pid}: found={result.found} "
            f"steps={result.num_steps} "
            f"sims={result.stats.total_simulations} "
            f"nodes_expanded={result.stats.nodes_expanded} "
            f"time_ms={result.stats.time_ms:.1f}"
        )
        if result.found:
            ok += 1
    print(f"\n{ok}/{len(problems)} oracle problems solved")
    return 0 if ok == len(problems) else 1


if __name__ == "__main__":
    import sys

    if "--oracle-test" in sys.argv:
        sys.exit(_oracle_sanity_main())
    print("Usage: python -m ggmr.search.mcts --oracle-test")
    sys.exit(2)
