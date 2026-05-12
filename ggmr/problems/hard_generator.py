"""Hard-problem generator: wraps `ReverseGenerator` with an A*-difficulty filter.

Produces problems where A* with `WeightedSumCompositeHeuristic` expands at
least `min_astar_nodes` nodes — i.e., problems the Phase A foil heuristic
struggles with. These are the training/eval targets for the Phase 2 learned
value network.

Four recipes (each = template + weighted multiset of inverse rules) cover
the hardness families described in the plan:

    - nested_rational       (linear template + heavy InvNestInRational)
    - complete_square       (linear_irrational template + InvDisguiseByExpansion)
    - cross_side_rational   (rational template + InvNestInRational/Split)
    - polynomial_disguised  (polynomial template + InvDisguiseByExpansion)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..expr.tree import canonical_repr
from ..heuristics.composite import WeightedSumCompositeHeuristic
from ..search.astar import astar
from ..state import EqState
from .generator import GeneratedProblem, ReverseGenerator
from .hard_inverse_rules import (
    InvDisguiseByExpansion,
    InvEmbedInFraction,
    InvNestInRational,
    InvSplitAcrossSides,
    hard_inverse_registry,
)
from .inverse_rules import (
    InvAddToBothSides,
    InvClearFractions,
    InvCombineLikeTerms,
    InvDistributeOverSum,
    InvExpandProduct,
    InvFlipSides,
    InvMultiplyBothSides,
    InverseRegistry,
)


# ---------------------------------------------------------------------------
# A* difficulty filter
# ---------------------------------------------------------------------------


def make_astar_filter(
    min_nodes: int = 50,
    max_nodes: int = 50_000,
    max_depth: int = 40,
):
    """Build an `accept_predicate` that runs A* with the hand heuristic.

    Accepts a `GeneratedProblem` iff A* (with `WeightedSumCompositeHeuristic`)
    solves it and expands >= min_nodes nodes. Stores the measured node count
    on `problem.astar_nodes_expanded` as a side effect so it can be emitted
    to YAML.

    Note: do NOT pre-filter by BFS node count. A problem can be easy for BFS
    (small node count) but very hard for A* if the heuristic actively
    misguides — that's exactly the "hard for hand heuristic" signal we want.
    """
    heur = WeightedSumCompositeHeuristic()

    def _is_target_for(target: EqState):
        t_l = canonical_repr(target.lhs)
        t_r = canonical_repr(target.rhs)

        def chk(s: EqState) -> bool:
            return (
                canonical_repr(s.lhs) == t_l and canonical_repr(s.rhs) == t_r
            ) or s.is_canonical_target()

        return chk

    def predicate(problem: GeneratedProblem) -> bool:
        try:
            result = astar(
                problem.initial,
                _is_target_for(problem.target),
                heuristic=heur,
                max_nodes=max_nodes,
                max_depth=max_depth,
                check_soundness=False,
                problem_id=f"filter_{problem.template}_{problem.seed}",
            )
        except Exception:
            return False
        if not result.found:
            return False
        problem.astar_nodes_expanded = int(result.stats.nodes_expanded)
        return result.stats.nodes_expanded >= min_nodes

    return predicate


# ---------------------------------------------------------------------------
# Recipe registries
# ---------------------------------------------------------------------------


def _registry_from_weights(weights: dict) -> InverseRegistry:
    """Build an InverseRegistry from a {rule_instance: count} dict.

    Counts > 1 duplicate the rule in `.all_rules()`, which biases the random
    selection in `ReverseGenerator._generate_attempt` (which uses rng.choice).
    Fractional weights round to 1 (rule still included with single weight).
    """
    reg = InverseRegistry()
    for rule, weight in weights.items():
        n = max(1, int(round(weight)))
        for _ in range(n):
            reg.register(rule)
    return reg


def _nested_rational_registry() -> InverseRegistry:
    # Lower weight on the "explosive" new rules so chains don't blow up sympy.
    return _registry_from_weights({
        InvNestInRational():          1,
        InvEmbedInFraction():         2,
        InvSplitAcrossSides():        4,
        InvClearFractions():          2,
        InvAddToBothSides():          2,
        InvMultiplyBothSides():       1,
        InvCombineLikeTerms():        1,
    })


def _complete_square_registry() -> InverseRegistry:
    return _registry_from_weights({
        InvDisguiseByExpansion():     2,
        InvAddToBothSides():          2,
        InvMultiplyBothSides():       1,
        InvCombineLikeTerms():        2,
        InvDistributeOverSum():       1,
    })


def _cross_side_rational_registry() -> InverseRegistry:
    return _registry_from_weights({
        InvNestInRational():          1,
        InvSplitAcrossSides():        4,
        InvClearFractions():          2,
        InvEmbedInFraction():         2,
        InvAddToBothSides():          2,
        InvMultiplyBothSides():       1,
    })


def _polynomial_disguised_registry() -> InverseRegistry:
    return _registry_from_weights({
        InvDisguiseByExpansion():     2,
        InvCombineLikeTerms():        2,
        InvAddToBothSides():          2,
        InvExpandProduct():           1,
        InvMultiplyBothSides():       1,
        InvDistributeOverSum():       1,
    })


@dataclass(frozen=True)
class Recipe:
    name: str
    template: str
    registry_factory: callable = field(repr=False)


RECIPES: list[Recipe] = [
    Recipe("nested_rational",      "linear",     _nested_rational_registry),
    # complete_square uses quadratic_seed so InvDisguiseByExpansion can fire
    # (which requires lhs to be a Mul of Add factors). The forward path on the
    # disguised quadratic requires FACTOR_POLYNOMIAL or COMPLETE_THE_SQUARE,
    # both rare rules the structural heuristic doesn't anticipate.
    Recipe("complete_square",      "quadratic",  _complete_square_registry),
    Recipe("cross_side_rational",  "rational",   _cross_side_rational_registry),
    Recipe("polynomial_disguised", "polynomial", _polynomial_disguised_registry),
]


RECIPES_BY_NAME: dict[str, Recipe] = {r.name: r for r in RECIPES}


# ---------------------------------------------------------------------------
# HardProblemGenerator
# ---------------------------------------------------------------------------


@dataclass
class HardProblemGenerator:
    """Generator that wraps `ReverseGenerator` with an A*-difficulty filter."""

    recipe: Recipe
    depth: int = 20
    seed: int = 0
    min_astar_nodes: int = 50
    max_bfs_nodes: int = 50_000
    max_bfs_depth: int = 40
    astar_max_nodes: int = 50_000
    astar_max_depth: int = 40
    pre_bfs_complexity_max: Optional[int] = 60

    def __post_init__(self) -> None:
        self._predicate = make_astar_filter(
            min_nodes=self.min_astar_nodes,
            max_nodes=self.astar_max_nodes,
            max_depth=self.astar_max_depth,
        )
        self._gen = ReverseGenerator(
            seed=self.seed,
            depth=self.depth,
            template=self.recipe.template,
            inverse_registry=self.recipe.registry_factory(),
            max_nodes=self.max_bfs_nodes,
            max_depth_for_bfs=self.max_bfs_depth,
            accept_predicate=self._predicate,
            pre_bfs_complexity_max=self.pre_bfs_complexity_max,
        )

    def generate_one(self, max_attempts: int = 10) -> Optional[GeneratedProblem]:
        return self._gen.generate_one(max_attempts=max_attempts)
