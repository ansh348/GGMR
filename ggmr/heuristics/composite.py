"""Phase A hand-designed structural heuristics, graduated from `phase0/src/features.py`.

Per `ggmr_v10.pdf` §4.1, Phase A is the foil — its heuristic is a composite of
four hand-designed structural features. This module preserves Phase 0's exact
feature definitions (re-exported as-is) so monotonicity-rate parity holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
import sympy as sp

from ..state import EqState

# Re-export Phase 0 features verbatim. Importing from `phase0.src.features`
# directly keeps Phase 0 untouched while ensuring numerical parity.
from phase0.src.features import (  # noqa: F401
    composite_z,
    features as _phase0_features_for_eq,
    leaf_count,
    op_count,
    tree_depth,
    var_isolation_score,
    FeatureRow,
)


@dataclass(frozen=True)
class StateFeatures:
    depth: int
    ops: int
    leaves: int
    isolation: int

    def as_dict(self) -> dict:
        return {
            "depth": self.depth,
            "ops": self.ops,
            "leaves": self.leaves,
            "isolation": self.isolation,
        }

    def to_phase0_row(self) -> FeatureRow:
        """Convert to phase0's FeatureRow for use with phase0's composite_z."""
        return FeatureRow(
            depth=self.depth, ops=self.ops, leaves=self.leaves, isolation=self.isolation
        )


def state_features(state: EqState) -> StateFeatures:
    """Compute the four Phase A features on an EqState.

    Features are computed on `lhs - rhs` per `phase0/PHASE0_PREREG.md` §4.
    """
    eq = sp.Eq(state.lhs, state.rhs, evaluate=False)
    row = _phase0_features_for_eq(eq, state.var)
    return StateFeatures(
        depth=row.depth, ops=row.ops, leaves=row.leaves, isolation=row.isolation
    )


@runtime_checkable
class Heuristic(Protocol):
    """Pluggable heuristic interface used by A* / beam search (Phase 1b)."""

    def evaluate(self, state: EqState) -> float: ...


@dataclass
class WeightedSumCompositeHeuristic:
    """Weighted sum of raw structural features.

    Default weights (1.0 each) reproduce a simple linear combination — useful
    when we don't have a corpus to z-score against (per-state evaluation).
    """

    weights: dict[str, float] | None = None

    def evaluate(self, state: EqState) -> float:
        feats = state_features(state).as_dict()
        w = self.weights or {k: 1.0 for k in feats}
        return float(sum(feats[k] * w.get(k, 1.0) for k in feats))


@dataclass
class ZScoredCompositeHeuristic:
    """Phase 0's z-scored composite over a fixed corpus.

    The corpus is a list of (lhs, rhs, var) triples. At construction time we
    compute the per-feature mean and std across the corpus; `evaluate(state)`
    returns the z-scored sum for `state` against those statistics.
    """

    corpus_features: list[StateFeatures]
    _mu: np.ndarray | None = None
    _sd: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.corpus_features:
            raise ValueError(
                "ZScoredCompositeHeuristic requires a non-empty corpus for z-score statistics"
            )
        arr = np.array(
            [[f.depth, f.ops, f.leaves, f.isolation] for f in self.corpus_features],
            dtype=float,
        )
        self._mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        self._sd = np.where(sd < 1e-9, 1.0, sd)

    def evaluate(self, state: EqState) -> float:
        f = state_features(state)
        v = np.array([f.depth, f.ops, f.leaves, f.isolation], dtype=float)
        z = (v - self._mu) / self._sd
        return float(z.sum())

    @classmethod
    def from_states(cls, states: Iterable[EqState]) -> "ZScoredCompositeHeuristic":
        """Build by computing features for each state in the corpus."""
        feats = [state_features(s) for s in states]
        return cls(corpus_features=feats)
