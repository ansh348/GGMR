"""Monotonicity rate computation per PHASE0_PREREG.md §5.

Primary metric: step-level non-strict decrease rate of composite feature.
Also tracks per-feature, per-category, and AC-variant σ.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev


def step_rate(values: list[float], strict: bool = False) -> float:
    """Fraction of consecutive pairs (v[i], v[i+1]) where v[i+1] decreases.

    If `strict=False` (default): counts v[i+1] <= v[i] (plateaus included).
    If `strict=True`: counts v[i+1] < v[i] only.

    For sequences of length < 2, returns 1.0 by convention (vacuously monotone).
    """
    if len(values) < 2:
        return 1.0
    transitions = len(values) - 1
    if strict:
        decreases = sum(1 for i in range(transitions) if values[i + 1] < values[i])
    else:
        decreases = sum(1 for i in range(transitions) if values[i + 1] <= values[i])
    return decreases / transitions


@dataclass(frozen=True)
class PerProblemRate:
    problem_id: str
    category: str
    n_steps: int
    depth_rate: float
    ops_rate: float
    leaves_rate: float
    isolation_rate: float
    composite_rate: float
    composite_strict_rate: float


@dataclass(frozen=True)
class AggregateResult:
    composite_mean: float
    depth_mean: float
    ops_mean: float
    leaves_mean: float
    isolation_mean: float
    composite_strict_mean: float
    per_category: dict[str, float]
    per_problem: list[PerProblemRate]


def compute_per_problem(
    problem_id: str,
    category: str,
    feature_series: dict[str, list[float]],
) -> PerProblemRate:
    """Compute per-problem monotonicity from a dict of feature_name → list of values
    along the trace.

    Required keys: depth, ops, leaves, isolation, composite.
    """
    n = max(0, len(feature_series["composite"]) - 1)
    return PerProblemRate(
        problem_id=problem_id,
        category=category,
        n_steps=n,
        depth_rate=step_rate(feature_series["depth"]),
        ops_rate=step_rate(feature_series["ops"]),
        leaves_rate=step_rate(feature_series["leaves"]),
        isolation_rate=step_rate(feature_series["isolation"]),
        composite_rate=step_rate(feature_series["composite"]),
        composite_strict_rate=step_rate(feature_series["composite"], strict=True),
    )


def aggregate(per_problem: list[PerProblemRate]) -> AggregateResult:
    """Mean across problems, plus per-category breakdown."""
    by_cat: dict[str, list[float]] = defaultdict(list)
    for p in per_problem:
        by_cat[p.category].append(p.composite_rate)

    return AggregateResult(
        composite_mean=mean(p.composite_rate for p in per_problem),
        depth_mean=mean(p.depth_rate for p in per_problem),
        ops_mean=mean(p.ops_rate for p in per_problem),
        leaves_mean=mean(p.leaves_rate for p in per_problem),
        isolation_mean=mean(p.isolation_rate for p in per_problem),
        composite_strict_mean=mean(p.composite_strict_rate for p in per_problem),
        per_category={cat: mean(vals) for cat, vals in by_cat.items()},
        per_problem=per_problem,
    )


def variant_sigma(per_variant_rates: dict[str, float]) -> float:
    """Standard deviation of monotonicity rates across (original + variants).

    Returns 0.0 if fewer than 2 entries.
    """
    rates = list(per_variant_rates.values())
    if len(rates) < 2:
        return 0.0
    return stdev(rates)


def decision(headline: float, threshold_high: float = 0.80,
             threshold_low: float = 0.50, tie_breaker: float = 0.02) -> str:
    """Apply the §6 decision rule from PHASE0_PREREG.md.

    Tie-breaker: if |headline - threshold| < tie_breaker, round AGAINST the
    stronger claim (conservative side wins).
    """
    # Conservative rounding: a value within tie_breaker of a threshold falls
    # on the LOWER side.
    eff_high = threshold_high + tie_breaker
    eff_low = threshold_low + tie_breaker
    if headline >= eff_high:
        return "PHASE_A_MEANINGFUL_BASELINE"
    if headline >= eff_low:
        return "PHASE_A_WEAK_NONMYOPIC_PRIMARY"
    return "PHASE_A_NEGATIVE_RESULT"


def decision_text(label: str) -> str:
    return {
        "PHASE_A_MEANINGFUL_BASELINE": (
            "Phase A is a meaningful baseline. Continue paper as planned: "
            "§4.1 hand-heuristic + beam search is reportable."
        ),
        "PHASE_A_WEAK_NONMYOPIC_PRIMARY": (
            "Phase A is weak. Paper restructures to emphasize non-myopic "
            "learned value (Phase B/C) as primary contribution; §4.1 becomes a foil."
        ),
        "PHASE_A_NEGATIVE_RESULT": (
            "Phase A is a negative result. Paper restructures around demonstrating "
            "why naive structural distance fails for algebraic rewrite."
        ),
    }[label]
