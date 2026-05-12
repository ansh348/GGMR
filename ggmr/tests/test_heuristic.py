"""Phase 0 monotonicity-rate parity test.

Per `ggmr/PHASE1A_PREREG.md` §3.3, the graduated heuristic must reproduce
H = 0.6417 ± 0.05 over the Phase 0 traces. Per-feature rates likewise within ±0.05.

The Phase 0 published numbers (from `phase0/PHASE0_FINDINGS.md`):
    composite (z-scored):  0.6417
    depth:                 0.863
    ops:                   0.825
    leaves:                0.817
    isolation:             0.858
"""

from __future__ import annotations

import sympy as sp
import yaml

from ggmr.heuristics.composite import (
    StateFeatures,
    state_features,
    ZScoredCompositeHeuristic,
    composite_z,
)
from ggmr.state import EqState
from ggmr.tests.conftest import PHASE0_PROBLEMS_PATH


def _step_rate(values: list[float], strict: bool = False) -> float:
    if len(values) < 2:
        return 1.0
    transitions = len(values) - 1
    if strict:
        decreases = sum(1 for i in range(transitions) if values[i + 1] < values[i])
    else:
        decreases = sum(1 for i in range(transitions) if values[i + 1] <= values[i])
    return decreases / transitions


def _trace_states(entry: dict) -> list[EqState]:
    """Build the EqState sequence (initial + each trace step) for a Phase 0 problem."""
    states: list[EqState] = []
    var_name = entry["variable"]
    states.append(
        EqState.from_strings(entry["initial"]["lhs"], entry["initial"]["rhs"], var_name=var_name)
    )
    for step in entry["trace"]:
        states.append(
            EqState.from_strings(step["lhs"], step["rhs"], var_name=var_name)
        )
    return states


def test_phase0_monotonicity_reproduced():
    with open(PHASE0_PROBLEMS_PATH, "r", encoding="utf-8") as f:
        problems = yaml.safe_load(f)

    # Compute features for every state of every trace
    all_features: list[StateFeatures] = []
    per_problem_states: list[tuple[str, list[StateFeatures]]] = []
    for entry in problems:
        states = _trace_states(entry)
        feats = [state_features(s) for s in states]
        per_problem_states.append((entry["id"], feats))
        all_features.extend(feats)

    # Z-scored composite over the full corpus (Phase 0's pooling)
    composite_seq = composite_z([f.to_phase0_row() for f in all_features])

    # Slice composite_seq back per-problem
    idx = 0
    per_problem_composite_rates = []
    per_problem_depth_rates = []
    per_problem_ops_rates = []
    per_problem_leaves_rates = []
    per_problem_isolation_rates = []
    for pid, feats in per_problem_states:
        n = len(feats)
        comp_slice = composite_seq[idx : idx + n]
        idx += n
        per_problem_composite_rates.append(_step_rate(comp_slice))
        per_problem_depth_rates.append(_step_rate([f.depth for f in feats]))
        per_problem_ops_rates.append(_step_rate([f.ops for f in feats]))
        per_problem_leaves_rates.append(_step_rate([f.leaves for f in feats]))
        per_problem_isolation_rates.append(_step_rate([f.isolation for f in feats]))

    composite_mean = sum(per_problem_composite_rates) / len(per_problem_composite_rates)
    depth_mean = sum(per_problem_depth_rates) / len(per_problem_depth_rates)
    ops_mean = sum(per_problem_ops_rates) / len(per_problem_ops_rates)
    leaves_mean = sum(per_problem_leaves_rates) / len(per_problem_leaves_rates)
    isolation_mean = sum(per_problem_isolation_rates) / len(per_problem_isolation_rates)

    # Phase 0 published values
    expected = {
        "composite": 0.6417,
        "depth": 0.863,
        "ops": 0.825,
        "leaves": 0.817,
        "isolation": 0.858,
    }
    actual = {
        "composite": composite_mean,
        "depth": depth_mean,
        "ops": ops_mean,
        "leaves": leaves_mean,
        "isolation": isolation_mean,
    }
    tol = 0.05
    deltas = {k: abs(actual[k] - expected[k]) for k in expected}
    failures = {k: (actual[k], expected[k], deltas[k]) for k in expected if deltas[k] >= tol}
    assert not failures, (
        "Phase 0 parity failed:\n"
        + "\n".join(f"  {k}: actual={a:.4f} expected={e:.4f} delta={d:.4f}" for k, (a, e, d) in failures.items())
    )


def test_zscored_heuristic_evaluate_returns_float():
    """Sanity: ZScoredCompositeHeuristic produces a float for any new state."""
    s1 = EqState.from_strings("x", "2")
    s2 = EqState.from_strings("2*x", "4")
    s3 = EqState.from_strings("2*x + 3", "7")
    h = ZScoredCompositeHeuristic.from_states([s1, s2, s3])
    v = h.evaluate(s1)
    assert isinstance(v, float)
