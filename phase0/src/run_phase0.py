"""Phase 0 orchestrator: load → verify → features → composite → monotonicity →
variants → CSVs → report.

Run as a module:
    & .\\.venv\\Scripts\\python.exe -m phase0.src.run_phase0
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .features import FeatureRow, composite_z, features
from .monotonicity import (
    AggregateResult,
    PerProblemRate,
    aggregate,
    compute_per_problem,
    step_rate,
    variant_sigma,
)
from .report import generate_report
from .trace_loader import Problem, load_problems, validate_canonical_targets
from .variants import (
    assert_variants_equivalent,
    make_variants,
)
from .verifier import verify_all


PHASE0_DIR = Path(__file__).resolve().parent.parent
PROBLEMS_PATH = PHASE0_DIR / "problems" / "problems.yaml"
OUTPUTS_DIR = PHASE0_DIR / "outputs"


def _series(problem: Problem) -> tuple[list[FeatureRow], list[str]]:
    """Return (feature rows, equation strings) for every state on a trace."""
    rows: list[FeatureRow] = []
    eq_strs: list[str] = []
    for eq in problem.states:
        rows.append(features(eq, problem.variable))
        eq_strs.append(f"{eq.lhs} = {eq.rhs}")
    return rows, eq_strs


def _problem_to_traces_rows(
    problem: Problem,
    rows: list[FeatureRow],
    eq_strs: list[str],
    composites: list[float],
    is_variant: bool = False,
) -> list[dict]:
    out: list[dict] = []
    for step_idx, (r, e, c) in enumerate(zip(rows, eq_strs, composites)):
        out.append(
            {
                "problem_id": problem.id,
                "category": problem.category,
                "is_variant": is_variant,
                "step_idx": step_idx,
                "expr": e,
                "depth": r.depth,
                "ops": r.ops,
                "leaves": r.leaves,
                "isolation": r.isolation,
                "composite": c,
            }
        )
    return out


def _per_problem_from_rows(problem: Problem,
                           rows: list[FeatureRow],
                           composites: list[float]) -> PerProblemRate:
    feature_series = {
        "depth": [r.depth for r in rows],
        "ops": [r.ops for r in rows],
        "leaves": [r.leaves for r in rows],
        "isolation": [r.isolation for r in rows],
        "composite": composites,
    }
    return compute_per_problem(problem.id, problem.category, feature_series)


def main() -> int:
    print("=" * 72)
    print("PHASE 0 — GGMR Prerequisite Validation")
    print("=" * 72)

    # 1. Load problems
    problems = load_problems(PROBLEMS_PATH)
    print(f"[load] {len(problems)} problems loaded from {PROBLEMS_PATH.name}")

    # 2. Validate canonical targets
    target_failures = validate_canonical_targets(problems)
    if target_failures:
        print("[FAIL] canonical_target mismatches:")
        for f in target_failures:
            print(f"  - {f}")
        return 1
    print(f"[ok]   canonical_target match for all {len(problems)} problems")

    # 3. Verify step legality
    try:
        checks = verify_all(problems, strict=True)
        print(f"[ok]   {len(checks)} step transitions verified (solution-set equality)")
    except Exception as e:
        print(f"[FAIL] step verification: {e}")
        return 2

    # 4. Compute features for original problems
    all_rows: list[FeatureRow] = []
    per_problem_rows: list[list[FeatureRow]] = []
    per_problem_eqs: list[list[str]] = []
    for p in problems:
        rows, eqs = _series(p)
        per_problem_rows.append(rows)
        per_problem_eqs.append(eqs)
        all_rows.extend(rows)

    # 5. Z-score composite across the whole corpus (originals only for headline)
    composites = composite_z(all_rows)
    # Slice composites back into per-problem lists
    per_problem_composites: list[list[float]] = []
    cursor = 0
    for rows in per_problem_rows:
        per_problem_composites.append(composites[cursor:cursor + len(rows)])
        cursor += len(rows)

    # 6. Per-problem monotonicity
    per_problem_rates: list[PerProblemRate] = []
    traces_records: list[dict] = []
    for p, rows, eqs, comp in zip(problems, per_problem_rows, per_problem_eqs,
                                   per_problem_composites):
        per_problem_rates.append(_per_problem_from_rows(p, rows, comp))
        traces_records.extend(_problem_to_traces_rows(p, rows, eqs, comp,
                                                      is_variant=False))

    agg = aggregate(per_problem_rates)
    print(f"[stat] composite mean monotonicity rate = {agg.composite_mean:.4f}")
    print(f"[stat] depth={agg.depth_mean:.3f} ops={agg.ops_mean:.3f} "
          f"leaves={agg.leaves_mean:.3f} isolation={agg.isolation_mean:.3f}")

    # 7. AC-variant analysis
    variant_records: list[dict] = []
    variant_summary_rows: list[dict] = []

    # Pool all states (originals + variants) for a shared z-score baseline
    variant_problems_all: list[tuple[Problem, list[Problem]]] = []
    pooled_rows: list[FeatureRow] = list(all_rows)
    for p in problems:
        variants = make_variants(p)
        # Sanity: variants preserve solution sets
        eq_failures = assert_variants_equivalent(p, variants)
        if eq_failures:
            print(f"[FAIL] AC-variant equivalence: {eq_failures}")
            return 3
        variant_problems_all.append((p, variants))
        for v in variants:
            for eq in v.states:
                pooled_rows.append(features(eq, v.variable))

    pooled_composites = composite_z(pooled_rows)

    # First N entries of pooled_composites correspond to original-problem states (in order).
    # Continue cursor for variants.
    cursor = sum(len(r) for r in per_problem_rows)
    for orig, variants in variant_problems_all:
        # Original rate computed with per_problem_composites (already done above)
        orig_rate = next(
            r.composite_rate for r in per_problem_rates if r.problem_id == orig.id
        )
        per_variant_rates = {orig.id: orig_rate}
        for v in variants:
            v_rows = [features(eq, v.variable) for eq in v.states]
            v_comp = pooled_composites[cursor:cursor + len(v_rows)]
            cursor += len(v_rows)
            v_rate = step_rate(v_comp)
            per_variant_rates[v.id] = v_rate
            variant_records.extend(_problem_to_traces_rows(v, v_rows,
                                                          [f"{e.lhs} = {e.rhs}" for e in v.states],
                                                          v_comp, is_variant=True))
        sigma = variant_sigma(per_variant_rates)
        variant_summary_rows.append({
            "problem_id": orig.id,
            "original_rate": orig_rate,
            "var1_addperm_rate": per_variant_rates.get(f"{orig.id}_var1_addperm", float("nan")),
            "var2_mulperm_rate": per_variant_rates.get(f"{orig.id}_var2_mulperm", float("nan")),
            "var3_rename_rate": per_variant_rates.get(f"{orig.id}_var3_rename", float("nan")),
            "sigma": sigma,
        })

    variant_summary_df = pd.DataFrame(variant_summary_rows)
    print(f"[stat] AC-variant mean sigma = {variant_summary_df['sigma'].mean():.4f}, "
          f"max sigma = {variant_summary_df['sigma'].max():.4f}")

    # 8. Write CSVs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    traces_path = OUTPUTS_DIR / "traces.csv"
    variants_path = OUTPUTS_DIR / "variants_traces.csv"
    monotonicity_path = OUTPUTS_DIR / "monotonicity.csv"
    variant_summary_path = OUTPUTS_DIR / "variants_summary.csv"

    pd.DataFrame(traces_records).to_csv(traces_path, index=False)
    pd.DataFrame(variant_records).to_csv(variants_path, index=False)
    pd.DataFrame(
        [
            {
                "problem_id": p.problem_id,
                "category": p.category,
                "n_steps": p.n_steps,
                "depth_rate": p.depth_rate,
                "ops_rate": p.ops_rate,
                "leaves_rate": p.leaves_rate,
                "isolation_rate": p.isolation_rate,
                "composite_rate": p.composite_rate,
                "composite_strict_rate": p.composite_strict_rate,
            }
            for p in per_problem_rates
        ]
    ).to_csv(monotonicity_path, index=False)
    variant_summary_df.to_csv(variant_summary_path, index=False)

    print(f"[ok]   wrote {traces_path.name}, {variants_path.name}, "
          f"{monotonicity_path.name}, {variant_summary_path.name}")

    # 9. Generate report
    report_path = OUTPUTS_DIR / "PHASE0_REPORT.md"
    generate_report(agg, variant_summary_df, str(traces_path), str(report_path))
    print(f"[ok]   wrote {report_path.name}")

    print("=" * 72)
    print(f"DONE. Headline H = {agg.composite_mean:.4f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
