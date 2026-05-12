"""Generate PHASE0_REPORT.md from CSV outputs.

Implements the headline + per-feature + per-category + AC-variant σ tables
specified in PHASE0_PREREG.md §7 and §9.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from statistics import mean

import pandas as pd

from .monotonicity import AggregateResult, decision, decision_text


def _fmt_pct(x: float) -> str:
    return f"{x:.3f}"


def _table_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table."""
    if df.empty:
        return "_(empty)_\n"
    cols = list(df.columns)
    out = "| " + " | ".join(str(c) for c in cols) + " |\n"
    out += "|" + "|".join("---" for _ in cols) + "|\n"
    for _, row in df.iterrows():
        out += "| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in row
        ) + " |\n"
    return out


def generate_report(
    aggregate: AggregateResult,
    variant_summary: pd.DataFrame,
    traces_csv: str,
    output_path: str | Path,
) -> None:
    """Write PHASE0_REPORT.md based on aggregated results."""
    headline = aggregate.composite_mean
    label = decision(headline)
    decision_msg = decision_text(label)

    # Per-feature summary
    feat_df = pd.DataFrame(
        [
            ("composite (primary)", aggregate.composite_mean),
            ("depth", aggregate.depth_mean),
            ("ops", aggregate.ops_mean),
            ("leaves", aggregate.leaves_mean),
            ("isolation", aggregate.isolation_mean),
            ("composite_strict (secondary)", aggregate.composite_strict_mean),
        ],
        columns=["feature", "monotonicity_rate"],
    )

    # Per-category
    cat_df = pd.DataFrame(
        [(c, r) for c, r in aggregate.per_category.items()],
        columns=["category", "composite_rate"],
    ).sort_values("category").reset_index(drop=True)

    # Per-problem
    prob_df = pd.DataFrame(
        [
            {
                "id": p.problem_id,
                "category": p.category,
                "n_steps": p.n_steps,
                "depth": p.depth_rate,
                "ops": p.ops_rate,
                "leaves": p.leaves_rate,
                "isolation": p.isolation_rate,
                "composite": p.composite_rate,
                "composite_strict": p.composite_strict_rate,
            }
            for p in aggregate.per_problem
        ]
    )

    # AC-variant σ
    if not variant_summary.empty:
        var_mean_sigma = variant_summary["sigma"].mean()
        var_max_sigma = variant_summary["sigma"].max()
        var_max_id = variant_summary.loc[variant_summary["sigma"].idxmax(), "problem_id"]
        fragility_flag = "FLAGGED" if var_max_sigma > 0.10 else "OK"
    else:
        var_mean_sigma = 0.0
        var_max_sigma = 0.0
        var_max_id = "(none)"
        fragility_flag = "OK"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# Phase 0 Report — Monotonicity Validation

**Generated:** {now}
**Pre-registration:** `phase0/PHASE0_PREREG.md` (committed before this report)
**Test set:** 20 problems, frozen in `phase0/problems/problems.yaml`

---

## Headline (per pre-registration §6)

**Primary metric:** step-level monotonicity rate of the composite feature, mean across 20 problems.

**Headline `H` = `{headline:.4f}`**

**Decision:** `{label}`

> {decision_msg}

---

## Per-feature monotonicity rates (mean across problems)

{_table_md(feat_df)}

---

## Per-category breakdown (composite rate)

{_table_md(cat_df)}

---

## AC-variant fragility (σ across original + 3 variants)

| Stat | Value |
|---|---|
| mean σ across 20 problems | {var_mean_sigma:.4f} |
| max σ | {var_max_sigma:.4f} |
| max σ problem | {var_max_id} |
| fragility flag (σ_max > 0.10) | {fragility_flag} |

Per-problem σ table:

{_table_md(variant_summary) if not variant_summary.empty else "_(no variants computed)_"}

---

## Per-problem detail

{_table_md(prob_df)}

---

## Limitations (acknowledged at pre-registration)

- Hand-curated traces are not provably BFS-optimal (textbook ≠ BFS-optimal); bias direction is *toward* monotonicity, so a fail is still a real fail.
- 20 problems is a small sample; no bootstrap CI is reported (would be unjustified at n=20).
- Feature definitions are arbitrary. All 4 features and the composite are reported separately to mitigate cherry-picking.
- Test-set selection is textbook-typical, not adversarial. Phase 1+ generalization splits compensate.

---

## Reproduction

```powershell
& .\\.venv\\Scripts\\python.exe -m phase0.src.run_phase0
```

Long-form step-level data is in `{traces_csv}`. Open `phase0/notebooks/phase0_analysis.ipynb` for visual inspection.
"""

    Path(output_path).write_text(md, encoding="utf-8")
