# Phase 0 — Prerequisite Validation for GGMR

Per `ggmr_v9.pdf` §5.6 and §2.1 Layer 1.

**Purpose:** Determine whether structural complexity decreases monotonically along near-optimal solution paths for canonical algebraic rewrite problems. Outcome dictates the entire paper's framing.

## Result (this run)

**Headline `H` = 0.6417** → `PHASE_A_WEAK_NONMYOPIC_PRIMARY`
> Phase A is weak. Paper restructures to emphasize non-myopic learned value (Phase B/C) as primary contribution; §4.1 becomes a foil.

AC-variant fragility flag: **TRIGGERED** (max σ = 0.50 on `qua01`).

See `outputs/PHASE0_REPORT.md` for the full auto-generated report and `PHASE0_FINDINGS.md` for hand-annotated observations.

## File layout

```
phase0/
  PHASE0_PREREG.md            # PRE-REGISTRATION (committed before any features computed)
  PHASE0_FINDINGS.md          # manual annotations from notebook inspection
  README.md                   # this file
  requirements.txt
  problems/
    problems.yaml             # 20 hand-curated problem traces (5 each: linear/quad/rat/poly)
  src/
    features.py               # 4 features: depth, ops, leaves, var-isolation; z-scored composite
    trace_loader.py           # YAML → typed Problem dataclasses (parse_expr evaluate=False)
    verifier.py               # step-legality via solution-set equality
    variants.py               # AC-equivalent variant generator (Add perm / Mul perm / rename)
    monotonicity.py           # step_rate primitive + per-feature, per-category aggregation
    report.py                 # CSV → PHASE0_REPORT.md auto-generation
    run_phase0.py             # orchestrator
  outputs/
    traces.csv                # long-format step-level data (one row per state)
    monotonicity.csv          # per-problem rates (4 features + composite + strict)
    variants_summary.csv      # per-problem σ across 3 AC variants
    variants_traces.csv       # full feature data for AC variants
    PHASE0_REPORT.md          # auto-generated, do not hand-edit (will be overwritten)
    fig_*.png                 # 4 figures rendered by the notebook
  notebooks/
    phase0_analysis.ipynb     # visual inspection (trajectories, histograms, heatmap)
  tests/
    test_features.py
    test_verifier.py
    test_monotonicity.py
    test_variants.py
```

## Reproduction

From the project root (`MonumentalLeapForward/`):

```powershell
# 1. Install dependencies (one-time)
& .\.venv\Scripts\pip.exe install -r phase0\requirements.txt

# 2. Run all tests
$env:PYTHONIOENCODING='utf-8'
& .\.venv\Scripts\python.exe -m pytest phase0\tests\ -v

# 3. Run the experiment end-to-end
& .\.venv\Scripts\python.exe -m phase0.src.run_phase0

# 4. Render the analysis notebook
& .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute `
    phase0\notebooks\phase0_analysis.ipynb --output phase0_analysis.ipynb
```

Outputs land in `phase0/outputs/`. The notebook also writes 4 PNGs there.

## Pre-registration

Decision thresholds (from `PHASE0_PREREG.md` §6, committed before this run):

| Headline `H` | Decision | Action |
|---|---|---|
| `H ≥ 0.80` | `PHASE_A_MEANINGFUL_BASELINE` | Continue paper as planned |
| `0.50 ≤ H < 0.80` | `PHASE_A_WEAK_NONMYOPIC_PRIMARY` | Restructure: non-myopic value as primary contribution |
| `H < 0.50` | `PHASE_A_NEGATIVE_RESULT` | Restructure: paper is the negative result itself |

Tie-breaker: borderline values within ±0.02 of a threshold round AGAINST the stronger claim (conservative wins).

## Scope

This is the **Lean+** Phase 0 per the approved plan: hand-curated problem traces with programmatic step-legality verification and AC-variant fragility check. **No full ~57-rule rewrite engine, no BFS, no e-graph**. Those are Phase 1 work per `ggmr_v9.pdf` §5.8 timeline.

Rule names in `problems.yaml` are metadata only; this Phase 0 does not implement them as rewrite rules.
