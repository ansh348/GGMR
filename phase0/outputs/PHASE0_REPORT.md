# Phase 0 Report — Monotonicity Validation

**Generated:** 2026-05-10 17:06:35
**Pre-registration:** `phase0/PHASE0_PREREG.md` (committed before this report)
**Test set:** 20 problems, frozen in `phase0/problems/problems.yaml`

---

## Headline (per pre-registration §6)

**Primary metric:** step-level monotonicity rate of the composite feature, mean across 20 problems.

**Headline `H` = `0.6417`**

**Decision:** `PHASE_A_WEAK_NONMYOPIC_PRIMARY`

> Phase A is weak. Paper restructures to emphasize non-myopic learned value (Phase B/C) as primary contribution; §4.1 becomes a foil.

---

## Per-feature monotonicity rates (mean across problems)

| feature | monotonicity_rate |
|---|---|
| composite (primary) | 0.642 |
| depth | 0.863 |
| ops | 0.825 |
| leaves | 0.817 |
| isolation | 0.858 |
| composite_strict (secondary) | 0.565 |


---

## Per-category breakdown (composite rate)

| category | composite_rate |
|---|---|
| linear | 1.000 |
| polynomial | 0.467 |
| quadratic | 0.233 |
| rational | 0.867 |


---

## AC-variant fragility (σ across original + 3 variants)

| Stat | Value |
|---|---|
| mean σ across 20 problems | 0.1167 |
| max σ | 0.5000 |
| max σ problem | qua01 |
| fragility flag (σ_max > 0.10) | FLAGGED |

Per-problem σ table:

| problem_id | original_rate | var1_addperm_rate | var2_mulperm_rate | var3_rename_rate | sigma |
|---|---|---|---|---|---|
| lin01 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| lin02 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| lin03 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| lin04 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| lin05 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| qua01 | 0.000 | 0.000 | 0.000 | 1.000 | 0.500 |
| qua02 | 0.000 | 0.000 | 0.000 | 1.000 | 0.500 |
| qua03 | 0.000 | 0.000 | 0.000 | 0.667 | 0.333 |
| qua04 | 0.667 | 0.667 | 0.667 | 1.000 | 0.167 |
| qua05 | 0.500 | 0.500 | 0.500 | 1.000 | 0.250 |
| rat01 | 0.667 | 0.667 | 0.667 | 1.000 | 0.167 |
| rat02 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| rat03 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| rat04 | 0.667 | 0.667 | 0.667 | 1.000 | 0.167 |
| rat05 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| poly01 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| poly02 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| poly03 | 0.333 | 0.333 | 0.333 | 0.333 | 0.000 |
| poly04 | 1.000 | 1.000 | 1.000 | 0.500 | 0.250 |
| poly05 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |


---

## Per-problem detail

| id | category | n_steps | depth | ops | leaves | isolation | composite | composite_strict |
|---|---|---|---|---|---|---|---|---|
| lin01 | linear | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| lin02 | linear | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| lin03 | linear | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| lin04 | linear | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 |
| lin05 | linear | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| qua01 | quadratic | 1 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| qua02 | quadratic | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| qua03 | quadratic | 3 | 0.667 | 0.667 | 0.333 | 0.667 | 0.000 | 0.000 |
| qua04 | quadratic | 3 | 0.667 | 0.667 | 1.000 | 1.000 | 0.667 | 0.333 |
| qua05 | quadratic | 4 | 0.750 | 0.500 | 1.000 | 1.000 | 0.500 | 0.500 |
| rat01 | rational | 3 | 1.000 | 1.000 | 0.667 | 0.667 | 0.667 | 0.667 |
| rat02 | rational | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 |
| rat03 | rational | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| rat04 | rational | 3 | 1.000 | 1.000 | 0.667 | 0.667 | 0.667 | 0.667 |
| rat05 | rational | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 |
| poly01 | polynomial | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| poly02 | polynomial | 2 | 1.000 | 0.500 | 0.000 | 0.500 | 0.000 | 0.000 |
| poly03 | polynomial | 3 | 0.667 | 0.667 | 0.667 | 0.667 | 0.333 | 0.333 |
| poly04 | polynomial | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| poly05 | polynomial | 2 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |


---

## Limitations (acknowledged at pre-registration)

- Hand-curated traces are not provably BFS-optimal (textbook ≠ BFS-optimal); bias direction is *toward* monotonicity, so a fail is still a real fail.
- 20 problems is a small sample; no bootstrap CI is reported (would be unjustified at n=20).
- Feature definitions are arbitrary. All 4 features and the composite are reported separately to mitigate cherry-picking.
- Test-set selection is textbook-typical, not adversarial. Phase 1+ generalization splits compensate.

---

## Reproduction

```powershell
& .\.venv\Scripts\python.exe -m phase0.src.run_phase0
```

Long-form step-level data is in `C:\Users\anshu\PycharmProjects\MonumentalLeapForward\phase0\outputs\traces.csv`. Open `phase0/notebooks/phase0_analysis.ipynb` for visual inspection.
