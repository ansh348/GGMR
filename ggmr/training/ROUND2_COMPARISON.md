# Round 1 vs Round 2 Comparison

**Round 1 model**: `checkpoints/full/best.pt` — trained on 23,403 pairs (Phase 1b dataset).
**Round 2 model**: `checkpoints/round2/best.pt` — trained on 89,499 pairs across 34 categories spanning the full difficulty spectrum (per `ggmr/problems/round2_categories.py`). After dedup: 27,346 unique training rows.

Training time: 220.6s on RTX 3050 Laptop GPU, early-stopped at epoch 45 (patience=15).
Validation metrics: val_mae_step=0.093, val_pearson_log=0.991, val_spearman_log=0.899.

## Headline table

| Metric                       | Round 1 (22k)        | Round 2 (89k)        | Verdict |
|------------------------------|----------------------|----------------------|---------|
| Hard set geomean             | 7.21x                | **7.69x**            | ✅ PASS (≥5×) — improved |
| Hard set peak                | 549.67x (L1_004)     | **698.00x (R2_001)** | ✅ improved 27% |
| Hard set median              | 4.15x                | **4.67x**            | improved |
| Hard set solved              | 50/50                | **50/50**            | match |
| Hard set regressions (<1.0×) | 1 (P4_006 at 0.28×)  | **1 (P4_006 at 0.67×)** | improved magnitude |
| Phase 0 solved               | 19/20 (qua04 lost)   | **20/20**            | ✅ PASS — fixed |
| Phase 0 geomean              | 1.061x               | **1.083x**           | slight improvement |
| OOD external solved          | 14/14 (joint)        | **14/14**            | match |
| OOD external geomean         | 0.76x ¹              | **0.89x**            | ❌ FAIL (<1.0×) — improved 17% but criterion missed |
| OOD external regressions     | 5/13 ¹               | **5/14**             | ❌ FAIL (same count) |
| L2 timeout (cross-recip)     | 3 nodes ¹            | **3 nodes**          | ✅ PASS — match |
| R3 timeout (cross-recip-frac)| 88 nodes ¹           | **14 nodes**         | ✅ PASS — 6.3× better |

¹ Round 1 OOD/L2/R3 numbers are from the user spec (not persisted on disk for Round 1). Closest on-disk reference: `ggmr/problems/round2_categories.py:15` cites "0.755× OOD geomean" for Round 1.

## Success criteria scorecard

| # | Criterion                            | Status |
|---|--------------------------------------|--------|
| 1 | Hard set geomean ≥ 5×                | ✅ PASS (7.69×) |
| 2 | OOD geomean ≥ 1.0× (was 0.76×)       | ❌ FAIL (0.89×) |
| 3 | OOD regressions ≤ 1 (was 5)          | ❌ FAIL (5) |
| 4 | Phase 0: 20/20 solved                | ✅ PASS |
| 5 | L2 and R3 still solved               | ✅ PASS (R3 6.3× better) |

**Verdict: 3 of 5 criteria pass.** Hard-set / Phase 0 / timeouts all hit. The Round 2 dataset broadening fixed Phase 0 (qua04 now solved) and improved OOD by 17%, but did not fully solve the "over-engineering easy problems" issue.

## Hard set per-family deltas

| family | n | R1 geomean | R2 geomean | delta |
|---|---|---|---|---|
| L1 | 7 | 22.26x | 16.32x | -27% |
| L3 | 7 | 10.05x | 9.73x | -3% |
| P3 | 7 | 18.21x | 24.32x | +34% |
| P4 | 7 | 3.94x | 7.67x | +95% (was weakest) |
| R1 | 7 | 3.57x | 2.30x | -36% |
| R2 | 7 | 5.18x | 4.07x | -21% |
| v1_ex1 | 8 | 3.75x | 5.97x | +59% |

P4 (was R1's weakest hard family) nearly doubled. P3 and v1_ex1 also improved. R1/R2 families regressed; L1/L3 nearly held. Net hard geomean improved because P4 lifted the floor more than R1/R2 lowered it.

## OOD per-problem deltas

Round 1 OOD numbers were not persisted, so deltas show only Round 2 values. The five Round 2 regressions (ratio<1.0):

| id | R2 hand | R2 learned | R2 ratio |
|---|---|---|---|
| math_01 | 5 | 20 | 0.25x |
| math_05 | 6 | 19 | 0.32x |
| amc_05 | 2 | 5 | 0.40x |
| text_04 | 2 | 4 | 0.50x |
| text_05 | 2 | 14 | 0.14x |

All five are "simple problems where hand A* finishes in 2-6 nodes" — exactly the pattern Round 2 was supposed to fix. The GIN still slightly over-engineers them (predicts >1 step needed when hand picks the optimal rule first). Magnitude is small (5-20 learned-nodes), but enough to hurt the geomean.

## What worked

- **Phase 0 fix**: qua04 (irreducible quadratic where R1 timed out at 50k nodes) now solves in 6 nodes — exact same root cause that produced R1's only Phase 0 regression.
- **R3 timeout**: 88 → 14 nodes (6.3× better). Validates the broader rational/fractional coverage.
- **Hard set peak**: 549× → 698×. The R2_001 cross-reciprocal motif compresses better than R1's L1_004.
- **P4 family**: nearly doubled (3.94× → 7.67×). Round 2's expansion of quad/poly categories helped.

## What didn't work

- **OOD over-engineering persists**: Round 2 reduced severity but not count. The five regressing problems are all "single-step easy" — the GIN still predicts non-trivial remaining steps when the optimal action is immediate.
- **R1/R2 hard families regressed** (-36% / -21%): broader training distribution diluted the strong reverse-easy reciprocal signal that R1 had concentrated on.

## Files

- `ggmr/training/ROUND2_HARD_RESULTS.md`, `.csv` — merged 50-problem hard set
- `ggmr/training/ROUND2_HARD_chunk1..4.md`, `.csv` — per-chunk outputs (kept for audit)
- `ggmr/training/ROUND2_PHASE0_RESULTS.md`, `.csv` — phase 0
- `ggmr/training/EXTERNAL_RESULTS.csv`, `round2_external_results.log` — OOD external
- `round2_timeout_results.log` — L2/R3
- `round2_training.log` — training log
- `checkpoints/round2/best.pt`, `history.json` — model
