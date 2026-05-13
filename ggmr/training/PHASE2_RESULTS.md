# Phase 2 Evaluation Results (FULL)

Checkpoint: `checkpoints/full/best.pt`
Train data: 23,403 rows. GPU: RTX 3050. Eval: 50 hard_v2 + 20 phase0.

## Aggregate (all 70)

- Total problems: 70
- Joint solved (hand AND learned): **69/70**
- Hand-only solved: 1  (regressions: qua04)
- Learned-only solved: 0
- Both failed: 0
- **Geomean compression (joint)**: 4.255x
- Median compression (joint): 3.000x
- Max compression: 550x
- Min compression: 0.280x

## Aggregate (50 hard_v2 only)

- Joint solved: **50/50**
- **Geomean compression**: 7.211x
- Median: 4.148x
- Total hand nodes: 12465
- Total learned nodes: 878
- Total ratio: 14.20x

## Aggregate (20 phase0 only — regression set)

- Joint solved: **19/20**
- Regressions: ['qua04']
- Geomean: 1.061x
- Median: 1.000x

## Per-family geomean (joint-solved)

| family | n | geomean | median | min | max |
|---|---|---|---|---|---|
| L1 | 7 | 22.256x | 5.333x | 3.333x | 550x |
| L3 | 7 | 10.048x | 2.500x | 2.250x | 303x |
| P3 | 7 | 18.205x | 13.000x | 5.667x | 295x |
| P4 | 7 | 3.939x | 5.733x | 0.280x | 16x |
| R1 | 7 | 3.567x | 3.000x | 3.000x | 6x |
| R2 | 7 | 5.184x | 2.250x | 2.250x | 524x |
| linear | 5 | 1.201x | 1.000x | 1.000x | 2x |
| polynomial | 5 | 1.000x | 1.000x | 1.000x | 1x |
| quadratic | 4 | 1.328x | 1.154x | 1.000x | 2x |
| rational | 5 | 0.833x | 1.000x | 0.400x | 2x |
| v1_ex1 | 8 | 3.754x | 3.054x | 1.382x | 22x |

## Per-problem (sorted by compression ratio, descending)

| id | family | source | hand | hand_nodes | learned | learned_nodes | ratio |
|---|---|---|---|---|---|---|---|
| hard_motif_L1_004 | L1 | hard | Y | 1649 | Y | 3 | 549.67x |
| hard_motif_R2_001 | R2 | hard | Y | 2094 | Y | 4 | 523.50x |
| hard_motif_L3_005 | L3 | hard | Y | 1211 | Y | 4 | 302.75x |
| hard_motif_P3_000 | P3 | hard | Y | 886 | Y | 3 | 295.33x |
| hard_motif_L3_000 | L3 | hard | Y | 914 | Y | 4 | 228.50x |
| hard_motif_L1_002 | L1 | hard | Y | 679 | Y | 3 | 226.33x |
| hard_motif_L1_006 | L1 | hard | Y | 172 | Y | 3 | 57.33x |
| hard_motif_v1_ex1_004 | v1_ex1 | hard | Y | 1365 | Y | 62 | 22.02x |
| hard_motif_P4_002 | P4 | hard | Y | 530 | Y | 34 | 15.59x |
| hard_motif_P3_003 | P3 | hard | Y | 45 | Y | 3 | 15.00x |
| hard_motif_P4_004 | P4 | hard | Y | 520 | Y | 39 | 13.33x |
| hard_motif_P3_001 | P3 | hard | Y | 26 | Y | 2 | 13.00x |
| hard_motif_P3_005 | P3 | hard | Y | 26 | Y | 2 | 13.00x |
| hard_motif_P3_004 | P3 | hard | Y | 38 | Y | 3 | 12.67x |
| hard_motif_P3_006 | P3 | hard | Y | 37 | Y | 3 | 12.33x |
| hard_motif_P4_000 | P4 | hard | Y | 257 | Y | 40 | 6.42x |
| hard_motif_P4_001 | P4 | hard | Y | 258 | Y | 45 | 5.73x |
| hard_motif_R1_001 | R1 | hard | Y | 17 | Y | 3 | 5.67x |
| hard_motif_P3_002 | P3 | hard | Y | 17 | Y | 3 | 5.67x |
| hard_motif_L1_000 | L1 | hard | Y | 16 | Y | 3 | 5.33x |
| hard_motif_L1_005 | L1 | hard | Y | 16 | Y | 3 | 5.33x |
| hard_motif_R1_005 | R1 | hard | Y | 16 | Y | 3 | 5.33x |
| hard_motif_L3_003 | L3 | hard | Y | 21 | Y | 4 | 5.25x |
| hard_motif_v1_ex1_006 | v1_ex1 | hard | Y | 236 | Y | 48 | 4.92x |
| hard_motif_v1_ex1_002 | v1_ex1 | hard | Y | 262 | Y | 61 | 4.30x |
| hard_motif_L1_003 | L1 | hard | Y | 12 | Y | 3 | 4.00x |
| hard_motif_P4_005 | P4 | hard | Y | 159 | Y | 43 | 3.70x |
| hard_motif_L1_001 | L1 | hard | Y | 10 | Y | 3 | 3.33x |
| hard_motif_v1_ex1_005 | v1_ex1 | hard | Y | 175 | Y | 56 | 3.12x |
| hard_motif_R1_000 | R1 | hard | Y | 9 | Y | 3 | 3.00x |
| hard_motif_R1_002 | R1 | hard | Y | 9 | Y | 3 | 3.00x |
| hard_motif_R1_003 | R1 | hard | Y | 9 | Y | 3 | 3.00x |
| hard_motif_R1_004 | R1 | hard | Y | 9 | Y | 3 | 3.00x |
| hard_motif_R2_005 | R2 | hard | Y | 9 | Y | 3 | 3.00x |
| hard_motif_R1_006 | R1 | hard | Y | 9 | Y | 3 | 3.00x |
| hard_motif_v1_ex1_003 | v1_ex1 | hard | Y | 173 | Y | 58 | 2.98x |
| hard_motif_v1_ex1_001 | v1_ex1 | hard | Y | 163 | Y | 58 | 2.81x |
| hard_motif_L3_004 | L3 | hard | Y | 10 | Y | 4 | 2.50x |
| hard_motif_R2_006 | R2 | hard | Y | 10 | Y | 4 | 2.50x |
| qua05 | quadratic | phase0 | Y | 19 | Y | 8 | 2.38x |
| hard_motif_v1_ex1_007 | v1_ex1 | hard | Y | 136 | Y | 58 | 2.34x |
| hard_motif_R2_000 | R2 | hard | Y | 9 | Y | 4 | 2.25x |
| hard_motif_L3_001 | L3 | hard | Y | 9 | Y | 4 | 2.25x |
| hard_motif_L3_002 | L3 | hard | Y | 9 | Y | 4 | 2.25x |
| hard_motif_R2_002 | R2 | hard | Y | 9 | Y | 4 | 2.25x |
| hard_motif_R2_003 | R2 | hard | Y | 9 | Y | 4 | 2.25x |
| hard_motif_R2_004 | R2 | hard | Y | 9 | Y | 4 | 2.25x |
| hard_motif_L3_006 | L3 | hard | Y | 9 | Y | 4 | 2.25x |
| lin04 | linear | phase0 | Y | 6 | Y | 3 | 2.00x |
| rat05 | rational | phase0 | Y | 8 | Y | 4 | 2.00x |
| hard_motif_P4_003 | P4 | hard | Y | 102 | Y | 55 | 1.85x |
| hard_motif_v1_ex1_000 | v1_ex1 | hard | Y | 76 | Y | 55 | 1.38x |
| qua03 | quadratic | phase0 | Y | 17 | Y | 13 | 1.31x |
| lin02 | linear | phase0 | Y | 5 | Y | 4 | 1.25x |
| lin01 | linear | phase0 | Y | 1 | Y | 1 | 1.00x |
| lin03 | linear | phase0 | Y | 1 | Y | 1 | 1.00x |
| lin05 | linear | phase0 | Y | 1 | Y | 1 | 1.00x |
| qua01 | quadratic | phase0 | Y | 1 | Y | 1 | 1.00x |
| qua02 | quadratic | phase0 | Y | 1 | Y | 1 | 1.00x |
| rat01 | rational | phase0 | Y | 2 | Y | 2 | 1.00x |
| rat04 | rational | phase0 | Y | 2 | Y | 2 | 1.00x |
| poly01 | polynomial | phase0 | Y | 1 | Y | 1 | 1.00x |
| poly02 | polynomial | phase0 | Y | 1 | Y | 1 | 1.00x |
| poly03 | polynomial | phase0 | Y | 1 | Y | 1 | 1.00x |
| poly04 | polynomial | phase0 | Y | 1 | Y | 1 | 1.00x |
| poly05 | polynomial | phase0 | Y | 1 | Y | 1 | 1.00x |
| rat02 | rational | phase0 | Y | 3 | Y | 6 | 0.50x |
| rat03 | rational | phase0 | Y | 4 | Y | 10 | 0.40x |
| hard_motif_P4_006 | P4 | hard | Y | 14 | Y | 50 | 0.28x |
| qua04 | quadratic | phase0 | Y | 2 | N | 50000 | - |