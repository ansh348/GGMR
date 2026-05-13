# Phase 2 Evaluation Results

Checkpoint: `checkpoints/full/best.pt`


## Aggregate

- Total problems: 38
- Joint solved (hand AND learned): 37
- Hand-only solved: 1  (regression detector)
- Learned-only solved: 0  (new problems opened)
- Both failed: 0
- **Geomean compression (joint)**: 2.334x
- Median compression (joint): 2.000x
- Regression IDs: qua04

## Per-family geomean (joint-solved subset)

| family | geomean |
|---|---|
| L1 | 17.487x |
| L3 | 26.100x |
| P3 | 12.662x |
| P4 | 2.399x |
| R1 | 3.634x |
| R2 | 2.565x |
| linear | 1.201x |
| polynomial | 1.000x |
| quadratic | 1.328x |
| rational | 0.833x |
| v1_ex1 | 3.303x |

## Per-problem

| id | family | hand | hand_nodes | learned | learned_nodes | ratio |
|---|---|---|---|---|---|---|
| hard_motif_P4_004 | P4 | Y | 520 | Y | 39 | 13.33x |
| hard_motif_R1_004 | R1 | Y | 9 | Y | 3 | 3.00x |
| hard_motif_R2_004 | R2 | Y | 9 | Y | 4 | 2.25x |
| hard_motif_v1_ex1_005 | v1_ex1 | Y | 175 | Y | 56 | 3.12x |
| hard_motif_L1_005 | L1 | Y | 16 | Y | 3 | 5.33x |
| hard_motif_L3_005 | L3 | Y | 1211 | Y | 4 | 302.75x |
| hard_motif_P3_005 | P3 | Y | 26 | Y | 2 | 13.00x |
| hard_motif_P4_005 | P4 | Y | 159 | Y | 43 | 3.70x |
| hard_motif_R1_005 | R1 | Y | 16 | Y | 3 | 5.33x |
| hard_motif_R2_005 | R2 | Y | 9 | Y | 3 | 3.00x |
| hard_motif_v1_ex1_006 | v1_ex1 | Y | 236 | Y | 48 | 4.92x |
| hard_motif_L1_006 | L1 | Y | 172 | Y | 3 | 57.33x |
| hard_motif_L3_006 | L3 | Y | 9 | Y | 4 | 2.25x |
| hard_motif_P3_006 | P3 | Y | 37 | Y | 3 | 12.33x |
| hard_motif_P4_006 | P4 | Y | 14 | Y | 50 | 0.28x |
| hard_motif_R1_006 | R1 | Y | 9 | Y | 3 | 3.00x |
| hard_motif_R2_006 | R2 | Y | 10 | Y | 4 | 2.50x |
| hard_motif_v1_ex1_007 | v1_ex1 | Y | 136 | Y | 58 | 2.34x |
| lin01 | linear | Y | 1 | Y | 1 | 1.00x |
| lin02 | linear | Y | 5 | Y | 4 | 1.25x |
| lin03 | linear | Y | 1 | Y | 1 | 1.00x |
| lin04 | linear | Y | 6 | Y | 3 | 2.00x |
| lin05 | linear | Y | 1 | Y | 1 | 1.00x |
| qua01 | quadratic | Y | 1 | Y | 1 | 1.00x |
| qua02 | quadratic | Y | 1 | Y | 1 | 1.00x |
| qua03 | quadratic | Y | 17 | Y | 13 | 1.31x |
| qua04 | quadratic | Y | 2 | N | 50000 | - |
| qua05 | quadratic | Y | 19 | Y | 8 | 2.38x |
| rat01 | rational | Y | 2 | Y | 2 | 1.00x |
| rat02 | rational | Y | 3 | Y | 6 | 0.50x |
| rat03 | rational | Y | 4 | Y | 10 | 0.40x |
| rat04 | rational | Y | 2 | Y | 2 | 1.00x |
| rat05 | rational | Y | 8 | Y | 4 | 2.00x |
| poly01 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly02 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly03 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly04 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly05 | polynomial | Y | 1 | Y | 1 | 1.00x |