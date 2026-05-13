# Phase 2 Evaluation Results

Checkpoint: `checkpoints/round2/best.pt`


## Aggregate

- Total problems: 12
- Joint solved (hand AND learned): 12
- Hand-only solved: 0  (regression detector)
- Learned-only solved: 0  (new problems opened)
- Both failed: 0
- **Geomean compression (joint)**: 4.340x
- Median compression (joint): 3.489x

## Per-family geomean (joint-solved subset)

| family | geomean |
|---|---|
| L1 | 34.400x |
| L3 | 2.250x |
| P3 | 15.508x |
| P4 | 2.102x |
| R1 | 2.400x |
| R2 | 1.897x |
| v1_ex1 | 5.121x |

## Per-problem

| id | family | hand | hand_nodes | learned | learned_nodes | ratio |
|---|---|---|---|---|---|---|
| hard_motif_P3_005 | P3 | Y | 26 | Y | 2 | 13.00x |
| hard_motif_P4_005 | P4 | Y | 159 | Y | 24 | 6.62x |
| hard_motif_R1_005 | R1 | Y | 16 | Y | 5 | 3.20x |
| hard_motif_R2_005 | R2 | Y | 9 | Y | 5 | 1.80x |
| hard_motif_v1_ex1_006 | v1_ex1 | Y | 236 | Y | 34 | 6.94x |
| hard_motif_L1_006 | L1 | Y | 172 | Y | 5 | 34.40x |
| hard_motif_L3_006 | L3 | Y | 9 | Y | 4 | 2.25x |
| hard_motif_P3_006 | P3 | Y | 37 | Y | 2 | 18.50x |
| hard_motif_P4_006 | P4 | Y | 14 | Y | 21 | 0.67x |
| hard_motif_R1_006 | R1 | Y | 9 | Y | 5 | 1.80x |
| hard_motif_R2_006 | R2 | Y | 10 | Y | 5 | 2.00x |
| hard_motif_v1_ex1_007 | v1_ex1 | Y | 136 | Y | 36 | 3.78x |