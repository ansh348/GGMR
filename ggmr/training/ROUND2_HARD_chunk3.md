# Phase 2 Evaluation Results

Checkpoint: `checkpoints/round2/best.pt`


## Aggregate

- Total problems: 12
- Joint solved (hand AND learned): 12
- Hand-only solved: 0  (regression detector)
- Learned-only solved: 0  (new problems opened)
- Both failed: 0
- **Geomean compression (joint)**: 9.678x
- Median compression (joint): 4.431x

## Per-family geomean (joint-solved subset)

| family | geomean |
|---|---|
| L1 | 40.608x |
| L3 | 27.511x |
| P3 | 19.000x |
| P4 | 22.609x |
| R1 | 1.643x |
| R2 | 1.591x |
| v1_ex1 | 13.576x |

## Per-problem

| id | family | hand | hand_nodes | learned | learned_nodes | ratio |
|---|---|---|---|---|---|---|
| hard_motif_R1_003 | R1 | Y | 9 | Y | 5 | 1.80x |
| hard_motif_R2_003 | R2 | Y | 9 | Y | 4 | 2.25x |
| hard_motif_v1_ex1_004 | v1_ex1 | Y | 1365 | Y | 36 | 37.92x |
| hard_motif_L1_004 | L1 | Y | 1649 | Y | 4 | 412.25x |
| hard_motif_L3_004 | L3 | Y | 10 | Y | 4 | 2.50x |
| hard_motif_P3_004 | P3 | Y | 38 | Y | 2 | 19.00x |
| hard_motif_P4_004 | P4 | Y | 520 | Y | 23 | 22.61x |
| hard_motif_R1_004 | R1 | Y | 9 | Y | 6 | 1.50x |
| hard_motif_R2_004 | R2 | Y | 9 | Y | 8 | 1.12x |
| hard_motif_v1_ex1_005 | v1_ex1 | Y | 175 | Y | 36 | 4.86x |
| hard_motif_L1_005 | L1 | Y | 16 | Y | 4 | 4.00x |
| hard_motif_L3_005 | L3 | Y | 1211 | Y | 4 | 302.75x |