# Phase 2 Evaluation Results

Checkpoint: `checkpoints/round2/best.pt`


## Aggregate

- Total problems: 13
- Joint solved (hand AND learned): 13
- Hand-only solved: 0  (regression detector)
- Learned-only solved: 0  (new problems opened)
- Both failed: 0
- **Geomean compression (joint)**: 10.349x
- Median compression (joint): 5.250x

## Per-family geomean (joint-solved subset)

| family | geomean |
|---|---|
| L1 | 26.058x |
| L3 | 3.437x |
| P3 | 13.829x |
| P4 | 10.398x |
| R1 | 3.000x |
| R2 | 29.957x |
| v1_ex1 | 5.914x |

## Per-problem

| id | family | hand | hand_nodes | learned | learned_nodes | ratio |
|---|---|---|---|---|---|---|
| hard_motif_R2_001 | R2 | Y | 2094 | Y | 3 | 698.00x |
| hard_motif_v1_ex1_002 | v1_ex1 | Y | 262 | Y | 36 | 7.28x |
| hard_motif_L1_002 | L1 | Y | 679 | Y | 3 | 226.33x |
| hard_motif_L3_002 | L3 | Y | 9 | Y | 4 | 2.25x |
| hard_motif_P3_002 | P3 | Y | 17 | Y | 2 | 8.50x |
| hard_motif_P4_002 | P4 | Y | 530 | Y | 25 | 21.20x |
| hard_motif_R1_002 | R1 | Y | 9 | Y | 3 | 3.00x |
| hard_motif_R2_002 | R2 | Y | 9 | Y | 7 | 1.29x |
| hard_motif_v1_ex1_003 | v1_ex1 | Y | 173 | Y | 36 | 4.81x |
| hard_motif_L1_003 | L1 | Y | 12 | Y | 4 | 3.00x |
| hard_motif_L3_003 | L3 | Y | 21 | Y | 4 | 5.25x |
| hard_motif_P3_003 | P3 | Y | 45 | Y | 2 | 22.50x |
| hard_motif_P4_003 | P4 | Y | 102 | Y | 20 | 5.10x |