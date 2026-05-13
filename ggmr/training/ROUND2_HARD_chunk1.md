# Phase 2 Evaluation Results

Checkpoint: `checkpoints/round2/best.pt`


## Aggregate

- Total problems: 13
- Joint solved (hand AND learned): 13
- Hand-only solved: 0  (regression detector)
- Learned-only solved: 0  (new problems opened)
- Both failed: 0
- **Geomean compression (joint)**: 7.847x
- Median compression (joint): 4.000x

## Per-family geomean (joint-solved subset)

| family | geomean |
|---|---|
| L1 | 2.828x |
| L3 | 20.281x |
| P3 | 75.888x |
| P4 | 12.006x |
| R1 | 2.699x |
| R2 | 2.250x |
| v1_ex1 | 3.092x |

## Per-problem

| id | family | hand | hand_nodes | learned | learned_nodes | ratio |
|---|---|---|---|---|---|---|
| hard_motif_v1_ex1_000 | v1_ex1 | Y | 76 | Y | 36 | 2.11x |
| hard_motif_L1_000 | L1 | Y | 16 | Y | 4 | 4.00x |
| hard_motif_L3_000 | L3 | Y | 914 | Y | 4 | 228.50x |
| hard_motif_P3_000 | P3 | Y | 886 | Y | 2 | 443.00x |
| hard_motif_P4_000 | P4 | Y | 257 | Y | 20 | 12.85x |
| hard_motif_R1_000 | R1 | Y | 9 | Y | 3 | 3.00x |
| hard_motif_R2_000 | R2 | Y | 9 | Y | 4 | 2.25x |
| hard_motif_v1_ex1_001 | v1_ex1 | Y | 163 | Y | 36 | 4.53x |
| hard_motif_L1_001 | L1 | Y | 10 | Y | 5 | 2.00x |
| hard_motif_L3_001 | L3 | Y | 9 | Y | 5 | 1.80x |
| hard_motif_P3_001 | P3 | Y | 26 | Y | 2 | 13.00x |
| hard_motif_P4_001 | P4 | Y | 258 | Y | 23 | 11.22x |
| hard_motif_R1_001 | R1 | Y | 17 | Y | 7 | 2.43x |