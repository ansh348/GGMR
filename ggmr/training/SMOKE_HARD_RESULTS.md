# Phase 2 Evaluation Results

Checkpoint: `checkpoints/smoke/best.pt`


## Aggregate

- Total problems: 8
- Joint solved (hand AND learned): 8
- Hand-only solved: 0  (regression detector)
- Learned-only solved: 0  (new problems opened)
- Both failed: 0
- **Geomean compression (joint)**: 17.116x
- Median compression (joint): 8.528x

## Per-family geomean (joint-solved subset)

| family | geomean |
|---|---|
| L1 | 8.000x |
| L3 | 457.000x |
| P3 | 443.000x |
| P4 | 10.280x |
| R1 | 4.500x |
| R2 | 3.000x |
| v1_ex1 | 5.725x |

## Per-problem

| id | family | hand | hand_nodes | learned | learned_nodes | ratio |
|---|---|---|---|---|---|---|
| hard_motif_v1_ex1_000 | v1_ex1 | Y | 76 | Y | 21 | 3.62x |
| hard_motif_L1_000 | L1 | Y | 16 | Y | 2 | 8.00x |
| hard_motif_L3_000 | L3 | Y | 914 | Y | 2 | 457.00x |
| hard_motif_P3_000 | P3 | Y | 886 | Y | 2 | 443.00x |
| hard_motif_P4_000 | P4 | Y | 257 | Y | 25 | 10.28x |
| hard_motif_R1_000 | R1 | Y | 9 | Y | 2 | 4.50x |
| hard_motif_R2_000 | R2 | Y | 9 | Y | 3 | 3.00x |
| hard_motif_v1_ex1_001 | v1_ex1 | Y | 163 | Y | 18 | 9.06x |