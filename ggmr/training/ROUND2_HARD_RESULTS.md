# Round 2 Hard Set Evaluation Results

Checkpoint: `checkpoints/round2/best.pt`
Train data: 89,499 rows (27,346 unique after dedup). GPU: trained on CUDA, eval on CPU (4-chunk parallel).
Eval: 50 hard_v2 problems, max_nodes=50000, max_depth=25.

## Aggregate

- Total problems: 50
- Joint solved (hand AND learned): **50/50**
- Hand-only solved: 0  (regression detector)
- Learned-only solved: 0  (new problems opened)
- Both failed: 0
- **Geomean compression (joint)**: 7.693x
- Median compression (joint): 4.667x
- Peak compression: 698.00x
- Worst compression: 0.667x
- Joint regressions (ratio<1.0): 1  ids: ['hard_motif_P4_006']

## Per-family geomean (joint-solved subset)

| family | n | geomean | median | min | max |
|---|---|---|---|---|---|
| L1 | 7 | 16.318x | 4.000x | 2.000x | 412.250x |
| L3 | 7 | 9.733x | 2.500x | 1.800x | 302.750x |
| P3 | 7 | 24.321x | 18.500x | 8.500x | 443.000x |
| P4 | 7 | 7.666x | 11.217x | 0.667x | 22.609x |
| R1 | 7 | 2.299x | 2.429x | 1.500x | 3.200x |
| R2 | 7 | 4.067x | 2.000x | 1.125x | 698.000x |
| v1_ex1 | 8 | 5.971x | 4.833x | 2.111x | 37.917x |

## Per-problem (sorted by compression ratio, descending)

| id | family | hand | hand_nodes | learned | learned_nodes | ratio |
|---|---|---|---|---|---|---|
| hard_motif_R2_001 | R2 | Y | 2094 | Y | 3 | 698.00x |
| hard_motif_P3_000 | P3 | Y | 886 | Y | 2 | 443.00x |
| hard_motif_L1_004 | L1 | Y | 1649 | Y | 4 | 412.25x |
| hard_motif_L3_005 | L3 | Y | 1211 | Y | 4 | 302.75x |
| hard_motif_L3_000 | L3 | Y | 914 | Y | 4 | 228.50x |
| hard_motif_L1_002 | L1 | Y | 679 | Y | 3 | 226.33x |
| hard_motif_v1_ex1_004 | v1_ex1 | Y | 1365 | Y | 36 | 37.92x |
| hard_motif_L1_006 | L1 | Y | 172 | Y | 5 | 34.40x |
| hard_motif_P4_004 | P4 | Y | 520 | Y | 23 | 22.61x |
| hard_motif_P3_003 | P3 | Y | 45 | Y | 2 | 22.50x |
| hard_motif_P4_002 | P4 | Y | 530 | Y | 25 | 21.20x |
| hard_motif_P3_004 | P3 | Y | 38 | Y | 2 | 19.00x |
| hard_motif_P3_006 | P3 | Y | 37 | Y | 2 | 18.50x |
| hard_motif_P3_001 | P3 | Y | 26 | Y | 2 | 13.00x |
| hard_motif_P3_005 | P3 | Y | 26 | Y | 2 | 13.00x |
| hard_motif_P4_000 | P4 | Y | 257 | Y | 20 | 12.85x |
| hard_motif_P4_001 | P4 | Y | 258 | Y | 23 | 11.22x |
| hard_motif_P3_002 | P3 | Y | 17 | Y | 2 | 8.50x |
| hard_motif_v1_ex1_002 | v1_ex1 | Y | 262 | Y | 36 | 7.28x |
| hard_motif_v1_ex1_006 | v1_ex1 | Y | 236 | Y | 34 | 6.94x |
| hard_motif_P4_005 | P4 | Y | 159 | Y | 24 | 6.62x |
| hard_motif_L3_003 | L3 | Y | 21 | Y | 4 | 5.25x |
| hard_motif_P4_003 | P4 | Y | 102 | Y | 20 | 5.10x |
| hard_motif_v1_ex1_005 | v1_ex1 | Y | 175 | Y | 36 | 4.86x |
| hard_motif_v1_ex1_003 | v1_ex1 | Y | 173 | Y | 36 | 4.81x |
| hard_motif_v1_ex1_001 | v1_ex1 | Y | 163 | Y | 36 | 4.53x |
| hard_motif_L1_000 | L1 | Y | 16 | Y | 4 | 4.00x |
| hard_motif_L1_005 | L1 | Y | 16 | Y | 4 | 4.00x |
| hard_motif_v1_ex1_007 | v1_ex1 | Y | 136 | Y | 36 | 3.78x |
| hard_motif_R1_005 | R1 | Y | 16 | Y | 5 | 3.20x |
| hard_motif_R1_000 | R1 | Y | 9 | Y | 3 | 3.00x |
| hard_motif_R1_002 | R1 | Y | 9 | Y | 3 | 3.00x |
| hard_motif_L1_003 | L1 | Y | 12 | Y | 4 | 3.00x |
| hard_motif_L3_004 | L3 | Y | 10 | Y | 4 | 2.50x |
| hard_motif_R1_001 | R1 | Y | 17 | Y | 7 | 2.43x |
| hard_motif_R2_000 | R2 | Y | 9 | Y | 4 | 2.25x |
| hard_motif_L3_002 | L3 | Y | 9 | Y | 4 | 2.25x |
| hard_motif_R2_003 | R2 | Y | 9 | Y | 4 | 2.25x |
| hard_motif_L3_006 | L3 | Y | 9 | Y | 4 | 2.25x |
| hard_motif_v1_ex1_000 | v1_ex1 | Y | 76 | Y | 36 | 2.11x |
| hard_motif_L1_001 | L1 | Y | 10 | Y | 5 | 2.00x |
| hard_motif_R2_006 | R2 | Y | 10 | Y | 5 | 2.00x |
| hard_motif_L3_001 | L3 | Y | 9 | Y | 5 | 1.80x |
| hard_motif_R1_003 | R1 | Y | 9 | Y | 5 | 1.80x |
| hard_motif_R2_005 | R2 | Y | 9 | Y | 5 | 1.80x |
| hard_motif_R1_006 | R1 | Y | 9 | Y | 5 | 1.80x |
| hard_motif_R1_004 | R1 | Y | 9 | Y | 6 | 1.50x |
| hard_motif_R2_002 | R2 | Y | 9 | Y | 7 | 1.29x |
| hard_motif_R2_004 | R2 | Y | 9 | Y | 8 | 1.12x |
| hard_motif_P4_006 | P4 | Y | 14 | Y | 21 | 0.67x |