# Phase 2 Evaluation Results

Checkpoint: `checkpoints/round2/best.pt`


## Aggregate

- Total problems: 20
- Joint solved (hand AND learned): 20
- Hand-only solved: 0  (regression detector)
- Learned-only solved: 0  (new problems opened)
- Both failed: 0
- **Geomean compression (joint)**: 1.083x
- Median compression (joint): 1.000x

## Per-family geomean (joint-solved subset)

| family | geomean |
|---|---|
| linear | 1.084x |
| polynomial | 1.000x |
| quadratic | 1.682x |
| rational | 0.753x |

## Per-problem

| id | family | hand | hand_nodes | learned | learned_nodes | ratio |
|---|---|---|---|---|---|---|
| lin01 | linear | Y | 1 | Y | 1 | 1.00x |
| lin02 | linear | Y | 5 | Y | 5 | 1.00x |
| lin03 | linear | Y | 1 | Y | 1 | 1.00x |
| lin04 | linear | Y | 6 | Y | 4 | 1.50x |
| lin05 | linear | Y | 1 | Y | 1 | 1.00x |
| qua01 | quadratic | Y | 1 | Y | 1 | 1.00x |
| qua02 | quadratic | Y | 1 | Y | 1 | 1.00x |
| qua03 | quadratic | Y | 17 | Y | 4 | 4.25x |
| qua04 | quadratic | Y | 2 | Y | 6 | 0.33x |
| qua05 | quadratic | Y | 19 | Y | 2 | 9.50x |
| rat01 | rational | Y | 2 | Y | 4 | 0.50x |
| rat02 | rational | Y | 3 | Y | 11 | 0.27x |
| rat03 | rational | Y | 4 | Y | 3 | 1.33x |
| rat04 | rational | Y | 2 | Y | 3 | 0.67x |
| rat05 | rational | Y | 8 | Y | 4 | 2.00x |
| poly01 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly02 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly03 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly04 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly05 | polynomial | Y | 1 | Y | 1 | 1.00x |