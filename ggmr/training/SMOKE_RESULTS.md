# Phase 2 Evaluation Results

Checkpoint: `checkpoints/smoke/best.pt`


## Aggregate

- Total problems: 20
- Joint solved (hand AND learned): 20
- Hand-only solved: 0  (regression detector)
- Learned-only solved: 0  (new problems opened)
- Both failed: 0
- **Geomean compression (joint)**: 0.928x
- Median compression (joint): 1.000x

## Per-family geomean (joint-solved subset)

| family | geomean |
|---|---|
| linear | 0.944x |
| polynomial | 1.000x |
| quadratic | 1.219x |
| rational | 0.645x |

## Per-problem

| id | family | hand | hand_nodes | learned | learned_nodes | ratio |
|---|---|---|---|---|---|---|
| lin01 | linear | Y | 1 | Y | 1 | 1.00x |
| lin02 | linear | Y | 5 | Y | 5 | 1.00x |
| lin03 | linear | Y | 1 | Y | 1 | 1.00x |
| lin04 | linear | Y | 6 | Y | 8 | 0.75x |
| lin05 | linear | Y | 1 | Y | 1 | 1.00x |
| qua01 | quadratic | Y | 1 | Y | 1 | 1.00x |
| qua02 | quadratic | Y | 1 | Y | 1 | 1.00x |
| qua03 | quadratic | Y | 17 | Y | 8 | 2.12x |
| qua04 | quadratic | Y | 2 | Y | 10 | 0.20x |
| qua05 | quadratic | Y | 19 | Y | 3 | 6.33x |
| rat01 | rational | Y | 2 | Y | 2 | 1.00x |
| rat02 | rational | Y | 3 | Y | 13 | 0.23x |
| rat03 | rational | Y | 4 | Y | 11 | 0.36x |
| rat04 | rational | Y | 2 | Y | 2 | 1.00x |
| rat05 | rational | Y | 8 | Y | 6 | 1.33x |
| poly01 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly02 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly03 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly04 | polynomial | Y | 1 | Y | 1 | 1.00x |
| poly05 | polynomial | Y | 1 | Y | 1 | 1.00x |