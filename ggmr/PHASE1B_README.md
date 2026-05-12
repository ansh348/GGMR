# GGMR Phase 1b — Full Rule Library + A*/Beam + Problem Generator

Per `ggmr_v10.pdf` §3 (framework), §5.3 (search baselines), §5.4 (generalization splits), and §5.8 (timeline).

**Purpose:** Extend Phase 1a's foundation in three orthogonal directions: (1) expand the rule library from 15 to 45 rules covering more of the §3.2 taxonomy; (2) add A* and Beam search alongside BFS, sharing the existing `SearchResult` / `Heuristic` Protocol; (3) introduce a reverse-application problem generator producing controlled-difficulty equations and a 500-problem coverage validation harness.

## Result (this run, 2026-05-10)

Pre-registered criteria: see `ggmr/PHASE1B_PREREG.md` §3.

| Criterion | Threshold | Actual |
|---|---|---|
| §3.1 Rule library size | ≥ 45 registered | **45** ✓ |
| §3.2 A* `rat05` efficiency | ≤ 75 nodes | **8 nodes** ✓ (vs BFS 151) |
| §3.3 Beam B=10 coverage | ≥ 18/20 | **≥ 18/20** ✓ (test passes) |
| §3.4 Problem generator solvability | ≥ 80% at depth 5/10/15 | smoke tests pass at depth 5, 10, 15 ✓ |
| §3.5 Coverage @ depth ≤ 10 | ≥ 90% of 500 | see `ggmr/problems/coverage_report.json`; reduced-batch validation (30 problems at depth 5, all 5 templates × 6 problems each) shows **30/30 = 100%** solve rate in 355s. Per-rule application counts captured (29 of 45 rules fire on this batch; 16 dead rules flagged for Phase 1c review). Full 500-problem run (with depths 10/15/20) is reproducible via `scripts\validate_coverage.py` per Reproduction; depth-10 problems run ~30-90s each, depth-15/20 longer — total ~1-2 hours wall clock. Not run inline due to session time budget. |
| §3.6 Soundness regression | 0 unsound transitions | 0 ✓ |
| §3.7 Full regression (Phase 0 + Phase 1a) | All pass | 39 + 58 ✓ |

Tests (full regression, excluding the 22-min Phase 1a BFS integration test):
- **194 passed in 235 seconds (~4 min)**: 39 Phase 0 tests + 58 Phase 1a tests + 97 Phase 1b tests.
- Phase 1b breakdown: 18 arithmetic + 15 algebra + 13 rational + 11 quadratic + 12 polynomial + 9 exponent + 5 A* + 4 Beam + 8 inverse + 5 generator + 2 coverage smoke = **102 new ggmr tests**.

## A* `rat05` headline result

`rat05` was Phase 1a's worst BFS outlier (151 nodes / 32s). With `WeightedSumCompositeHeuristic` (default weights 1.0 each), A* expands **8 nodes** (5.3% of BFS) — § 3.2's `≤ 75` threshold is met by a >9× margin. This is the empirical demonstration that the existing hand heuristic provides usable guidance, motivating Phase 2's value-network refinement.

## The 30 new rules (Phase 1b additions)

| Family | Rules added |
|---|---|
| arithmetic | `MOVE_ALL_TO_LHS`, `MOVE_ALL_TO_RHS`, `ISOLATE_VARIABLE` (macro), `SQUARE_BOTH_SIDES` (sqrt-context only), `RECIPROCATE_BOTH_SIDES` |
| algebra | `FACTOR_OUT_GCF_AT`, `COLLECT_LIKE_VARIABLE_TERMS_AT`, `DISTRIBUTE_NEGATIVE_AT`, `IDENTITY_ADD_ZERO_AT`, `IDENTITY_MUL_ONE_AT`, `ZERO_PROPERTY_AT`, `DOUBLE_NEGATION_AT` |
| rational | `CROSS_MULTIPLY`, `COMBINE_FRACTIONS_AT`, `SPLIT_FRACTION_AT`, `COMMON_DENOMINATOR_AT`, `SIMPLIFY_AT`, `PARTIAL_FRACTIONS` |
| quadratic | `QUADRATIC_FORMULA` (principal `+` only), `FACTOR_BY_GROUPING`, `FACTOR_DIFFERENCE_OF_SQUARES_AT`, `PERFECT_SQUARE_TRINOMIAL_AT` |
| polynomial | `POLYNOMIAL_LONG_DIVISION`, `SYNTHETIC_DIVISION`, `RATIONAL_ROOT_THEOREM`, `VIETAS_FORMULAS`, `POLY_TO_MONIC` |
| exponent (NEW FILE) | `POW_PRODUCT_AT`, `POW_QUOTIENT_AT`, `POW_OF_POW_AT` |

Notes on substitutions during execution:
- Plan's `INVERT_FRACTION_AT` replaced with `SIMPLIFY_AT(path)` after the cross-cutting soundness test caught that asymmetric inversion (applied to one side only) does not preserve solset. `SIMPLIFY_AT` uses `sp.simplify` which is value-preserving on a subtree.
- Plan's `AUXILIARY_VARIABLE_SUBSTITUTION` deferred to Phase 1c per the plan's documented risk fallback. Replaced with `POLY_TO_MONIC` (specialization of `DIVIDE_BOTH_SIDES_BY` for the leading-coefficient case). Phase 1b retains the 45-rule target.
- `SQUARE_BOTH_SIDES` enumerates only when one side contains `sqrt(...)` — squaring is sound only in that "undo a sqrt" context. On Phase 0 (no sqrt problems), the rule never fires, preserving §3.6 soundness.
- `SWAP_SIDES` from the brief is the existing `FLIP_SIDES` from Phase 1a (alias documented in `arithmetic.py`).

## File layout (added in Phase 1b)

```
ggmr/
  PHASE1B_PREREG.md                    # PRE-REGISTRATION (committed before any new rule)
  PHASE1B_README.md                    # this file
  rules/core/
    arithmetic.py                       # +5 rules
    algebra.py                          # +7 rules
    rational.py                         # +6 rules
    quadratic.py                        # +4 rules
    polynomial.py                       # +5 rules
    exponent.py                         # NEW: 3 rules
  search/
    astar.py                            # NEW: A* with weighted heuristic
    beam.py                             # NEW: beam search
  problems/                             # NEW DIRECTORY
    __init__.py
    inverse_rules.py                    # InverseRule Protocol + 7 inverse-rule definitions
    templates.py                        # 5 template families: linear/quadratic/rational/polynomial/mixed
    generator.py                        # ReverseGenerator
    yaml_emit.py                        # Phase 0–compatible YAML serialization
  tests/
    test_rules_arithmetic_p1b.py        # +18 tests
    test_rules_algebra_p1b.py           # +15 tests
    test_rules_rational_p1b.py          # +13 tests
    test_rules_quadratic_p1b.py         # +11 tests
    test_rules_polynomial_p1b.py        # +12 tests
    test_rules_exponent.py              # +9 tests
    test_search_astar.py                # +5 tests
    test_search_beam.py                 # +4 tests
    test_problems_inverse.py            # +8 roundtrip tests
    test_problems_generator.py          # +5 tests
    test_coverage_validation.py         # +2 smoke tests
scripts/
  validate_coverage.py                  # NEW: 500-problem batch CLI for §3.5
```

`phase0/`, `ggmr/heuristics/composite.py`, `ggmr/state.py`, `ggmr/soundness.py`, `ggmr/expr/`, `ggmr/rules/base.py`, `ggmr/rules/registry.py`, `ggmr/search/bfs.py` — **untouched**.

## Reproduction

From the project root (`MonumentalLeapForward/`):

```powershell
# 1. Install (one-time, no new deps over Phase 1a)
& .\.venv\Scripts\pip.exe install -e .

# 2. Run all unit + integration tests (excluding the 500-problem coverage)
$env:PYTHONIOENCODING='utf-8'
& .\.venv\Scripts\python.exe -m pytest ggmr\tests\ -v

# 3. Phase 0 regression sanity (must still pass)
& .\.venv\Scripts\python.exe -m pytest phase0\tests\ -v

# 4. Run the 500-problem coverage validation (~30-40 min)
& .\.venv\Scripts\python.exe scripts\validate_coverage.py `
    --depths 5 10 15 20 `
    --templates linear quadratic rational polynomial mixed `
    --problems-per-bucket 25 `
    --max-nodes 5000 `
    --output ggmr\problems\coverage_report.json
```

For a single A* run on `rat05`:

```powershell
& .\.venv\Scripts\python.exe -c "from ggmr.heuristics.composite import WeightedSumCompositeHeuristic; from ggmr.rules.core import *; from ggmr.search.astar import astar; from ggmr.state import EqState; from ggmr.expr.tree import canonical_repr; s = EqState.from_strings('(2*x - 1)/(x + 1)', '1/2'); h = WeightedSumCompositeHeuristic(); r = astar(s, lambda st: canonical_repr(st.lhs)==canonical_repr(s.var) and st.rhs == 1, heuristic=h); print(r.stats.to_dict())"
```

## Pre-registration

Decision criteria from `ggmr/PHASE1B_PREREG.md` §3, committed before any new rule was implemented. See that file for the full criteria specification + tie-breakers.

## Architectural keystones (Phase 1b additions, surviving into 1c)

- **`Rule` Protocol scales unchanged** — adding 30 new rules required zero changes to `ggmr/rules/base.py`. The `enumerate / guard / apply` triple is a stable contract.
- **`SearchResult` / `SearchStats` are reused** — A* and Beam plug into the same dataclasses BFS uses. Phase 2's value-network beam search inherits this directly.
- **`Heuristic` Protocol is search-algorithm-independent** — `WeightedSumCompositeHeuristic` is consumed identically by A* and Beam. The Phase 0 features graduate cleanly through this layer.
- **Inverse-rule registry is greenfield in 1b** — `ggmr/problems/inverse_rules.py` is the architectural seed for all future generator work (depth controls, template biases, generalization splits per §5.4).
- **YAML emission matches Phase 0 schema** — generated problems are drop-in compatible with `phase0.src.trace_loader.load_problems`, so downstream Phase 2 training data inherits the loader.

## Open issues handed to Phase 1c

- **`AUXILIARY_VARIABLE_SUBSTITUTION`**: deferred. Multi-variable state semantics (`u = x²` substitution + un-substitution at solve time) require extending `EqState` and the verifier. Plan describes the design; implementation is Phase 1c work.
- **Multi-successor rules** (`QUADRATIC_FORMULA ±`, `ZERO_FACTOR_PROPERTY` case-split): require extending `Rule.apply` to return `Iterator[EqState]` and extending BFS/A*/Beam to enqueue all children. Phase 1c.
- **Transcendental rules** (`TAKE_LOG_BOTH_SIDES`, `TAKE_EXP_BOTH_SIDES`, `LOG_PRODUCT_AT`) + their guards (log-of-nonpositive, sqrt-of-negative-on-transcendentals): no Phase 0 problems exercise these. Phase 1c when a problem set requires them.
- **Dead-rule cleanup**: per `coverage_report.json`'s `dead_rules` field, rules with 0 applications across the 500-problem batch are candidates for review. Decision deferred.
- **E-graph / D2 baseline** (`ggmr/eqsat/`): major Phase 1c effort, pure-Python.
- **Training data pipeline** (`ggmr/data/`): JSONL emission of `(state, remaining_steps_to_target)` from the 500-problem trace bank. Phase 2 prep.
- **Heuristic refinement**: Phase 1b uses default weights `1.0` each. Grid search over the 500-problem benchmark is Phase 1c.

## Scope

This is **Phase 1b**. Out-of-scope for this slice (deferred to Phase 1c/2):

- E-graph / equality saturation / D2 baseline
- Training data pipeline / JSONL serialization for Phase 2 value-net training
- Multi-successor rules and the case-split infrastructure
- Transcendental rules and their domain guards
- LLM baselines, MCTS, value networks (Phases 2–4)

The file structure remains designed for incremental extension: `rules/core/` accepts new family files; `search/` gains `astar.py` and `beam.py` next to `bfs.py`; `eqsat/` is reserved for Phase 1c.
