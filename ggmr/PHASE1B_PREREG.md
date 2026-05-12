# PHASE 1b — PRE-REGISTRATION

**Project:** GGMR (Learned Heuristics for Sound Algebraic Rewrite Search), v10 proposal
**Phase:** 1b — Full rule library expansion (15 → 45) + A*/Beam search + problem generator + coverage validation
**Author:** Ansuman Mullick (Bilkent University)
**Pre-registration date:** 2026-05-10
**Status when written:** Phase 1a is complete (`ggmr/PHASE1A_PREREG.md`, `ggmr/PHASE1A_README.md` — 4/4 criteria met). No Phase 1b rule, search algorithm, or problem-generator code has been written. This file is committed to `ggmr/` BEFORE any new rule, A*, Beam, or generator code lands.

---

## 1. Research Question (Phase 1b scope)

Phase 1a validated the rule architecture against the Phase 0 problem set (15 rules, BFS, 20/20 within 5,000-node budget). Phase 1b's question is whether the architecture scales:

> Does extending the rule library to 45 rules, adding A*/Beam search to the existing infrastructure, and introducing a controlled-difficulty problem generator (a) preserve all Phase 1a guarantees (soundness, determinism, regression on the 20-problem set), (b) demonstrate concrete search-efficiency gains via A* on a Phase 1a outlier (`rat05`), and (c) validate the rule library's coverage against a 500-problem reverse-generated benchmark?

A pass on all 7 criteria below validates the foundation is fit for Phase 1c (e-graph + training data) and Phase 2 (value-network beam search). A fail on any criterion is a structural signal demanding revision before further phases.

---

## 2. Frozen Inputs

- **Phase 0 test set:** the 20 problems in `phase0/problems/problems.yaml`, frozen by Phase 0's pre-reg §2 and used unchanged in Phase 1a.
- **Phase 1a result:** 20/20 BFS coverage at 5,000 nodes; `rat05` expanded **151 nodes**; `lin02` expanded **338 nodes** (the wide-expansion outliers Phase 1b's new infrastructure should compress).
- **Phase 1a rule library:** 15 rules in `ggmr/rules/core/` (registered in `default_registry` in registration order).
- **Heuristic:** `WeightedSumCompositeHeuristic` from `ggmr/heuristics/composite.py` with default weights `{depth: 1.0, ops: 1.0, leaves: 1.0, isolation: 1.0}`. No grid search or weight tuning in Phase 1b — that is deferred to Phase 1c. The "hand heuristic" referenced in §3.2 is exactly this configuration.
- **User-confirmed scope (this turn):**
  - Multi-successor rules: principal-branch only (`QUADRATIC_FORMULA` emits `+√` only; no `ZERO_FACTOR_PROPERTY` case-split).
  - Skip transcendentals: no `TAKE_LOG`, `TAKE_EXP`, `LOG_PRODUCT`. Their guards (log-of-nonpositive, sqrt-of-negative-on-transcendentals) deferred to Phase 1c.
  - Problem-generator YAML: full Phase 0 schema with `trace` field included.

---

## 3. Success Criteria (7 criteria; ALL must pass)

### 3.1 Rule Library Size

`len(default_registry.rules) ≥ 45`. Each new rule (16–45) has at least:
- One positive-apply test (`apply` produces the expected next state),
- One guard-rejection test (or, for unguarded rules, an explicit no-op sentinel test that documents the absence of a guard),
- One soundness assertion (`solution_set(child) ⊆ solution_set(parent)` per `verify_transition`).

Tested by `test_rules_*_p1b.py` (per family) plus `test_rule_count_at_least_45` (cross-cutting).

### 3.2 A* Efficiency on `rat05`

A* with `WeightedSumCompositeHeuristic` (default weights, 1.0 each) solves `rat05` in:

```
nodes_expanded ≤ 75
```

within `max_nodes = 5,000` and `max_depth = 20`. The threshold of 75 is **strict** (76 fails) and reflects < 50% of Phase 1a's BFS measurement of 151 nodes.

This is the core empirical claim of Phase 1b: A* with the existing hand heuristic compresses search dramatically on a problem where BFS exhaustively explored cross-multiply variants. Tested by `test_astar_rat05_node_efficiency`.

### 3.3 Beam Search Coverage

Beam search with `beam_width = 10`, same `WeightedSumCompositeHeuristic`, solves **≥ 18/20** Phase 0 problems within `max_depth = 20`. The 2-problem failure budget reflects expected cases where greedy pruning misses the canonical-target path. Tested by `test_beam_solves_phase0_problem_set[B=10]`.

### 3.4 Problem Generator Solvability

For each depth ∈ {5, 10, 15}, ≥ **80%** of generated problems solve via BFS within `max_nodes = 5,000`, `max_depth = 20`. Computed per-depth (not aggregated). Depth 20 is reported alongside 5/10/15 but has **no threshold** — depth-20 timeout is anticipated and documented per the user's brief.

This validates that reverse generation produces solvable, well-formed equations across difficulty buckets. Tested by `test_problems_generator` for small batches; full-scale validation comes from §3.5.

### 3.5 Coverage Validation @ depth ≤ 10

`scripts/validate_coverage.py` generates 500 problems (5 templates × 4 depths × 25 problems) and runs BFS on each. The criterion:

```
solve_rate(depth ≤ 10) ≥ 0.90
```

Computed across the 250 problems at depth 5 and depth 10 combined (5 templates × 2 depths × 25 = 250). Per-depth + per-template breakdowns are reported in the JSON output. Tested by `test_coverage_validation` reading the script's output JSON.

### 3.6 Soundness Regression

0 unsound transitions across all paths in any test (any verdict of `VERIFY_UNSOUND` from `ggmr/soundness.py:verify_transition` aborts the run with `IllegalStepError`). Continued from Phase 1a §3.2 — the addition of 30 new rules and 2 new search algorithms must not introduce a single soundness violation.

### 3.7 Full Regression

After Phase 1b lands:
- All 58 Phase 1a tests pass.
- All 39 Phase 0 tests pass.

Phase 1b must not break any Phase 1a or Phase 0 functionality. Tested by running `pytest phase0/tests/ ggmr/tests/` end-to-end.

---

## 4. Tie-Breakers (committed)

- **§3.1 Rule count**: `≥ 45` is strict; 44 fails.
- **§3.2 `rat05` node count**: `≤ 75` is strict (76 fails). The number is a fixed % gate, not a "best effort." If A* solves `rat05` in 76 nodes, this criterion fails and is recorded; the criterion is not retroactively loosened.
- **§3.3 Beam coverage**: `≥ 18/20` is strict; 17/20 fails. The 2-problem failure budget covers expected edge cases (e.g., `lin02` if the heuristic prunes the multi-step isolation path before reaching the canonical target).
- **§3.4 Per-depth solvability**: `≥ 80%` is computed per depth, not aggregated. A depth bucket where < 80% solve fails the criterion for that bucket.
- **§3.5 `depth ≤ 10`**: includes both depth-5 and depth-10 buckets (250 problems total). `≥ 90%` is strict.
- **§3.7 Regression**: any single test failure in `phase0/tests/` or `ggmr/tests/test_*[!_p1b].py` (i.e., the original 58 + 39) fails this criterion.

Phase 0's `H = 0.6417` parity from Phase 1a §3.3 is **explicitly NOT re-asserted** in Phase 1b. The `ISOLATE_VARIABLE` macro (rule 18) and the addition of guard-introducing rules change the search-tree shape, and we accept the parity claim was a Phase 1a foundation finding valid for the Phase 1a tree shape. This is a deliberate scope choice; any future Phase 1c heuristic refinement may revisit parity then.

---

## 5. Secondary Metrics (always reported, no decision authority)

1. **A*/Beam time per problem** (median, p95) on the Phase 0 set
2. **Per-rule application frequency** across the 500-problem coverage batch — rules with **0 applications** are flagged as candidates for Phase 1c review (kept or removed)
3. **Productive-middle identification**: distribution of `nodes_expanded` per depth bucket; problems with `100 < nodes < 50,000` are the "productive middle" Phase 2's value network targets
4. **Dedup hit rate** for A* (closed-set hit rate) and BFS on the 500-problem batch
5. **Generator retry rate**: how often a generated problem is rejected (BFS doesn't solve within slack) and re-generated
6. **A* weighted-aggressiveness sweep**: `weight ∈ {1.0, 1.5, 2.0}` on `rat05` — reported but no decision authority (Phase 1c may use this for tuning)
7. **Beam-width sweep on Phase 0**: `B ∈ {2, 5, 10, 20}` solve rates — reported

---

## 6. Risks Owned

- **§3.2's `≤ 75 nodes` is aggressive.** With `WeightedSumCompositeHeuristic` and default weights, A* should aggressively prefer states with fewer ops. If actual is 80–100, that still beats BFS but fails the criterion. If criterion is missed, it is recorded as missed; the pre-reg is not retroactively loosened. Phase 1c may re-evaluate with tuned weights.
- **Inverse-rule infrastructure is novel.** No Phase 1a precedent. The 15 inverse rules must produce equations whose forward-BFS solve length matches inverse step count within `1.5×` slack. Forward-BFS finding shorter paths consistently means "depth" no longer reflects difficulty; this is monitored via §3.4's per-depth gate.
- **`AUXILIARY_VARIABLE_SUBSTITUTION` complexity.** Variable reassignment + un-substitution at solve time is nontrivial. If overruns, fallback is to defer this rule to Phase 1c and replace with `EVALUATE_AT_INTEGER`. Documented as a deferred risk; the 45-rule target is preserved either way.
- **`ISOLATE_VARIABLE` macro changes search-tree shape.** Phase 0 BFS results from Phase 1a will not exactly reproduce in Phase 1b. The Phase 1a tests covering exact node counts are **not** modified; if any pin a specific BFS expansion count for `lin02`, that test will break and require rebaselining. This is anticipated and acceptable.
- **500-problem coverage runtime.** ≈ 5s/problem average × 500 = ~40 minutes wall-clock. Acceptable as `pytest -m slow` or via `validate_coverage.py` invoked manually. Same pattern as Phase 1a's 22-min integration test.
- **Determinism under parallel pytest.** Both A* and Beam are single-process deterministic. If `pytest-xdist` is added in a later phase, RNG seeds in `templates.py` must be partition-deterministic. Deferred to Phase 1c.

---

## 7. Reporting Commitments

The Phase 1b smoke run (`pytest ggmr/tests/`) plus the coverage script (`scripts/validate_coverage.py`) together produce:

- Rule count assertion (≥ 45)
- A* `rat05` `nodes_expanded` value
- Beam `B=10` Phase 0 solve count
- Problem-generator per-depth solve rates (depth 5/10/15/20)
- 500-problem aggregate + per-depth + per-template solve rates
- Total soundness violations (must be 0)
- Phase 0 + Phase 1a regression pass count
- A*/Beam determinism check (run-vs-run hash of trace output)

These are reported in `ggmr/PHASE1B_README.md` post-execution. Modifications to this pre-registration must be appended as dated deltas, never silent edits.

---

## 8. Reproducibility

- Python 3.13.7, SymPy 1.14.0, NumPy ≥ 2.0, PyYAML ≥ 6.0, pytest ≥ 8.0
- Rule iteration order is fixed in `ggmr/rules/registry.py` (registration order)
- Action iteration order is fixed (sorted by `Action.canonical_key()`)
- BFS uses a deterministic FIFO queue
- A* uses a stable monotonic counter for tie-breaking on equal `f_score`
- Beam search uses `Action.canonical_key()` for tie-breaking on equal heuristic scores
- Problem generator: `random.Random(seed)` is used; the generator's output is deterministic per (seed, depth, template) tuple

---

*This pre-registration was written before any new rule, search algorithm, or generator code was implemented. Any post-execution edit must be tracked as a dated delta below this line.*

---

## Deltas (post-execution edits, if any)

*(none yet)*
