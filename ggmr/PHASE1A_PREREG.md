# PHASE 1a — PRE-REGISTRATION

**Project:** GGMR (Learned Heuristics for Sound Algebraic Rewrite Search), v10 proposal
**Phase:** 1a — Foundation: expression tree + state + 15 core rewrite rules + BFS search engine
**Author:** Ansuman Mullick (Bilkent University)
**Pre-registration date:** 2026-05-10
**Status when written:** No `ggmr/` rule has been implemented. No BFS run has been executed. This file is committed to `ggmr/` BEFORE any rule definition or search invocation.

---

## 1. Research Question (Phase 1a scope)

Per `ggmr_v10.pdf` §3.1–§3.2 and §5.8 timeline, Phase 1 builds the algebra encoder, a ~57-rule guarded rewrite library, and the D1/D2 baselines. **Phase 1a is the foundation slice.** The research question for this slice is:

> Can we construct a guarded rule architecture and a BFS search engine that (a) solves the Phase 0 problem set end-to-end, (b) preserves solution-set soundness on every transition, and (c) preserves the Phase 0 monotonicity-rate measurement, without modifying `phase0/`?

A pass on all four criteria below validates the rule architecture is fit to scale to the full ~57-rule library in Phase 1b. A fail on any criterion is a structural signal that the architecture must be revised before Phase 1b begins.

---

## 2. Frozen Inputs

- **Test set:** the 20 problems in `phase0/problems/problems.yaml` (frozen by Phase 0 pre-reg §2). No new problems are added in Phase 1a.
- **Phase 0 result:** `H = 0.6417` (composite z-scored mean across 20 problems, per `phase0/PHASE0_FINDINGS.md`). Per-feature: depth=0.863, ops=0.825, leaves=0.817, isolation=0.858.
- **Rule library at execution time:** the 15 rules registered in `ggmr/rules/core/` at the moment `pytest ggmr/tests/` is run. Modifications post-execution must be tracked as deltas in this file.

---

## 3. Success Criteria (4 criteria; ALL must pass)

### 3.1 Primary: BFS Coverage

BFS with the 15-rule library solves **≥18/20 Phase 0 problems** within budget:
- `max_nodes = 50,000` expanded (excluding deduplicated revisits)
- `max_depth = 20`

Either limit hit on a given problem counts as failure for that problem.

The 2-problem failure budget reflects that 15 rules ≠ 57; some Phase 0 problems may legitimately require rules not yet implemented (e.g., `poly02`'s `x⁴ - 5x² + 4 = 0` needs auxiliary-variable substitution if `FACTOR_POLYNOMIAL` cannot factor a quartic in one shot). Failure on the *wrong* problems (any linear, or more than 1 from any single category) is treated as a coverage gap that signals an architecture bug, even if total coverage is ≥18/20.

### 3.2 Soundness

100% of (state, action, next_state) triples produced by `bfs()` on every Phase 0 problem satisfy:

```
solution_set(next_state) ⊆ solution_set(state)
```

Equality is the dominant case; strict subset is allowed when the rule legitimately removes an extraneous root (e.g., `CANCEL_COMMON_FACTOR` removing a factor at an excluded value), per the same predicate Phase 0's verifier already encodes.

Any unsound transition aborts the run with `IllegalStepError(problem_id, step_idx, parent_canonical, child_canonical)`. Soundness assertion is enabled inside `bfs()` by default in Phase 1a.

### 3.3 Heuristic Parity (Phase 0 Reproducibility)

`ggmr/heuristics/composite.py` evaluated on the Phase 0 traces (loaded via `phase0.src.trace_loader.load_problems`) reproduces:
- Composite mean rate: `0.6417 ± 0.05`
- Per-feature rates: each within ±0.05 of `{depth: 0.863, ops: 0.825, leaves: 0.817, isolation: 0.858}`

Parity proves the graduated heuristic preserves Phase 0's measurement under refactor. Any deviation outside ±0.05 is a structural change in the heuristic and demands explicit deltas in this pre-reg before merging.

### 3.4 Determinism

Two consecutive runs of `bfs(initial)` on the same input produce byte-identical:
- `path: list[(state, action)]`
- `stats.rule_application_counts: dict[str, int]`
- `stats.dedup_hits` and `stats.nodes_expanded`

Determinism is enforced by canonical iteration order in `ggmr/rules/registry.py` (rules in registration order; actions in `repr(params)` lexicographic order).

---

## 4. Tie-Breakers (committed)

- **Budget**: "within budget" means `nodes_expanded ≤ 50,000` AND `depth_reached ≤ 20`. Hitting either threshold is treated as failure.
- **Heuristic parity**: the ±0.05 tolerance is **strict**. A deviation of exactly 0.05 fails (i.e., 0.692 fails, 0.694 fails — the parity criterion uses `< 0.05`, not `≤ 0.05`).
- **Per-category coverage**: even if total ≥18/20 holds, failures clustered in one category trigger a manual architecture review before Phase 1b. Specifically:
  - All 5 linear must solve.
  - At most 1 failure each in quadratic, rational, polynomial.

---

## 5. Secondary Metrics (always reported, no decision authority)

1. `nodes_expanded` distribution per problem (mean, max, p95)
2. Wall-clock time per problem
3. Dedup hit rate (`dedup_hits / nodes_generated`) — should be > 0 on at least one problem
4. Per-rule application frequency across all BFS runs
5. Guard rejection rate — fraction of enumerated actions rejected before `apply()`
6. Path length distribution — for transparency; longer paths bias the soundness-assertion runtime upward

---

## 6. Risks Owned

- **`canonical_repr` correctness is load-bearing.** A buggy AC-canonicalization either over-dedups (silent missed solutions) or under-dedups (budget exhaustion). Mitigated by AC-permutation hash test in `test_state.py` and by §3.1 BFS coverage acting as a downstream sanity check.
- **15 rules ≠ 57.** ~2/20 Phase 0 problems may legitimately fail; architecture is unaffected. Failure on the *wrong* problems is the structural signal we care about (see §4).
- **SymPy auto-canonicalization in `apply()`.** Phase 0 documented systematic AST inflation under `evaluate=False`. Phase 1a inherits this; tests pin the expected behavior.
- **Soundness assertion overhead.** Calling `_solution_set` on every BFS expansion is expensive but acceptable in Phase 1a since the budget is only 50k nodes. If it dominates wall-clock, Phase 1b adds an opt-out for repeated subtree calls.
- **Phase 0 trace mtimes.** This pre-reg references Phase 0 results numerically; `phase0/` files are read but not modified, so the cross-phase verification in §3.3 is a snapshot, not a coupling.

---

## 7. Reporting Commitments

The Phase 1a smoke run (`python -m ggmr.search.bfs --problems phase0\problems\problems.yaml`) and `pytest ggmr/tests/` together produce:

- Per-problem solve outcome (pass/fail) + nodes expanded + path length
- Aggregate ≥18/20 ratio
- Total soundness violations (must be 0)
- Heuristic parity deltas (composite + per-feature)
- Determinism check (run-vs-run hash of trace output)

These are reported in `ggmr/PHASE1A_README.md` post-execution. Modifications to this pre-registration must be appended as dated deltas, never silent edits.

---

## 8. Reproducibility

- Python 3.13.7, deterministic SymPy 1.14.0
- Rule iteration order is fixed in `ggmr/rules/registry.py` (registration order)
- Action iteration order is fixed (sorted by `repr(action.params)`)
- BFS uses a deterministic FIFO queue
- No randomness in the Phase 1a code path; `random` module is not imported

---

*This pre-registration was written before any rule was implemented, before any BFS execution, and before any feature was computed in `ggmr/`. Any post-execution edit must be tracked as a dated delta below this line.*

---

## Deltas (post-execution edits, if any)

*(none yet)*
