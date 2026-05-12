# PHASE 0 — PRE-REGISTRATION

**Project:** GGMR (Learned Heuristics for Sound Algebraic Rewrite Search), v9 proposal
**Phase:** 0 — Prerequisite Validation (per §5.6, §2.1 Layer 1)
**Author:** Ansuman Mullick (Bilkent University)
**Pre-registration date:** 2026-05-10
**Status when written:** No features computed. No monotonicity rates observed. This file is committed to the repository BEFORE any execution of `run_phase0.py`.

---

## 1. Research Question

Does a hand-designed **structural complexity heuristic** φ(s) decrease monotonically along near-optimal solution paths for canonical algebraic rewrite problems?

If yes (≥80% step-level monotonicity), Phase A's hand-heuristic + beam search baseline is meaningful. If no, the GGMR paper's framing must shift toward non-myopic learned value as the primary contribution.

---

## 2. Test Set (frozen)

20 hand-curated equation-solving problems across four templates, 5 each:

| Category | IDs | Source |
|---|---|---|
| Linear | lin01–lin05 | Standard textbook simple/multi-step |
| Quadratic | qua01–qua05 | Factor / sqrt / complete-square / quadratic-formula |
| Rational | rat01–rat05 | Includes §2.1 motivating example `(x²−1)/(x−1)=3` |
| Polynomial | poly01–poly05 | Cubic/quartic, rational roots, substitution, grouping |

Problems are frozen in `phase0/problems/problems.yaml` before execution. The exact contents of that file at experiment runtime constitute the test set; modifications post-execution must be tracked as deltas in this pre-registration.

---

## 3. Solution Trace Methodology

Each problem has a **hand-curated solution trace** consisting of an ordered sequence of intermediate states (equations) leading from the initial form to the canonical target. Trace authorship aims for textbook-canonical near-shortest paths; we do **not** claim BFS-optimality (see §6 Risks).

Step-legality of every (s_t, s_{t+1}) pair on every trace is verified programmatically by `phase0/src/verifier.py` (SymPy solution-set equality). Any failed verification aborts the experiment. The verifier's results are recorded in `outputs/traces.csv` alongside features.

---

## 4. Structural Features φ(s)

Four features computed at every state, on `lhs - rhs` of the equation:

1. **`depth`** — maximum AST depth of the SymPy expression tree
2. **`ops`** — count of internal nodes (Add, Mul, Pow, Function)
3. **`leaves`** — count of atoms (Symbol, Number)
4. **`isolation`** — variable isolation score: `0` if equation is `Symbol = constant`; otherwise the count of target-variable occurrences across both sides (lower = more isolated)

**Composite**: `composite = z(depth) + z(ops) + z(leaves) + z(isolation)`, where `z(·)` is z-score normalization computed over **the full corpus of all states across all 20 problems**. The composite is the primary feature for the headline metric.

---

## 5. Monotonicity Metric

For a sequence of feature values `[v_0, v_1, …, v_T]` along a trace:

```
step_rate(v) = |{i : v[i+1] ≤ v[i]}| / (len(v) - 1)
```

Note: **non-strict** decrease (≤). Plateaus count as monotone. Strict-decrease rate is also reported as a secondary metric.

**Per-problem rate**: `step_rate` computed on the composite-feature sequence for that problem.
**Aggregate rate**: arithmetic mean of per-problem rates across all 20 problems.
**Headline number**: aggregate rate on the composite feature, on the **original** trace (not AC-variants).

---

## 6. Decision Rule

Let `H` = headline number defined in §5. Then:

| Condition | Framing |
|---|---|
| `H ≥ 0.80` | **Phase A is a meaningful baseline.** Continue paper as planned: §4.1 hand-heuristic + beam search is reportable. |
| `0.50 ≤ H < 0.80` | **Phase A is weak.** Paper restructures to emphasize non-myopic learned value (Phase B/C) as the primary contribution. §4.1 reported as a foil. |
| `H < 0.50` | **Phase A is a negative result.** Paper restructures around demonstrating why naive structural distance fails for algebraic rewrite. The negative result is itself the contribution. |

### Tie-breakers (committed)

- **Borderline values within ±0.02 of a threshold** round AGAINST the stronger claim. Examples:
  - `H = 0.795` is treated as `< 0.80` (weak Phase A)
  - `H = 0.485` is treated as `< 0.50` (negative result)
  - `H = 0.815` is treated as `≥ 0.80` only because 0.815 − 0.80 > 0.02 — wait, the rule is the reverse: 0.815 − 0.80 = 0.015 < 0.02, so 0.815 ALSO rounds against the stronger claim and is treated as `< 0.80` (weak Phase A).
  - **Clarification, in plain words:** if `|H − 0.80| < 0.02` → treat as `< 0.80`. If `|H − 0.50| < 0.02` → treat as `< 0.50`. The conservative side wins.

- **Per-feature divergence**: if at least 3 of the 4 individual features disagree with the composite by ≥0.10 in either direction, flag the composite as unstable in the report (decision still uses composite, but report annotates).

---

## 7. Secondary Metrics (always reported, no decision authority)

1. **Per-feature monotonicity rates** — depth, ops, leaves, isolation, each separately
2. **Per-category breakdown** — linear / quadratic / rational / polynomial, separately
3. **Strict-decrease rate** — `|{i : v[i+1] < v[i]}| / (len(v) - 1)`, headline-feature-only
4. **AC-variant fragility (σ)** — for each problem, generate 3 AC-equivalent variants (permute Add args, permute Mul args, rename variable). Recompute monotonicity on each. Report σ across {original, var1, var2, var3} per problem; aggregate as mean σ. **σ > 0.10 anywhere** is flagged as parser-dependent fragility.
5. **Path-length distribution** — for transparency; long paths bias monotonicity rate upward via more decreasing pairs available.

---

## 8. Risks Owned (acknowledged before execution)

- **Hand-curated traces are not provably BFS-optimal.** Bias direction is *toward* monotonicity (longer paths give more decreasing pairs available), so a fail is still a real fail; a pass with `H` near 0.80 is suspicious. Phase 1's e-graph baseline will validate.
- **20 problems is a small sample.** No bootstrap CI is reported (would be an unjustified statistical claim at n=20). The threshold rule is a coarse Go/pivot trigger, not a statistical test.
- **Feature definitions are arbitrary.** Reporting all 4 features individually and the composite mitigates cherry-picking; per-feature divergence is flagged.
- **Test-set selection bias.** The 20 problems are textbook-typical, not adversarial. Phase 1+ generalization splits (§5.4) compensate.

---

## 9. Reporting Commitments

The auto-generated `phase0/outputs/PHASE0_REPORT.md` will report ALL of:
- The headline number `H` (composite, mean across 20 problems)
- The framing decision per §6 (deterministic mapping from `H`)
- All §7 secondary metrics
- Per-problem composite trajectories (in the notebook)
- Verification outcome for each trace

Modifications to this pre-registration must be appended to the file as dated deltas, never silent edits.

---

## 10. Reproducibility

- Random seeds: deterministic. SymPy `parse_expr` is deterministic given inputs. AC-variant generation uses a fixed seed (set in `variants.py`).
- Re-running `run_phase0.py` on the frozen `problems.yaml` produces byte-identical CSV outputs.
- Environment captured in `phase0/requirements.txt`.

---

*This pre-registration was written before any features were computed, before any monotonicity rates were observed, and before the 20-problem test set was validated by the verifier. Any post-execution edit must be tracked as a dated delta below this line.*

---

## Deltas (post-execution edits, if any)

*(none yet)*
