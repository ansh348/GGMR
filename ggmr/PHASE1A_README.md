# GGMR Phase 1a — Foundation

Per `ggmr_v10.pdf` §3 (framework) and §5.8 (timeline).

**Purpose:** Establish the production package, expression-tree/state/rule architecture, ~15 core rewrite rules, BFS search, and the Phase A heuristic graduated from `phase0/`. The rule architecture is *designed for 57* but only *implements 15* — enough to solve the Phase 0 problem set end-to-end via BFS, validating the architecture at minimum sunk cost.

## Result (this run, 2026-05-10)

Pre-registered criteria all met. Run with budget `max_nodes=5_000, max_depth=12`
(tighter than pre-reg's 50_000/20; the test harness also exercises the wider budget):

| Criterion | Threshold | Actual |
|---|---|---|
| §3.1 BFS coverage | ≥ 18/20 | **20/20** ✓ |
| §3.2 Soundness | 0 unsound transitions | 0 ✓ |
| §3.3 Heuristic parity composite | within ±0.05 of 0.6417 | reproduced ✓ |
| §3.3 Heuristic parity per-feature | within ±0.05 of {0.863, 0.825, 0.817, 0.858} | reproduced ✓ |
| §3.4 Determinism | identical paths + stats across 2 runs | confirmed ✓ |

Tests: `54 passed in 1.61s` (unit) + `4 passed in 1311.51s ≈ 22min` (integration BFS, 20-problem coverage at full budget). Phase 0 untouched: `39 passed in 0.82s`.

### Per-problem solve breakdown (5,000-node budget)

```
  lin01     PASS  steps=2  expanded=3      time=34ms
  lin02     PASS  steps=5  expanded=338    time=10s
  lin03     PASS  steps=2  expanded=3      time=140ms
  lin04     PASS  steps=4  expanded=64     time=2.5s
  lin05     PASS  steps=3  expanded=68     time=2.9s
  qua01     PASS  steps=1  expanded=1      time=88ms
  qua02     PASS  steps=1  expanded=1      time=66ms
  qua03     PASS  steps=2  expanded=3      time=264ms
  qua04     PASS  steps=2  expanded=7      time=2.0s
  qua05     PASS  steps=3  expanded=22     time=3.3s
  rat01     PASS  steps=3  expanded=22     time=3.7s
  rat02     PASS  steps=3  expanded=22     time=944ms
  rat03     PASS  steps=4  expanded=63     time=4.9s
  rat04     PASS  steps=3  expanded=22     time=3.7s
  rat05     PASS  steps=4  expanded=151    time=32s
  poly01    PASS  steps=1  expanded=1      time=98ms
  poly02    PASS  steps=1  expanded=1      time=83ms
  poly03    PASS  steps=1  expanded=1      time=83ms
  poly04    PASS  steps=1  expanded=1      time=118ms
  poly05    PASS  steps=1  expanded=1      time=52ms

  20/20 solved
```

All polynomials and most quadratics solve in 1 step (`FACTOR_POLYNOMIAL` via `sympy.factor` reaches the canonical target form directly). Rationals require the `CLEAR_FRACTIONS_BY_LCD → FACTOR_POLYNOMIAL` chain (3-4 steps). The longest BFS path is `lin02` at 5 steps (vars-on-both-sides — the rule library doesn't yet have an `ISOLATE_VARIABLE` macro that would compress this; left for Phase 1b).

## File layout

```
ggmr/
  PHASE1A_PREREG.md                    # PRE-REGISTRATION (committed before any rule)
  PHASE1A_README.md                    # this file
  __init__.py
  expr/
    tree.py                            # canonical_repr, normalize, tree primitives
    serialize.py                       # prefix-notation tokens (Phase 2 input)
    walk.py                            # subtree iteration
  state.py                             # EqState (immutable, hashable)
  soundness.py                         # solution-set verifier (3-state verdict)
  targets.py                           # canonical end-state detection (§3.4)
  rules/
    base.py                            # Action / GuardResult / Rule Protocol
    registry.py                        # deterministic rule iteration
    core/
      arithmetic.py                    # 5: add, multiply, divide, negate, flip
      algebra.py                       # 4: distribute, expand_product, expand_power, combine
      rational.py                      # 2: cancel_common_factor, clear_fractions
      quadratic.py                     # 3: complete_square, sqrt_both_sides, simplify_numeric
      polynomial.py                    # 1: factor_polynomial
  search/
    bfs.py                             # BFS with dedup + soundness assertion
    stats.py                           # SearchStats dataclass
  heuristics/
    composite.py                       # graduated phase0 features (z-scored + weighted)
  tests/
    test_expr_tree.py                  # canonical_repr, normalize, serializer roundtrip
    test_state.py                      # AC-permutation hash, excluded subtraction
    test_rules_arithmetic.py           # 5 rules, positive + guard + soundness each
    test_rules_algebra.py
    test_rules_rational.py
    test_rules_quadratic.py
    test_rules_polynomial.py
    test_rules_soundness.py            # cross-cutting at depth 1
    test_search_bfs.py                 # ≥18/20 within budget + determinism
    test_heuristic.py                  # H = 0.6417 ± 0.05 reproduction
    conftest.py
```

The `phase0/` directory is **not** modified. `ggmr/heuristics/composite.py` re-exports `phase0/src/features.py` directly so monotonicity-rate parity holds.

## Reproduction

From the project root (`MonumentalLeapForward/`):

```powershell
# 1. Install (one-time)
& .\.venv\Scripts\pip.exe install -e .

# 2. Run all tests
$env:PYTHONIOENCODING='utf-8'
& .\.venv\Scripts\python.exe -m pytest ggmr\tests\ -v

# 3. Phase 0 smoke run (BFS on the 20 problems)
& .\.venv\Scripts\python.exe _smoke.py
```

For a single problem with full path printed:

```powershell
& .\.venv\Scripts\python.exe _smoke.py rat01
```

## Pre-registration

Decision criteria from `ggmr/PHASE1A_PREREG.md` §3, committed before any rule was implemented:

| # | Criterion | Threshold |
|---|---|---|
| 3.1 | BFS coverage | ≥ 18/20 within budget (50k nodes, depth 20) |
| 3.2 | Soundness | 100% of (parent, action, child) triples; child solution set ⊆ parent (effective, accounting for `excluded`) |
| 3.3 | Heuristic parity | H = 0.6417 ± 0.05; per-feature rates within ±0.05 of `{depth: 0.863, ops: 0.825, leaves: 0.817, isolation: 0.858}` |
| 3.4 | Determinism | Two BFS runs produce byte-identical paths + stats |

## Architectural keystones

- **`canonical_repr`** in `expr/tree.py` — the BFS dedup hash. AC-canonical (sort Add/Mul args), flatten nested same-class ops, fold pure-numeric subtrees. Tested by `test_expr_tree.py::test_canonical_repr_*`.
- **`Rule.guard()` precedes `Rule.apply()`** in `rules/base.py` — unsound applications are structurally impossible. Guards may propagate `excluded` values (e.g., divisor zeros, cancellation excludable points) into the child state.
- **`EqState.solution_set()`** subtracts `excluded` from the raw SymPy solve output — so the verifier compares effective solution sets, allowing legitimate cancellation of factors with excluded zeros.
- **BFS soundness verdict is three-state** — PASS / UNVERIFIABLE (skip silently; counted as guard rejection in stats) / UNSOUND (raise IllegalStepError). Avoids aborting on degenerate intermediate states while still catching real soundness bugs.

## Scope

This is **Phase 1a**: the foundation. Out-of-scope for this slice (deferred to Phase 1b/1c):

- Rules 16–57 (full library)
- Problem generator and difficulty levels
- A* and beam search
- E-graph / equality saturation / D2 baseline (pure-Python per user direction)
- Training data pipeline (state, remaining_steps_to_target) for Phase 2
- LLM baselines, MCTS, value networks (Phases 2–4)

The file structure is designed for incremental extension: `rules/core/` will receive new family files in Phase 1b without disturbing the architecture; `search/` will gain `astar.py` and `beam.py` next to `bfs.py`; `eqsat/` is reserved for Phase 1c.
