# GGMR — Learned Heuristics for Sound Algebraic Rewrite Search

Research code for the GGMR program targeting ICLR 2026. The repo holds the working implementation and pre-registered results for each phase. The accompanying paper drafts (`ggmr_v9.pdf`, `ggmr_v10.pdf`) describe the framework and pre-registration rationale.

## Phases

| Phase | Status | Headline |
|---|---|---|
| **Phase 0** — Prerequisite validation (structural-complexity monotonicity) | done | `H = 0.6417` → non-myopic value learning is the primary contribution |
| **Phase 1a** — 15-rule library + BFS baseline | done | 20/20 Phase 0 problems solved within a 5k-node budget |
| **Phase 1b** — 45 rules + A* + Beam + reverse problem generator | done | A* compresses `rat05` from 151 → 8 expanded nodes (5.3% of BFS); all 7 pre-reg criteria met |
| **Phase 1c / 2** — hard evaluation set + learned value network | in progress | v2 motif-template set: 50 problems, 43 at A* ≥ 50 expansions, 21 at A* ≥ 1000 |

Per-phase READMEs:
- [`phase0/README.md`](phase0/README.md)
- [`ggmr/PHASE1A_README.md`](ggmr/PHASE1A_README.md)
- [`ggmr/PHASE1B_README.md`](ggmr/PHASE1B_README.md)

Pre-registration documents (`*_PREREG.md`) sit next to each README and were committed before the corresponding experiments were run.

## Layout

```
ggmr/                Phase 1a/1b package (rules, search, problems, tests)
  expr/              Tree, walker, serializer over SymPy expressions
  rules/core/        Rule library by family (arithmetic, algebra, rational, quadratic, polynomial, exponent)
  search/            BFS, A*, Beam over the rule-application graph
  heuristics/        Composite hand-tuned heuristic used by A*/Beam
  problems/          Reverse-application generator + motif templates + hard evaluation sets
  tests/             pytest suite (~200 tests, Phase 0 + 1a + 1b)
phase0/              Phase-0 validation pipeline + outputs (figures, CSVs, auto-report)
scripts/             Coverage and motif-set generation/validation scripts
```

## Install & run

```bash
python -m venv .venv
.venv/Scripts/activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest                          # full suite (slow tests are marked and skipped by default)
pytest -m slow                  # include long-running BFS/A* integration tests
```

Python ≥ 3.13. Dependencies: `sympy>=1.14`, `numpy>=2.0`, `pyyaml>=6.0`.

## Reproducing key results

- **Phase 0 monotonicity number:** `python -m phase0.src.run_phase0` (outputs land in `phase0/outputs/`).
- **Phase 1b coverage report:** `python scripts/validate_coverage.py` (~1–2h wall clock for the full 500-problem run).
- **Hard evaluation set v2:** `python scripts/generate_hard_eval_set_v2.py` then `python scripts/validate_hard_set.py`.
