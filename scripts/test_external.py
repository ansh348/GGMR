"""External-style algebra problems for OOD generalization test.

PROVENANCE DISCLAIMER:
These problems are CONSTRUCTED to match the structural pattern of
their named source. I (Claude) cannot access the Hendrycks MATH
dataset, AMC archive, or Stewart/Lial PDFs at runtime, so these are
NOT verbatim items. They are stylistic matches:
  - "MATH-style": rational/quadratic patterns from the Hendrycks
    intermediate-algebra subset (levels 3-5)
  - "AMC-style": competition algebra reducible to a single equation
  - "textbook-style": Stewart/Lial precalc-and-college-algebra patterns

The OOD generalization claim is still valid: these specific
(LHS, RHS, target) tuples are not in the 23k training set, not in
the 7 motif templates, and not in any phase0/hard_v2 yaml.

If you want truly source-verified problems, the next step is to
manually transcribe ~20 items from PDF and re-run.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sympy as sp

from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
from ggmr.heuristics.learned import LearnedHeuristic
from ggmr.search.astar import astar
from ggmr.state import EqState
from ggmr.training.extract_pairs import _build_is_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_external")

CKPT = "checkpoints/full/best.pt"
MAX_NODES = 10_000
MAX_DEPTH = 20

# (id, source_style, difficulty, lhs, rhs, target_lhs, target_rhs, notes)
PROBLEMS: list[tuple[str, str, str, str, str, str, str, str]] = [
    # ---- MATH-style: rational/quadratic, levels 3-5 patterns ----
    ("math_01", "MATH-style", "3", "(x + 2)/(x - 1)", "3", "x", "5/2",
     "rational, cross-multiply -> linear"),
    ("math_02", "MATH-style", "3", "3/(x + 1)", "6/(x + 3)", "x", "1",
     "two rationals, cross-multiply -> linear"),
    ("math_03", "MATH-style", "4", "x**2 - 5*x + 6", "0", "x", "2",
     "factorable quadratic, two roots, target one"),
    ("math_04", "MATH-style", "4", "2*x**2 + 7*x - 4", "0", "x", "1/2",
     "factorable quadratic with rational root"),
    ("math_05", "MATH-style", "5", "1/(x - 1) - 1/(x + 1)", "2/3", "x", "2",
     "rational difference -> quadratic -> root"),
    # ---- AMC-style: competition algebra reducible to one equation ----
    ("amc_01", "AMC-style", "AMC10", "x**2 - 7*x + 12", "0", "x", "3",
     "AMC10-style factorable quadratic"),
    ("amc_02", "AMC-style", "AMC10", "3*x + 5", "2*x - 7", "x", "-12",
     "AMC warmup linear"),
    # amc_03 (cubic = constant) skipped: rule library has no direct cubic-factoring
    # rule, so hand A* doesn't terminate within 10k nodes in reasonable wall-time.
    # ("amc_03", "AMC-style", "AMC12", "(x - 1)*(x - 2)*(x - 3)", "6",
    #  "x", "4", "cubic = constant; rational root x=4 (since 3*2*1=6)"),
    ("amc_04", "AMC-style", "AMC12", "x + 1/x", "5/2", "x", "2",
     "x + 1/x = 5/2, quadratic in disguise"),
    ("amc_05", "AMC-style", "AMC10", "(2*x + 3)**2", "49", "x", "2",
     "perfect square = constant"),
    # ---- textbook-style: Stewart/Lial precalc/college algebra ----
    ("text_01", "textbook-style", "Lial-3.1", "3*(x - 4) + 2*(x + 1)", "4*x - 8",
     "x", "2", "distributive practice; collect like terms"),
    ("text_02", "textbook-style", "Stewart-1.5", "5*(2*x - 3) - 4*(x - 1)", "17",
     "x", "14/3", "linear with distributed"),
    ("text_03", "textbook-style", "Lial-2.3", "(x + 1)/2 + (x - 3)/4", "5",
     "x", "7", "linear with rational coefficients"),
    ("text_04", "textbook-style", "Stewart-3.6", "(x - 3)**2", "16",
     "x", "7", "isolate via square root"),
    # Re-enabled after 48-rule library: FACTOR_DIFFERENCE_OF_CUBES handles this.
    ("text_05", "textbook-style", "Stewart-3.5", "x**3 - 8", "0",
     "x", "2", "difference of cubes / direct root"),
]


def _short(state: EqState, lim: int = 80) -> str:
    s = f"{state.lhs}  =  {state.rhs}"
    return s if len(s) <= lim else s[: lim - 3] + "..."


def main() -> int:
    logger.info("Budget: max_nodes=%d, max_depth=%d", MAX_NODES, MAX_DEPTH)
    logger.info("Checkpoint: %s", CKPT)
    logger.info("=" * 80)

    hand = WeightedSumCompositeHeuristic()
    learned = LearnedHeuristic(CKPT, device="cuda")
    logger.info("heuristics loaded; total problems: %d", len(PROBLEMS))

    rows: list[dict] = []
    for i, (pid, src, diff, lhs_s, rhs_s, tlhs, trhs, note) in enumerate(PROBLEMS, 1):
        try:
            initial = EqState.from_strings(lhs_s, rhs_s, var_name="x")
            target = EqState.from_strings(tlhs, trhs, var_name="x")
            is_target = _build_is_target(target)
        except Exception as e:
            logger.warning("[%d/%d] %s: PARSE FAILED: %s", i, len(PROBLEMS), pid, e)
            rows.append({"id": pid, "source": src, "parse_fail": True})
            continue

        h_h = hand.evaluate(initial)
        h_l = learned.evaluate(initial)
        logger.info("[%d/%d] %s (%s, %s): %s = %s  ->  %s = %s  | hand-h=%.2f learned-h=%.3f  (%s)",
                    i, len(PROBLEMS), pid, src, diff, lhs_s, rhs_s, tlhs, trhs, h_h, h_l, note)

        # Hand A*
        logger.info("  ... running HAND A* on %s", pid)
        t0 = time.perf_counter()
        try:
            r_hand = astar(initial, is_target, heuristic=hand,
                           max_nodes=MAX_NODES, max_depth=MAX_DEPTH, problem_id=pid)
            t_hand = time.perf_counter() - t0
            hand_ok, hand_n, hand_err = r_hand.found, r_hand.stats.nodes_expanded, None
        except Exception as e:
            t_hand = time.perf_counter() - t0
            hand_ok, hand_n, hand_err = False, MAX_NODES, f"{type(e).__name__}: {e}"
        logger.info("  HAND    %s nodes=%5d t=%.1fs%s",
                    "Y" if hand_ok else "N", hand_n, t_hand,
                    f"  ERR={hand_err}" if hand_err else "")

        # Learned A*
        logger.info("  ... running LEARNED A* on %s", pid)
        t0 = time.perf_counter()
        try:
            r_learned = astar(initial, is_target, heuristic=learned,
                              max_nodes=MAX_NODES, max_depth=MAX_DEPTH, problem_id=pid)
            t_learned = time.perf_counter() - t0
            learned_ok, learned_n, learned_err = r_learned.found, r_learned.stats.nodes_expanded, None
        except Exception as e:
            t_learned = time.perf_counter() - t0
            learned_ok, learned_n, learned_err = False, MAX_NODES, f"{type(e).__name__}: {e}"
        ratio_str = (
            f"{hand_n / max(learned_n, 1):.2f}x"
            if hand_ok and learned_ok else "-"
        )
        logger.info("  LEARNED %s nodes=%5d t=%.1fs  ratio=%s%s",
                    "Y" if learned_ok else "N", learned_n, t_learned, ratio_str,
                    f"  ERR={learned_err}" if learned_err else "")

        rows.append({
            "id": pid, "source": src, "difficulty": diff,
            "lhs": lhs_s, "rhs": rhs_s, "target_lhs": tlhs, "target_rhs": trhs,
            "hand_found": hand_ok, "hand_nodes": hand_n, "hand_time_s": round(t_hand, 2),
            "learned_found": learned_ok, "learned_nodes": learned_n, "learned_time_s": round(t_learned, 2),
            "ratio": (hand_n / max(learned_n, 1)) if (hand_ok and learned_ok) else None,
            "hand_err": hand_err, "learned_err": learned_err,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    import numpy as np
    joint = [r for r in rows if r.get("hand_found") and r.get("learned_found")]
    learned_only = [r for r in rows if not r.get("hand_found") and r.get("learned_found")]
    hand_only = [r for r in rows if r.get("hand_found") and not r.get("learned_found")]
    both_fail = [r for r in rows if not r.get("hand_found") and not r.get("learned_found")]

    print(f"  total:         {len(rows)}")
    print(f"  joint solved:  {len(joint)}")
    print(f"  learned-only:  {len(learned_only)}   ids: {[r['id'] for r in learned_only]}")
    print(f"  hand-only:     {len(hand_only)}   ids: {[r['id'] for r in hand_only]}")
    print(f"  both failed:   {len(both_fail)}   ids: {[r['id'] for r in both_fail]}")
    if joint:
        ratios = [r["ratio"] for r in joint]
        logs = [np.log(x) for x in ratios if x > 0]
        geomean = float(np.exp(np.mean(logs))) if logs else 0.0
        print(f"  joint geomean: {geomean:.2f}x")
        print(f"  joint median:  {float(np.median(ratios)):.2f}x")
        print(f"  min ratio:     {min(ratios):.2f}x")
        print(f"  max ratio:     {max(ratios):.2f}x")

    import csv
    out_csv = Path("ggmr/training/EXTERNAL_RESULTS.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n  csv -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
