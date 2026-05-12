"""Generate the v2 hard evaluation set via motif templates.

Bypasses the reverse-application generator. Each of seven validated motif
families (v1_ex1, L1, L3, P3, P4, R1, R2) has ~10 hand-listed parameter
variations. Each is built via the corresponding template function in
`ggmr.problems.motif_templates`, then checked by `verify_instance`. Up to
the first 8 valid variations per family are kept, then the combined set
is trimmed to 50 problems and emitted as YAML in the format consumed by
`scripts/validate_hard_set.py`.

Run:
  python scripts/generate_hard_eval_set_v2.py
       [--output ggmr/problems/hard_evaluation_set_v2.yaml]
       [--per-family 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sympy as sp

from ggmr.problems.hard_yaml_emit import emit_hard_problems_yaml
from ggmr.problems.motif_templates import (
    MotifInstance,
    motif_v1_ex1,
    motif_l1,
    motif_l3,
    motif_p3,
    motif_p4,
    motif_r1,
    motif_r2,
    verify_instance,
)


PARAM_SWEEPS: dict[str, list[dict]] = {
    "v1_ex1": [
        {"linear_coef": 2, "lhs_const": 3,  "rhs_const": 7,  "twin_a": 1,  "twin_b": -4, "rational_root": 1},
        {"linear_coef": 3, "lhs_const": 1,  "rhs_const": 7,  "twin_a": 2,  "twin_b": -3, "rational_root": 4},
        {"linear_coef": 4, "lhs_const": -2, "rhs_const": 6,  "twin_a": 1,  "twin_b": 5,  "rational_root": 3},
        {"linear_coef": 5, "lhs_const": 0,  "rhs_const": 10, "twin_a": -2, "twin_b": 4,  "rational_root": 1},
        {"linear_coef": 2, "lhs_const": 1,  "rhs_const": 9,  "twin_a": 3,  "twin_b": -5, "rational_root": 2},
        {"linear_coef": 3, "lhs_const": 2,  "rhs_const": 11, "twin_a": -1, "twin_b": 6,  "rational_root": 1},
        {"linear_coef": 4, "lhs_const": 0,  "rhs_const": 12, "twin_a": 1,  "twin_b": -2, "rational_root": 5},
        {"linear_coef": 2, "lhs_const": -5, "rhs_const": 5,  "twin_a": 2,  "twin_b": -3, "rational_root": 4},
        {"linear_coef": 3, "lhs_const": -2, "rhs_const": 13, "twin_a": 4,  "twin_b": -1, "rational_root": 2},
        {"linear_coef": 2, "lhs_const": 7,  "rhs_const": 13, "twin_a": -3, "twin_b": 2,  "rational_root": 1},
    ],
    "L1": [
        {"linear_coef": 4, "lhs_const": -1, "rhs_const": -13, "twin1": (6, -2),  "twin2": (-3, 5)},
        {"linear_coef": 3, "lhs_const": 0,  "rhs_const": -12, "twin1": (2, 5),   "twin2": (-1, 4)},
        {"linear_coef": 2, "lhs_const": 1,  "rhs_const": 7,   "twin1": (3, -1),  "twin2": (2, -4)},
        {"linear_coef": 5, "lhs_const": 2,  "rhs_const": 27,  "twin1": (1, -3),  "twin2": (4, -2)},
        {"linear_coef": 4, "lhs_const": -3, "rhs_const": 13,  "twin1": (1, -5),  "twin2": (2, 3)},
        {"linear_coef": 3, "lhs_const": 1,  "rhs_const": 19,  "twin1": (-2, 4),  "twin2": (3, -1)},
        {"linear_coef": 6, "lhs_const": 0,  "rhs_const": -6,  "twin1": (2, 3),   "twin2": (-1, -2)},
        {"linear_coef": 2, "lhs_const": 5,  "rhs_const": -3,  "twin1": (1, -4),  "twin2": (-2, 3)},
        {"linear_coef": 3, "lhs_const": -2, "rhs_const": 10,  "twin1": (4, -1),  "twin2": (2, 3)},
        {"linear_coef": 5, "lhs_const": -1, "rhs_const": 14,  "twin1": (1, -3),  "twin2": (-4, 2)},
    ],
    "L3": [
        {"scalar": 2, "inner_coef": 3, "inner_const": 1,  "rhs_const": 26, "inner_twin": (2, -5),  "outer_twin": (-1, 4)},
        {"scalar": 3, "inner_coef": 2, "inner_const": 0,  "rhs_const": 18, "inner_twin": (1, 3),   "outer_twin": (-2, 5)},
        {"scalar": 2, "inner_coef": 5, "inner_const": 1,  "rhs_const": 12, "inner_twin": (2, -3),  "outer_twin": (-1, 4)},
        {"scalar": 3, "inner_coef": 4, "inner_const": 1,  "rhs_const": 39, "inner_twin": (1, -2),  "outer_twin": (3, -4)},
        {"scalar": 5, "inner_coef": 2, "inner_const": 1,  "rhs_const": 25, "inner_twin": (-1, 3),  "outer_twin": (2, -4)},
        {"scalar": 2, "inner_coef": 4, "inner_const": 3,  "rhs_const": 22, "inner_twin": (1, -3),  "outer_twin": (2, -5)},
        {"scalar": 4, "inner_coef": 2, "inner_const": 1,  "rhs_const": 20, "inner_twin": (2, -3),  "outer_twin": (-1, 4)},
        {"scalar": 2, "inner_coef": 3, "inner_const": 0,  "rhs_const": 12, "inner_twin": (1, 2),   "outer_twin": (-3, 4)},
        {"scalar": 3, "inner_coef": 3, "inner_const": 1,  "rhs_const": 30, "inner_twin": (2, -5),  "outer_twin": (-1, 4)},
        {"scalar": 2, "inner_coef": 5, "inner_const": -1, "rhs_const": 18, "inner_twin": (1, -2),  "outer_twin": (3, -4)},
    ],
    "P3": [
        {"roots": (3, 1, -4),  "irreducible_p": -2, "irreducible_q": 7, "linear_decoy_pair": (-5, 1)},
        {"roots": (2, -1, 5),  "irreducible_p": 1,  "irreducible_q": 3, "linear_decoy_pair": (-3, 4)},
        {"roots": (4, 2, -3),  "irreducible_p": -1, "irreducible_q": 5, "linear_decoy_pair": (1, -2)},
        {"roots": (5, -2, 1),  "irreducible_p": 2,  "irreducible_q": 3, "linear_decoy_pair": (-1, 4)},
        {"roots": (6, -1, 2),  "irreducible_p": -3, "irreducible_q": 4, "linear_decoy_pair": (5, -4)},
        {"roots": (3, -2, 4),  "irreducible_p": 1,  "irreducible_q": 2, "linear_decoy_pair": (-1, -5)},
        {"roots": (-5, 2, 1),  "irreducible_p": 3,  "irreducible_q": 5, "linear_decoy_pair": (4, -2)},
        {"roots": (7, -3, -2), "irreducible_p": -1, "irreducible_q": 4, "linear_decoy_pair": (1, 5)},
        {"roots": (4, -3, 2),  "irreducible_p": 2,  "irreducible_q": 5, "linear_decoy_pair": (1, -2)},
        {"roots": (1, -2, 5),  "irreducible_p": -2, "irreducible_q": 7, "linear_decoy_pair": (3, -1)},
    ],
    "P4": [
        # Scalars chosen via search to give integer N1 coefficients across denom/target combos.
        {"target_roots": (5, 2, -3),  "denom1": -1, "denom2": 4,  "scalar": 5, "twin": (7, -1),  "free_d": 3, "free_e": 0},
        {"target_roots": (4, 1, -2),  "denom1": 2,  "denom2": -3, "scalar": 5, "twin": (5, -1)},
        {"target_roots": (3, -1, 5),  "denom1": -2, "denom2": 6,  "scalar": 8, "twin": (2, -3)},
        {"target_roots": (2, 7, -4),  "denom1": 1,  "denom2": -5, "scalar": 2, "twin": (3, -2),  "free_d": 5, "free_e": 0},
        {"target_roots": (6, 1, -3),  "denom1": -1, "denom2": 4,  "scalar": 5, "twin": (2, -5)},
        {"target_roots": (4, -2, 3),  "denom1": 5,  "denom2": -1, "scalar": 3, "twin": (1, -4)},
        {"target_roots": (5, 3, -1),  "denom1": -3, "denom2": 2,  "scalar": 5, "twin": (4, -3)},
        {"target_roots": (-2, 4, 1),  "denom1": 3,  "denom2": -4, "scalar": 7, "twin": (5, -1)},
        {"target_roots": (6, -3, 2),  "denom1": -1, "denom2": 5,  "scalar": 4, "twin": (1, -4),  "free_d": 3, "free_e": 0},
        {"target_roots": (1, -5, 3),  "denom1": 4,  "denom2": -2, "scalar": 2, "twin": (3, -1),  "free_d": 5, "free_e": 0},
    ],
    "R1": [
        {"lhs_linear": 8,  "rhs_linear": 5, "lhs_const": 1, "rhs_const": 6,  "twin1": (6, -2),  "twin2": (-5, 1)},
        {"lhs_linear": 7,  "rhs_linear": 4, "lhs_const": 0, "rhs_const": 2,  "twin1": (3, -1),  "twin2": (2, -4)},
        {"lhs_linear": 9,  "rhs_linear": 5, "lhs_const": 2, "rhs_const": 8,  "twin1": (-3, 2),  "twin2": (4, -1)},
        {"lhs_linear": 8,  "rhs_linear": 3, "lhs_const": 1, "rhs_const": 4,  "twin1": (1, 3),   "twin2": (-2, 4)},
        {"lhs_linear": 7,  "rhs_linear": 2, "lhs_const": 0, "rhs_const": 4,  "twin1": (2, -3),  "twin2": (-1, 5)},
        {"lhs_linear": 9,  "rhs_linear": 4, "lhs_const": 1, "rhs_const": 8,  "twin1": (1, -4),  "twin2": (3, -2)},
        {"lhs_linear": 6,  "rhs_linear": 1, "lhs_const": 3, "rhs_const": 11, "twin1": (-1, 4),  "twin2": (2, 3)},
        {"lhs_linear": 5,  "rhs_linear": 2, "lhs_const": 0, "rhs_const": 4,  "twin1": (3, -2),  "twin2": (-1, 4)},
        {"lhs_linear": 11, "rhs_linear": 5, "lhs_const": 2, "rhs_const": 9,  "twin1": (4, -1),  "twin2": (-3, 2)},
        {"lhs_linear": 7,  "rhs_linear": 3, "lhs_const": 1, "rhs_const": 12, "twin1": (2, -1),  "twin2": (-4, 3)},
    ],
    "R2": [
        {"scalar": 2, "inner_coef": 5,  "inner_const": -3, "rhs_const": -10, "inner_twin": (4, -1),  "outer_twin": (-6, 2)},
        {"scalar": 3, "inner_coef": 2,  "inner_const": 1,  "rhs_const": 5,   "inner_twin": (1, -3),  "outer_twin": (2, 5)},
        {"scalar": 5, "inner_coef": 3,  "inner_const": 0,  "rhs_const": 8,   "inner_twin": (2, -1),  "outer_twin": (-4, 3)},
        {"scalar": 4, "inner_coef": 3,  "inner_const": 1,  "rhs_const": 7,   "inner_twin": (1, -2),  "outer_twin": (3, -5)},
        {"scalar": 2, "inner_coef": 7,  "inner_const": 1,  "rhs_const": 5,   "inner_twin": (3, -2),  "outer_twin": (1, -4)},
        {"scalar": 3, "inner_coef": 4,  "inner_const": -1, "rhs_const": 6,   "inner_twin": (2, -3),  "outer_twin": (1, -5)},
        {"scalar": 5, "inner_coef": 2,  "inner_const": -1, "rhs_const": 4,   "inner_twin": (-1, 3),  "outer_twin": (2, -4)},
        {"scalar": 2, "inner_coef": 11, "inner_const": 0,  "rhs_const": 8,   "inner_twin": (1, -4),  "outer_twin": (3, -2)},
        {"scalar": 3, "inner_coef": 5,  "inner_const": 2,  "rhs_const": 11,  "inner_twin": (-2, 4),  "outer_twin": (1, -3)},
        {"scalar": 4, "inner_coef": 7,  "inner_const": 1,  "rhs_const": 10,  "inner_twin": (2, -3),  "outer_twin": (-1, 5)},
    ],
}


TEMPLATE_FNS = {
    "v1_ex1": motif_v1_ex1,
    "L1": motif_l1,
    "L3": motif_l3,
    "P3": motif_p3,
    "P4": motif_p4,
    "R1": motif_r1,
    "R2": motif_r2,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="ggmr/problems/hard_evaluation_set_v2.yaml",
        help="Output YAML path (relative to repo root)",
    )
    parser.add_argument(
        "--per-family",
        type=int,
        default=8,
        help="Max valid variations to keep per motif family",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=50,
        help="Trim the final set to this many records",
    )
    args = parser.parse_args()

    var = sp.Symbol("x")
    per_family_records: dict[str, list[dict]] = {f: [] for f in TEMPLATE_FNS}
    counters: dict[str, int] = {}
    rejections: list[tuple[str, int, str]] = []

    for family, fn in TEMPLATE_FNS.items():
        counters[family] = 0
        for i, params in enumerate(PARAM_SWEEPS[family]):
            if counters[family] >= args.per_family:
                break
            try:
                inst = fn(var=var, **params)
            except (ValueError, AssertionError) as e:
                rejections.append((family, i, f"build: {e}"))
                continue
            ok, reason = verify_instance(inst)
            if not ok:
                rejections.append((family, i, f"verify: {reason}"))
                continue
            problem_id = f"hard_motif_{family}_{counters[family]:03d}"
            per_family_records[family].append(inst.to_record(problem_id))
            counters[family] += 1

    # Round-robin interleave so trimming preserves category diversity
    records: list[dict] = []
    max_per = max((len(v) for v in per_family_records.values()), default=0)
    for i in range(max_per):
        for family in TEMPLATE_FNS:
            if i < len(per_family_records[family]):
                records.append(per_family_records[family][i])
    records = records[: args.target_total]

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emit_hard_problems_yaml(records, str(out_path))

    print(f"\nWrote {len(records)} problems to {out_path}")
    print("Per-family counters:")
    for family, n in counters.items():
        print(f"  {family}: {n}")
    if rejections:
        print(f"\nRejections ({len(rejections)}):")
        for family, i, reason in rejections:
            print(f"  {family}#{i}: {reason}")

    # Category breakdown
    cat_counts: dict[str, int] = {}
    for r in records:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
    print(f"\nCategory counts: {cat_counts}")


if __name__ == "__main__":
    main()
