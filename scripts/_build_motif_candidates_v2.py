"""Build 14 hand-crafted motif candidate equations (v2, isolation-respecting).

Tests whether the isolation principle (no shared denominators, no visible
target factors/coefficients, semantic-twin shields) generalizes beyond
the single Example 1 winner from v1.

Math has been independently verified for each motif. Two arithmetic
corrections to the input proposal:
  - R2: target is x=-2/5 (LHS-RHS = 2*(5x+2), not the inverse)
  - R3: target is x=7/4 (LHS-RHS = -3*(4x-7)/(denom), not 4/7)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sympy as sp
import yaml
from sympy import Add, Mul, Pow, Symbol, Integer

x = Symbol("x")


def A(*args):
    return Add(*args, evaluate=False)


def M(*args):
    return Mul(*args, evaluate=False)


def P(b, e):
    return Pow(b, e, evaluate=False)


def I(n):
    return Integer(n)


problems = []


def add(name, lhs, rhs, excluded, target_lhs, target_rhs, category):
    problems.append({
        "name": name, "lhs": lhs, "rhs": rhs,
        "excluded": excluded,
        "target_lhs": target_lhs, "target_rhs": target_rhs,
        "category": category,
    })


# ------------------------------------------------------------------
# A. Linear targets x=k
# ------------------------------------------------------------------

# L1: 4x-1 + (x+6)(x-2) + (x-3)(x+5) = -13 + (x^2+4x-12) + (x^2+2x-15)
# target x=-3 (LHS-RHS = 4(x+3))
add("L1_triple_twin_shield",
    A(M(I(4), x), I(-1),
      M(A(x, I(6)), A(x, I(-2))),
      M(A(x, I(-3)), A(x, I(5)))),
    A(I(-13),
      A(P(x, I(2)), M(I(4), x), I(-12)),
      A(P(x, I(2)), M(I(2), x), I(-15))),
    [], "x", "-3", "linear")

# L2: 2/(x+1) + (x+6)(x-2) + (x-3)(x+5) = 1/(x-2) + (x^2+4x-12) + (x^2+2x-15)
# target x=5, excluded {-1, 2}
add("L2_cross_reciprocal",
    A(M(I(2), P(A(x, I(1)), I(-1))),
      M(A(x, I(6)), A(x, I(-2))),
      M(A(x, I(-3)), A(x, I(5)))),
    A(P(A(x, I(-2)), I(-1)),
      A(P(x, I(2)), M(I(4), x), I(-12)),
      A(P(x, I(2)), M(I(2), x), I(-15))),
    [I(-1), I(2)], "x", "5", "linear")

# L3: 2(3x+1+(x+2)(x-5)) + (x-1)(x+4) = 26 + 2(x^2-3x-10) + (x^2+3x-4)
# target x=4 (LHS-RHS = 6(x-4))
add("L3_distributed_scalar_gate",
    A(M(I(2), A(M(I(3), x), I(1), M(A(x, I(2)), A(x, I(-5))))),
      M(A(x, I(-1)), A(x, I(4)))),
    A(I(26),
      M(I(2), A(P(x, I(2)), M(I(-3), x), I(-10))),
      A(P(x, I(2)), M(I(3), x), I(-4))),
    [], "x", "4", "linear")

# ------------------------------------------------------------------
# B. Quadratic targets (x-r1)(x-r2)=0
# ------------------------------------------------------------------

# Q1: (x+5)(x-1)+(x+1)(x-4)+(x+7)(x-1) = 5x+1+(x^2-3x-4)+(x^2+6x-7)
# target (x-3)(x+2)=0
add("Q1_unrelated_product_minus_linear",
    A(M(A(x, I(5)), A(x, I(-1))),
      M(A(x, I(1)), A(x, I(-4))),
      M(A(x, I(7)), A(x, I(-1)))),
    A(M(I(5), x), I(1),
      A(P(x, I(2)), M(I(-3), x), I(-4)),
      A(P(x, I(2)), M(I(6), x), I(-7))),
    [], "(x-3)*(x+2)", "0", "quadratic")

# Q2: (7x+20)/(x+2) = 2x/(x-3)
# target (x-4)(x+3)=0, excluded {-2, 3}
add("Q2_cross_ratio_no_shared_denom",
    M(A(M(I(7), x), I(20)), P(A(x, I(2)), I(-1))),
    M(M(I(2), x), P(A(x, I(-3)), I(-1))),
    [I(-2), I(3)], "(x-4)*(x+3)", "0", "quadratic")

# Q3: (x-5)^2 + (x+2)(x-6) = -2x+18 + (x^2-4x-12)
# target (x-1)(x-7)=0
add("Q3_off_center_square_gate",
    A(P(A(x, I(-5)), I(2)),
      M(A(x, I(2)), A(x, I(-6)))),
    A(M(I(-2), x), I(18),
      A(P(x, I(2)), M(I(-4), x), I(-12))),
    [], "(x-1)*(x-7)", "0", "quadratic")

# Q4: (x+7)(x-2)+(x-4)(x+1)+(x+6)(x-1) = 3x+1+(x^2-3x-4)+(x^2+5x-6)
# target (x+5)(x-3)=0
add("Q4_two_independent_twins",
    A(M(A(x, I(7)), A(x, I(-2))),
      M(A(x, I(-4)), A(x, I(1))),
      M(A(x, I(6)), A(x, I(-1)))),
    A(M(I(3), x), I(1),
      A(P(x, I(2)), M(I(-3), x), I(-4)),
      A(P(x, I(2)), M(I(5), x), I(-6))),
    [], "(x+5)*(x-3)", "0", "quadratic")

# ------------------------------------------------------------------
# C. Polynomial targets degree 3+
# ------------------------------------------------------------------

# P1: (x+5)(x-3)(x+1)+(x+7)(x-2) = (2x+5)(3x-4)-14x-3+(x^2+5x-14)
# target (x-4)(x-1)(x+2)=0
add("P1_cubic_unrelated_residual",
    A(M(A(x, I(5)), A(x, I(-3)), A(x, I(1))),
      M(A(x, I(7)), A(x, I(-2)))),
    A(M(A(M(I(2), x), I(5)), A(M(I(3), x), I(-4))),
      M(I(-14), x), I(-3),
      A(P(x, I(2)), M(I(5), x), I(-14))),
    [], "(x-4)*(x-1)*(x+2)", "0", "polynomial")

# P2: (4x^2-16x-4)/(x+3) + (x+4)(x-1) = x^2-5x+2 + (x^2+3x-4)
# target (x-5)(x-2)(x+1)=0, excluded {-3}
add("P2_one_sided_denom_cubic",
    A(M(A(M(I(4), P(x, I(2))), M(I(-16), x), I(-4)),
        P(A(x, I(3)), I(-1))),
      M(A(x, I(4)), A(x, I(-1)))),
    A(A(P(x, I(2)), M(I(-5), x), I(2)),
      A(P(x, I(2)), M(I(3), x), I(-4))),
    [I(-3)], "(x-5)*(x-2)*(x+1)", "0", "polynomial")

# P3: (x+2)(x^2-2x+7)+(x-5)(x+1) = 16x+2+(x^2-4x-5)
# target (x-3)(x-1)(x+4)=0
add("P3_irreducible_quadratic_disguise",
    A(M(A(x, I(2)), A(P(x, I(2)), M(I(-2), x), I(7))),
      M(A(x, I(-5)), A(x, I(1)))),
    A(M(I(16), x), I(2),
      A(P(x, I(2)), M(I(-4), x), I(-5))),
    [], "(x-3)*(x-1)*(x+4)", "0", "polynomial")

# P4: (8x^2+15x-29)/(x+1)+(x+7)(x-1) = (3x^2-34)/(x-4)+(x^2+6x-7)
# target (x-5)(x-2)(x+3)=0, excluded {-1, 4}
add("P4_cubic_two_denom_cross_ratio",
    A(M(A(M(I(8), P(x, I(2))), M(I(15), x), I(-29)),
        P(A(x, I(1)), I(-1))),
      M(A(x, I(7)), A(x, I(-1)))),
    A(M(A(M(I(3), P(x, I(2))), I(-34)),
        P(A(x, I(-4)), I(-1))),
      A(P(x, I(2)), M(I(6), x), I(-7))),
    [I(-1), I(4)], "(x-5)*(x-2)*(x+3)", "0", "polynomial")

# ------------------------------------------------------------------
# D. Rational canonical targets x=p/q
# ------------------------------------------------------------------

# R1: 8x+1+(x+6)(x-2)+(x-5)(x+1) = 5x+6+(x^2+4x-12)+(x^2-4x-5)
# target x=5/3 (LHS-RHS = 3x-5)
add("R1_fractional_difference_hiding",
    A(M(I(8), x), I(1),
      M(A(x, I(6)), A(x, I(-2))),
      M(A(x, I(-5)), A(x, I(1)))),
    A(M(I(5), x), I(6),
      A(P(x, I(2)), M(I(4), x), I(-12)),
      A(P(x, I(2)), M(I(-4), x), I(-5))),
    [], "x", "5/3", "linear")

# R2: 2(5x-3+(x+4)(x-1))+(x-6)(x+2) = -10+2(x^2+3x-4)+(x^2-4x-12)
# target x=-2/5 (LHS-RHS = 2(5x+2); GPT inverted the fraction)
add("R2_distributed_fractional",
    A(M(I(2), A(M(I(5), x), I(-3), M(A(x, I(4)), A(x, I(-1))))),
      M(A(x, I(-6)), A(x, I(2)))),
    A(I(-10),
      M(I(2), A(P(x, I(2)), M(I(3), x), I(-4))),
      A(P(x, I(2)), M(I(-4), x), I(-12))),
    [], "x", "-2/5", "linear")

# R3: -11/(x+1)+(x+6)(x-2)+(x-5)(x+1) = 1/(x-2)+(x^2+4x-12)+(x^2-4x-5)
# target x=7/4 (LHS-RHS = -3(4x-7)/((x+1)(x-2)); GPT inverted)
add("R3_cross_reciprocal_fractional",
    A(M(I(-11), P(A(x, I(1)), I(-1))),
      M(A(x, I(6)), A(x, I(-2))),
      M(A(x, I(-5)), A(x, I(1)))),
    A(P(A(x, I(-2)), I(-1)),
      A(P(x, I(2)), M(I(4), x), I(-12)),
      A(P(x, I(2)), M(I(-4), x), I(-5))),
    [I(-1), I(2)], "x", "7/4", "linear")


records = []
for i, p in enumerate(problems):
    records.append({
        "id": f"motif_v2_{i:03d}_{p['name']}",
        "category": p["category"],
        "recipe": "hand_crafted_motif_v2",
        "difficulty": "hard",
        "variable": "x",
        "source": "ggmr-motif-validation-v2",
        "seed": 0,
        "depth": 0,
        "astar_nodes_expanded": 0,
        "bfs_nodes_expanded": 0,
        "applied_inverses": [],
        "initial_srepr_lhs": sp.srepr(p["lhs"]),
        "initial_srepr_rhs": sp.srepr(p["rhs"]),
        "excluded_srepr": sorted(sp.srepr(e) for e in p["excluded"]),
        "initial": {
            "lhs": str(p["lhs"]),
            "rhs": str(p["rhs"]),
        },
        "canonical_target": {
            "lhs": p["target_lhs"],
            "rhs": p["target_rhs"],
        },
    })


out_path = ROOT / "ggmr" / "problems" / "motif_candidates_v2.yaml"
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(records, f, sort_keys=False, allow_unicode=True)

print(f"Wrote {len(records)} problems to {out_path}")
for r in records:
    print(f"  {r['id']}")
    print(f"    LHS: {r['initial']['lhs'][:100]}")
    print(f"    RHS: {r['initial']['rhs'][:100]}")
    print(f"    target: {r['canonical_target']['lhs']} = {r['canonical_target']['rhs']}")
