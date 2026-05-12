"""Build 5 hand-crafted motif candidate equations for heuristic-difficulty validation.

These equations test whether the proposed adversarial motifs (semantic-twin decoy,
LCM gate, complete-square gate, shallow Mobius) defeat the structural-simplicity
heuristic, BEFORE building rule infrastructure.

Writes to ggmr/problems/motif_candidates.yaml in the hard-eval-set schema.
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

# Example 1: linear target x=2, hidden by two semantic twins.
# 2x + 3 + (x+1)(x-4) + (x^2-1)/(x-1) = 7 + (x^2-3x-4) + (x+1)(x-1)/(x-1)
lhs1 = A(
    M(I(2), x),
    I(3),
    M(A(x, I(1)), A(x, I(-4))),
    M(A(P(x, I(2)), I(-1)), P(A(x, I(-1)), I(-1))),
)
rhs1 = A(
    I(7),
    A(P(x, I(2)), M(I(-3), x), I(-4)),
    M(M(A(x, I(1)), A(x, I(-1))), P(A(x, I(-1)), I(-1))),
)
problems.append({
    "name": "linear_two_twins",
    "lhs": lhs1, "rhs": rhs1,
    "excluded": [I(1)],
    "target_lhs": "x", "target_rhs": "2",
    "category": "linear",
})

# Example 2: quadratic hidden behind denominator gate.
# ((x-2)(x-5) + (x+4)(x-1)) / (x+3) = (x^2+3x-4) / (x+3)
lhs2 = M(
    A(
        M(A(x, I(-2)), A(x, I(-5))),
        M(A(x, I(4)), A(x, I(-1))),
    ),
    P(A(x, I(3)), I(-1)),
)
rhs2 = M(
    A(P(x, I(2)), M(I(3), x), I(-4)),
    P(A(x, I(3)), I(-1)),
)
problems.append({
    "name": "quadratic_denom_gate",
    "lhs": lhs2, "rhs": rhs2,
    "excluded": [I(-3)],
    "target_lhs": "(x-2)*(x-5)", "target_rhs": "0",
    "category": "quadratic",
})

# Example 3: shallow Mobius with semantic decoy. Target x=5.
# ((3x-6)(x+1) + 2x + (x+2)(x-3)) / (x+1) = (9(x+1) + 2x + (x^2-x-6)) / (x+1)
lhs3 = M(
    A(
        M(A(M(I(3), x), I(-6)), A(x, I(1))),
        M(I(2), x),
        M(A(x, I(2)), A(x, I(-3))),
    ),
    P(A(x, I(1)), I(-1)),
)
rhs3 = M(
    A(
        M(I(9), A(x, I(1))),
        M(I(2), x),
        A(P(x, I(2)), M(I(-1), x), I(-6)),
    ),
    P(A(x, I(1)), I(-1)),
)
problems.append({
    "name": "mobius_decoy",
    "lhs": lhs3, "rhs": rhs3,
    "excluded": [I(-1)],
    "target_lhs": "x", "target_rhs": "5",
    "category": "rational",
})

# Example 4: two-denominator LCM trap.
# (x-2)(x-5)/(x+1) + (x+3)(x-4)/(x+2) = (x^2-x-12)/(x+2)
lhs4 = A(
    M(M(A(x, I(-2)), A(x, I(-5))), P(A(x, I(1)), I(-1))),
    M(M(A(x, I(3)), A(x, I(-4))), P(A(x, I(2)), I(-1))),
)
rhs4 = M(
    A(P(x, I(2)), M(I(-1), x), I(-12)),
    P(A(x, I(2)), I(-1)),
)
problems.append({
    "name": "two_denom_lcm_trap",
    "lhs": lhs4, "rhs": rhs4,
    "excluded": [I(-1), I(-2)],
    "target_lhs": "(x-2)*(x-5)", "target_rhs": "0",
    "category": "rational",
})

# Example 5: complete-square with rational semantic twin. Target (x+1)(x+7)=0.
# x^2 + 8x + (x+2)(x-3) + (x^2-1)/(x-1) = -7 + (x^2-x-6) + (x+1)(x-1)/(x-1)
lhs5 = A(
    P(x, I(2)),
    M(I(8), x),
    M(A(x, I(2)), A(x, I(-3))),
    M(A(P(x, I(2)), I(-1)), P(A(x, I(-1)), I(-1))),
)
rhs5 = A(
    I(-7),
    A(P(x, I(2)), M(I(-1), x), I(-6)),
    M(M(A(x, I(1)), A(x, I(-1))), P(A(x, I(-1)), I(-1))),
)
problems.append({
    "name": "complete_square_rational",
    "lhs": lhs5, "rhs": rhs5,
    "excluded": [I(1)],
    "target_lhs": "(x+1)*(x+7)", "target_rhs": "0",
    "category": "quadratic",
})


records = []
for i, p in enumerate(problems):
    records.append({
        "id": f"motif_{i:03d}_{p['name']}",
        "category": p["category"],
        "recipe": "hand_crafted_motif",
        "difficulty": "hard",
        "variable": "x",
        "source": "ggmr-motif-validation-v1",
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


out_path = ROOT / "ggmr" / "problems" / "motif_candidates.yaml"
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(records, f, sort_keys=False, allow_unicode=True)

print(f"Wrote {len(records)} problems to {out_path}")
for r in records:
    print(f"  {r['id']}")
    print(f"    LHS: {r['initial']['lhs'][:90]}")
    print(f"    RHS: {r['initial']['rhs'][:90]}")
    print(f"    target: {r['canonical_target']['lhs']} = {r['canonical_target']['rhs']}")
    print(f"    excluded: {r['excluded_srepr']}")
