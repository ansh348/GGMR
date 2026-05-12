"""Extract the 9 v2 motif candidates that timed out at 5k BFS / 120s budget.

Re-run them with a much higher BFS budget and 30-min per-problem timeout to
distinguish productive-middle (BFS-solvable, A* hard) from truly pathological.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

TIMEOUT_IDS = {
    "motif_v2_000_L1_triple_twin_shield",
    "motif_v2_001_L2_cross_reciprocal",
    "motif_v2_002_L3_distributed_scalar_gate",
    "motif_v2_008_P2_one_sided_denom_cubic",
    "motif_v2_009_P3_irreducible_quadratic_disguise",
    "motif_v2_010_P4_cubic_two_denom_cross_ratio",
    "motif_v2_011_R1_fractional_difference_hiding",
    "motif_v2_012_R2_distributed_fractional",
    "motif_v2_013_R3_cross_reciprocal_fractional",
}

with open(ROOT / "ggmr" / "problems" / "motif_candidates_v2.yaml", "r", encoding="utf-8") as f:
    records = yaml.safe_load(f)

subset = [r for r in records if r["id"] in TIMEOUT_IDS]

out_path = ROOT / "ggmr" / "problems" / "motif_candidates_v2_timeouts.yaml"
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(subset, f, sort_keys=False, allow_unicode=True)

print(f"Wrote {len(subset)}/{len(records)} problems to {out_path}")
for r in subset:
    print(f"  {r['id']}")
