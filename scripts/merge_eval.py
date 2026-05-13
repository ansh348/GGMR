"""Merge PHASE2_RESULTS_part1.csv + part2.csv -> PHASE2_RESULTS.md + .csv.

Recomputes all aggregate metrics across the union.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PART1 = Path("ggmr/training/PHASE2_RESULTS_part1.csv")
PART2 = Path("ggmr/training/PHASE2_RESULTS_part2.csv")
OUT_MD = Path("ggmr/training/PHASE2_RESULTS.md")
OUT_CSV = Path("ggmr/training/PHASE2_RESULTS.csv")


def _read(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["hand_nodes"] = int(r["hand_nodes"])
        r["learned_nodes"] = int(r["learned_nodes"])
        r["hand_time_ms"] = float(r.get("hand_time_ms", 0) or 0)
        r["learned_time_ms"] = float(r.get("learned_time_ms", 0) or 0)
        r["hand_found"] = str(r["hand_found"]).lower() == "true"
        r["learned_found"] = str(r["learned_found"]).lower() == "true"
    return rows


def main() -> int:
    rows = _read(PART1) + _read(PART2)
    seen = set()
    dedup = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        dedup.append(r)
    rows = dedup
    print(f"merged {len(rows)} unique problems")

    # Buckets
    joint = [r for r in rows if r["hand_found"] and r["learned_found"]]
    hand_only = [r for r in rows if r["hand_found"] and not r["learned_found"]]
    learned_only = [r for r in rows if not r["hand_found"] and r["learned_found"]]
    both_fail = [r for r in rows if not r["hand_found"] and not r["learned_found"]]

    # Sub-cohort: hard only
    hard = [r for r in rows if r["source"] == "hard"]
    hard_joint = [r for r in hard if r["hand_found"] and r["learned_found"]]
    phase0 = [r for r in rows if r["source"] == "phase0"]
    phase0_joint = [r for r in phase0 if r["hand_found"] and r["learned_found"]]

    def _geomean(ratios: list[float]) -> float:
        logs = [np.log(x) for x in ratios if x > 0]
        return float(np.exp(np.mean(logs))) if logs else 0.0

    all_ratios = [r["hand_nodes"] / max(r["learned_nodes"], 1) for r in joint]
    hard_ratios = [r["hand_nodes"] / max(r["learned_nodes"], 1) for r in hard_joint]
    phase0_ratios = [r["hand_nodes"] / max(r["learned_nodes"], 1) for r in phase0_joint]

    by_family: dict[str, list[float]] = defaultdict(list)
    for r in joint:
        by_family[r["family"]].append(r["hand_nodes"] / max(r["learned_nodes"], 1))

    # ---- Write merged CSV ----
    fields = ["id", "family", "source", "hand_found", "hand_nodes", "hand_time_ms",
              "learned_found", "learned_nodes", "learned_time_ms"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # ---- Write merged Markdown ----
    lines: list[str] = []
    lines.append("# Phase 2 Evaluation Results (FULL)\n")
    lines.append("Checkpoint: `checkpoints/full/best.pt`")
    lines.append("Train data: 23,403 rows. GPU: RTX 3050. Eval: 50 hard_v2 + 20 phase0.\n")
    lines.append("## Aggregate (all 70)\n")
    lines.append(f"- Total problems: {len(rows)}")
    lines.append(f"- Joint solved (hand AND learned): **{len(joint)}/{len(rows)}**")
    lines.append(f"- Hand-only solved: {len(hand_only)}  (regressions: " +
                 ", ".join(r['id'] for r in hand_only) + ")")
    lines.append(f"- Learned-only solved: {len(learned_only)}")
    lines.append(f"- Both failed: {len(both_fail)}")
    lines.append(f"- **Geomean compression (joint)**: {_geomean(all_ratios):.3f}x")
    lines.append(f"- Median compression (joint): {float(np.median(all_ratios)) if all_ratios else 0:.3f}x")
    if all_ratios:
        lines.append(f"- Max compression: {max(all_ratios):.0f}x")
        lines.append(f"- Min compression: {min(all_ratios):.3f}x")
    lines.append("")
    lines.append("## Aggregate (50 hard_v2 only)\n")
    lines.append(f"- Joint solved: **{len(hard_joint)}/{len(hard)}**")
    lines.append(f"- **Geomean compression**: {_geomean(hard_ratios):.3f}x")
    lines.append(f"- Median: {float(np.median(hard_ratios)) if hard_ratios else 0:.3f}x")
    lines.append(f"- Total hand nodes: {sum(r['hand_nodes'] for r in hard_joint)}")
    lines.append(f"- Total learned nodes: {sum(r['learned_nodes'] for r in hard_joint)}")
    if hard_ratios:
        lines.append(f"- Total ratio: {sum(r['hand_nodes'] for r in hard_joint) / max(sum(r['learned_nodes'] for r in hard_joint), 1):.2f}x")
    lines.append("")
    lines.append("## Aggregate (20 phase0 only — regression set)\n")
    lines.append(f"- Joint solved: **{len(phase0_joint)}/{len(phase0)}**")
    lines.append(f"- Regressions: {[r['id'] for r in hand_only if r['source'] == 'phase0']}")
    lines.append(f"- Geomean: {_geomean(phase0_ratios):.3f}x")
    lines.append(f"- Median: {float(np.median(phase0_ratios)) if phase0_ratios else 0:.3f}x")
    lines.append("")
    lines.append("## Per-family geomean (joint-solved)\n")
    lines.append("| family | n | geomean | median | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for fam in sorted(by_family):
        ratios = by_family[fam]
        lines.append(
            f"| {fam} | {len(ratios)} | {_geomean(ratios):.3f}x | "
            f"{float(np.median(ratios)):.3f}x | {min(ratios):.3f}x | {max(ratios):.0f}x |"
        )
    lines.append("")
    lines.append("## Per-problem (sorted by compression ratio, descending)\n")
    lines.append("| id | family | source | hand | hand_nodes | learned | learned_nodes | ratio |")
    lines.append("|---|---|---|---|---|---|---|---|")
    rows_sorted = sorted(
        rows,
        key=lambda r: -(r["hand_nodes"] / max(r["learned_nodes"], 1))
        if r["hand_found"] and r["learned_found"] else 1e9,
    )
    for r in rows_sorted:
        ratio = (
            f"{r['hand_nodes'] / max(r['learned_nodes'], 1):.2f}x"
            if r["hand_found"] and r["learned_found"]
            else "-"
        )
        lines.append(
            f"| {r['id']} | {r['family']} | {r['source']} | "
            f"{'Y' if r['hand_found'] else 'N'} | {r['hand_nodes']} | "
            f"{'Y' if r['learned_found'] else 'N'} | {r['learned_nodes']} | {ratio} |"
        )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {OUT_MD}")
    print(f"-> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
