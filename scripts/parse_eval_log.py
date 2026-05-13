"""Parse an evaluate.py log (`eval_full.log` style) into a CSV recoverable after a crash.

Each per-problem INFO line is parsed and joined with the problem metadata
loaded from `loader.load_hard_evaluation_set` + `load_phase0_problems`,
so the family/source/baseline_astar_nodes columns match the original
evaluate.py CSV schema.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from ggmr.problems.loader import load_hard_evaluation_set, load_phase0_problems

LINE_RE = re.compile(
    r"\[(?P<i>\d+)/(?P<n>\d+)\]\s+(?P<id>\S+):\s+"
    r"hand=(?P<hf>[YN])/\s*(?P<hn>\d+)\s+"
    r"learned=(?P<lf>[YN])/\s*(?P<ln>\d+)"
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--problems", default="all", choices=["all", "hard_v2", "phase0"])
    args = p.parse_args()

    problems = []
    if args.problems in ("hard_v2", "all"):
        problems.extend(load_hard_evaluation_set())
    if args.problems in ("phase0", "all"):
        problems.extend(load_phase0_problems())
    by_id = {pr.id: pr for pr in problems}

    rows: list[dict] = []
    with open(args.log, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            pid = m["id"]
            prob = by_id.get(pid)
            if prob is None:
                continue
            rows.append({
                "id": pid,
                "family": prob.family,
                "source": prob.source,
                "hand_found": m["hf"] == "Y",
                "hand_nodes": int(m["hn"]),
                "hand_time_ms": 0.0,
                "learned_found": m["lf"] == "Y",
                "learned_nodes": int(m["ln"]),
                "learned_time_ms": 0.0,
            })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"parsed {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
