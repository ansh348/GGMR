"""Verify every registered rule fires at least once across the generated dataset.

Reads training_data_round2.jsonl and re-runs BFS on each problem's initial state
to recover the action sequence. Counts rule firings. Reports any rule with 0
firings — those are coverage gaps in the category design.

Usage:
    python scripts/verify_round2_rule_coverage.py \
        --data /data/training_data_round2.jsonl \
        --sample 2000   # sample N problems for speed (full set is huge)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ggmr.rules.core  # noqa: F401  (registers all rules)
from ggmr.rules.registry import default_registry
from ggmr.search.bfs import bfs
from ggmr.state import EqState
from ggmr.training.extract_pairs import _build_is_target
from ggmr.training.srepr_parse import parse_srepr


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, default="/data/training_data_round2.jsonl")
    p.add_argument("--sample", type=int, default=2000,
                   help="Sample N unique problem_ids (0 = all)")
    p.add_argument("--max-nodes", type=int, default=15_000)
    args = p.parse_args()

    rules = default_registry.names()
    rule_firings: Counter = Counter()

    # Group records by problem_id (we only need initial state, target state)
    by_pid: dict[str, list[dict]] = {}
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("problem_id", "")
            by_pid.setdefault(pid, []).append(rec)

    pids = list(by_pid.keys())
    if args.sample > 0 and len(pids) > args.sample:
        pids = random.Random(0).sample(pids, args.sample)
    print(f"Sampling {len(pids)} problems...", flush=True)

    solved = 0
    for i, pid in enumerate(pids):
        recs = by_pid[pid]
        # Order by remaining_steps descending — first row is initial, last is target
        recs_sorted = sorted(recs, key=lambda r: -r["remaining_steps"])
        first, last = recs_sorted[0], recs_sorted[-1]
        try:
            initial_lhs = parse_srepr(first["state_lhs_srepr"])
            initial_rhs = parse_srepr(first["state_rhs_srepr"])
            target_lhs = parse_srepr(last["state_lhs_srepr"])
            target_rhs = parse_srepr(last["state_rhs_srepr"])
        except Exception:
            continue
        import sympy as sp
        var = sp.Symbol(first["var"])
        initial = EqState(lhs=initial_lhs, rhs=initial_rhs, var=var)
        target = EqState(lhs=target_lhs, rhs=target_rhs, var=var)
        try:
            result = bfs(
                initial,
                _build_is_target(target),
                max_nodes=args.max_nodes,
                max_depth=30,
                check_soundness=False,
                problem_id=pid,
            )
        except Exception:
            continue
        if result.found:
            solved += 1
            for _state, action in result.path:
                rule_firings[action.rule_name] += 1
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(pids)}] solved={solved} unique_rules_fired={len(rule_firings)}",
                  flush=True)

    print()
    print(f"Solved {solved}/{len(pids)} problems")
    print(f"Total rule firings: {sum(rule_firings.values())}")
    print(f"Unique rules fired: {len(rule_firings)}/{len(rules)}")
    missing = [r for r in rules if r not in rule_firings]
    if missing:
        print(f"Rules with ZERO firings ({len(missing)}):")
        for r in missing:
            print(f"  - {r}")
    print()
    print("Top 20 by firing count:")
    for rule, count in rule_firings.most_common(20):
        print(f"  {rule}: {count}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
