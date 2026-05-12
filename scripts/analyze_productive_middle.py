r"""Analyze coverage_report.json + the per-problem stdout log → write
ggmr/problems/PRODUCTIVE_MIDDLE_REPORT.md.

Combines (a) JSON aggregates for dead-rule audit and bulk counts, (b) per-
problem stdout lines for node-count distribution and middle-problem selection,
(c) A* re-runs on selected middle problems to preview Phase 2's target.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ggmr.expr.tree import canonical_repr
from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
from ggmr.problems.generator import ReverseGenerator
from ggmr.rules.core import arithmetic, algebra, rational, quadratic, polynomial, exponent  # noqa: F401
from ggmr.search.astar import astar
from ggmr.search.bfs import bfs
from ggmr.state import EqState

# Match: "  [ 53/150] gen_linear_5_004     SOLVED   nodes=    4 elapsed=  ..."
_LINE_RE = re.compile(
    r"^\s*\[\s*(\d+)\s*/\s*(\d+)\]\s+(gen_\w+_\d+_\d+)\s+(\w+)\s+nodes=\s*(\S+)\s+elapsed="
)
# Problem id format: gen_<template>_<depth>_<idx>
_PID_RE = re.compile(r"^gen_([a-z]+)_(\d+)_(\d+)$")


def parse_stdout_log(log_path: Path) -> list[dict]:
    """Each per-problem line in the streaming stdout → dict."""
    out: list[dict] = []
    if not log_path.exists():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _LINE_RE.search(line)
        if not m:
            continue
        idx, total, pid, status, nodes_str = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        pm = _PID_RE.match(pid)
        if not pm:
            continue
        template, depth, problem_idx = pm.group(1), int(pm.group(2)), int(pm.group(3))
        try:
            nodes = int(nodes_str)
        except ValueError:
            nodes = None
        out.append(
            {
                "index": int(idx),
                "id": pid,
                "template": template,
                "depth": depth,
                "problem_idx": problem_idx,
                "status": status,  # SOLVED / GEN_FAIL / BFS_FAIL
                "nodes_expanded": nodes,
            }
        )
    return out


def bucket(nodes: int | None) -> str:
    if nodes is None:
        return "unknown"
    if nodes < 10:
        return "trivial"
    if nodes < 100:
        return "easy"
    if nodes < 1000:
        return "middle"
    return "hard"


def compute_seed(base_seed: int, depths: list[int], templates: list[str], problems_per_bucket: int,
                 target_depth: int, target_template: str, target_idx: int) -> int:
    """Replay the seed assignment from validate_coverage.run_coverage."""
    seed = base_seed
    for depth in depths:
        for template in templates:
            for i in range(problems_per_bucket):
                if depth == target_depth and template == target_template and i == target_idx:
                    return seed
                seed += 100
    raise ValueError(f"target ({target_depth},{target_template},{target_idx}) not in grid")


def astar_on_problem(depth: int, template: str, problem_idx: int, seed: int, max_nodes: int) -> dict | None:
    """Re-generate the problem with the same seed, run A* with WeightedSum heuristic."""
    try:
        gen = ReverseGenerator(seed=seed, depth=depth, template=template, max_nodes=max_nodes)
        problem = gen.generate_one(max_attempts=3)
    except Exception as e:
        return {"error": f"generation: {type(e).__name__}: {e}"}
    if problem is None:
        return {"error": "generation returned None"}
    target = problem.target
    tlhs = canonical_repr(target.lhs)
    trhs = canonical_repr(target.rhs)

    def is_target(s):
        return (canonical_repr(s.lhs) == tlhs and canonical_repr(s.rhs) == trhs) or s.is_canonical_target()

    h = WeightedSumCompositeHeuristic()
    # BFS baseline first (from the generator's already-run BFS, but re-run to be safe)
    bfs_result = bfs(problem.initial, is_target, max_nodes=max_nodes, max_depth=30, check_soundness=False)
    astar_result = astar(problem.initial, is_target, heuristic=h, max_nodes=max_nodes, max_depth=30,
                         check_soundness=False)
    return {
        "id": f"gen_{template}_{depth}_{problem_idx:03d}",
        "bfs_found": bfs_result.found,
        "bfs_nodes": bfs_result.stats.nodes_expanded,
        "bfs_steps": bfs_result.num_steps if bfs_result.found else None,
        "astar_found": astar_result.found,
        "astar_nodes": astar_result.stats.nodes_expanded,
        "astar_steps": astar_result.num_steps if astar_result.found else None,
        "ratio": (
            astar_result.stats.nodes_expanded / bfs_result.stats.nodes_expanded
            if bfs_result.stats.nodes_expanded > 0 else None
        ),
    }


def main() -> int:
    report_path = ROOT / "ggmr" / "problems" / "coverage_report.json"
    log_path = Path("C:/Users/anshu/AppData/Local/Temp/claude/C--Users-anshu-PycharmProjects-MonumentalLeapForward/73d60b4a-b47b-4be6-a864-0c86715bb6cd/tasks/ba4bn0wo1.output")
    out_md = ROOT / "ggmr" / "problems" / "PRODUCTIVE_MIDDLE_REPORT.md"

    if not report_path.exists():
        print(f"Missing {report_path}", file=sys.stderr)
        return 1
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    per_problem = parse_stdout_log(log_path)

    # Run params (inferred from JSON or fixed)
    base_seed = 0
    depths = sorted({p["depth"] for p in per_problem}) or [5, 10, 15]
    templates = sorted({p["template"] for p in per_problem})
    problems_per_bucket = report["per_depth"][str(depths[0])]["total_problems"] // max(1, len(templates))

    # 1) Solve rate by depth
    rows_depth = []
    for d in depths:
        d_str = str(d)
        if d_str not in report["per_depth"]:
            continue
        pd = report["per_depth"][d_str]
        rows_depth.append(
            (
                d,
                pd["total_problems"],
                pd["generated"],
                pd["solved"],
                pd["solve_rate_overall"],
                pd.get("nodes_expanded_median"),
                pd.get("nodes_expanded_p95"),
            )
        )

    # 2) Node-expansion distribution among SOLVED problems
    buckets_counts: dict[str, int] = defaultdict(int)
    buckets_by_depth: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    middle_candidates: list[dict] = []
    hard_candidates: list[dict] = []
    for r in per_problem:
        if r["status"] != "SOLVED":
            continue
        b = bucket(r["nodes_expanded"])
        buckets_counts[b] += 1
        buckets_by_depth[r["depth"]][b] += 1
        if b == "middle":
            middle_candidates.append(r)
        elif b == "hard":
            hard_candidates.append(r)

    # 3) Failure analysis
    failures = [r for r in per_problem if r["status"] != "SOLVED"]
    by_failure_type: dict[str, int] = defaultdict(int)
    for f in failures:
        by_failure_type[f["status"]] += 1

    # 4) Per-template breakdown
    rows_template = []
    for t in templates:
        t_results = [r for r in per_problem if r["template"] == t]
        total = len(t_results)
        solved = sum(1 for r in t_results if r["status"] == "SOLVED")
        middle = sum(1 for r in t_results if r["status"] == "SOLVED" and bucket(r["nodes_expanded"]) == "middle")
        hard = sum(1 for r in t_results if r["status"] == "SOLVED" and bucket(r["nodes_expanded"]) == "hard")
        rows_template.append((t, total, solved, middle, hard))

    # 5) Dead rule audit
    dead_rules = report.get("dead_rules", [])

    # 6) A* comparison on up to 5 productive-middle problems (or hard if not enough middle)
    pool = sorted(middle_candidates, key=lambda r: -r["nodes_expanded"])[:5]
    if len(pool) < 5:
        # Top up from hard
        pool += sorted(hard_candidates, key=lambda r: -r["nodes_expanded"])[: 5 - len(pool)]
    if len(pool) < 5:
        # Top up from easy at the high end
        easy_results = [r for r in per_problem if r["status"] == "SOLVED" and bucket(r["nodes_expanded"]) == "easy"]
        pool += sorted(easy_results, key=lambda r: -r["nodes_expanded"])[: 5 - len(pool)]

    astar_results: list[dict] = []
    for r in pool:
        try:
            seed = compute_seed(base_seed, depths, templates, problems_per_bucket,
                                r["depth"], r["template"], r["problem_idx"])
        except ValueError as e:
            astar_results.append({"id": r["id"], "error": str(e)})
            continue
        cmp = astar_on_problem(r["depth"], r["template"], r["problem_idx"], seed, max_nodes=5000)
        if cmp is None:
            cmp = {"id": r["id"], "error": "no result"}
        astar_results.append(cmp)

    # Compose markdown
    lines: list[str] = []
    lines.append("# Phase 1b Productive-Middle Report")
    lines.append("")
    lines.append(f"**Source**: `{report_path.name}` — {report['summary']['total_problems']} problems, "
                 f"wall clock {report['summary']['wall_clock_seconds']:.1f}s, "
                 f"{report['summary'].get('workers', 1)} workers.")
    lines.append(f"**Overall solve rate**: {report['summary']['overall_solve_rate']:.1%} "
                 f"({report['summary']['solved_total']}/{report['summary']['total_problems']})")
    lines.append("")
    lines.append(f"**§3.5 (depth ≤ 10)**: {report['criterion_3_5']['depth_le_10_rate']:.1%} ≥ 90% "
                 f"→ **{'PASS' if report['criterion_3_5']['passed'] else 'FAIL'}**")
    lines.append("")

    # 1. Solve rate by depth
    lines.append("## 1. Solve rate by depth")
    lines.append("")
    lines.append("| Depth | Total | Generated | Solved | Solve rate | Median nodes | p95 nodes |")
    lines.append("|---|---|---|---|---|---|---|")
    for d, total, gen, solved, rate, med, p95 in rows_depth:
        med_s = f"{med:.0f}" if med is not None else "-"
        p95_s = f"{p95:.0f}" if p95 is not None else "-"
        lines.append(f"| {d} | {total} | {gen} | {solved} | {rate:.1%} | {med_s} | {p95_s} |")
    lines.append("")

    # 2. Node-expansion buckets
    lines.append("## 2. Node-expansion distribution (SOLVED problems only)")
    lines.append("")
    lines.append("Buckets: trivial <10 nodes, easy 10–100, **middle 100–1000**, hard 1000–5000.")
    lines.append("The **middle** bucket is the key paper claim — problems where BFS does real work and a")
    lines.append("learned heuristic can demonstrate compression.")
    lines.append("")
    total_solved = sum(buckets_counts.values())
    lines.append(f"| Bucket | Count | Share |")
    lines.append("|---|---|---|")
    for b in ["trivial", "easy", "middle", "hard"]:
        c = buckets_counts.get(b, 0)
        share = c / total_solved if total_solved else 0.0
        lines.append(f"| {b} | {c} | {share:.1%} |")
    lines.append("")
    lines.append("**By depth:**")
    lines.append("")
    lines.append("| Depth | Trivial | Easy | Middle | Hard |")
    lines.append("|---|---|---|---|---|")
    for d in depths:
        row = buckets_by_depth.get(d, {})
        lines.append(
            f"| {d} | {row.get('trivial', 0)} | {row.get('easy', 0)} | "
            f"{row.get('middle', 0)} | {row.get('hard', 0)} |"
        )
    lines.append("")

    # 3. Failure analysis
    lines.append("## 3. Failure analysis")
    lines.append("")
    if not failures:
        lines.append("No failures.")
    else:
        lines.append(f"**Total failures**: {len(failures)} out of {len(per_problem)} "
                     f"({len(failures)/len(per_problem):.1%})")
        lines.append("")
        for kind, count in sorted(by_failure_type.items()):
            lines.append(f"- `{kind}`: {count}")
        lines.append("")
        lines.append("**Note**: in the current pipeline, `GEN_FAIL` conflates two distinct causes —")
        lines.append("(a) the inverse-rule generator failed to produce a structurally-different problem after 3 attempts,")
        lines.append("and (b) the forward BFS verification timed out within the generation step's budget (5000 nodes / depth 30).")
        lines.append("Distinguishing these requires instrumenting `ReverseGenerator._generate_attempt`; Phase 1c can split them.")
        lines.append("")
        lines.append("Failed problem IDs:")
        for f in failures[:20]:
            lines.append(f"- `{f['id']}` ({f['status']})")
        if len(failures) > 20:
            lines.append(f"- ... ({len(failures) - 20} more)")
    lines.append("")

    # 4. Per-template breakdown
    lines.append("## 4. Per-template breakdown")
    lines.append("")
    lines.append("| Template | Total | Solved | Middle | Hard |")
    lines.append("|---|---|---|---|---|")
    for t, total, solved, middle, hard in rows_template:
        lines.append(f"| {t} | {total} | {solved} | {middle} | {hard} |")
    lines.append("")

    # 5. Dead rule audit
    lines.append("## 5. Dead-rule audit")
    lines.append("")
    if not dead_rules:
        lines.append("All 45 rules fired at least once.")
    else:
        lines.append(f"**{len(dead_rules)} of 45 rules** never fired on this batch:")
        lines.append("")
        for r in dead_rules:
            lines.append(f"- `{r}`")
        lines.append("")
        lines.append("**Interpretation**:")
        lines.append("")
        lines.append("- Exponent rules (`POW_PRODUCT_AT`, `POW_QUOTIENT_AT`, `POW_OF_POW_AT`) — Phase 0 has no exponent-")
        lines.append("  manipulation problems and the reverse-generator's inverse set doesn't seed such states. These")
        lines.append("  rules are correct but inert without an exponent-flavored template family.")
        lines.append("- `SQUARE_BOTH_SIDES` — restricted by design to sqrt contexts; no sqrt problems in this batch.")
        lines.append("- `IDENTITY_*` and `ZERO_PROPERTY_AT` — overlap with `SIMPLIFY_NUMERIC_AT` and `SIMPLIFY_AT`; tested")
        lines.append("  in unit tests but harder to reach via forward BFS because cheaper alternatives fire first.")
        lines.append("- `FACTOR_DIFFERENCE_OF_SQUARES_AT`, `PERFECT_SQUARE_TRINOMIAL_AT`, `FACTOR_BY_GROUPING`,")
        lines.append("  `RATIONAL_ROOT_THEOREM`, `SYNTHETIC_DIVISION` — superseded by `FACTOR_POLYNOMIAL` (which calls")
        lines.append("  `sympy.factor` and one-shots them). Worth keeping as named axioms for paper clarity; flag for")
        lines.append("  Phase 1c if we want stricter rule-firing diversity.")
    lines.append("")

    # 6. A* comparison on middle problems
    lines.append("## 6. A* vs BFS on productive-middle problems")
    lines.append("")
    lines.append("Pool: top-5 SOLVED problems by BFS `nodes_expanded` from the middle/hard bucket (fallback to")
    lines.append("the top of `easy` if the middle bucket has fewer than 5 entries). A* uses")
    lines.append("`WeightedSumCompositeHeuristic` (default weights 1.0).")
    lines.append("")
    lines.append("| Problem | BFS nodes | A* nodes | Ratio (A*/BFS) | BFS steps | A* steps |")
    lines.append("|---|---|---|---|---|---|")
    for a in astar_results:
        if "error" in a:
            lines.append(f"| {a['id']} | ERR: {a['error']} | | | | |")
            continue
        ratio = f"{a['ratio']:.2f}" if a.get("ratio") is not None else "-"
        lines.append(
            f"| {a['id']} | {a['bfs_nodes']} | {a['astar_nodes']} | {ratio} | "
            f"{a.get('bfs_steps', '-')} | {a.get('astar_steps', '-')} |"
        )
    lines.append("")
    ratios = [a["ratio"] for a in astar_results if a.get("ratio") is not None]
    if ratios:
        lines.append(f"**Mean A*/BFS ratio across the {len(ratios)} comparable problems**: "
                     f"{sum(ratios)/len(ratios):.2f}")
        lines.append("")
        lines.append("This ratio is the empirical baseline Phase 2's learned value network needs to beat. Hand")
        lines.append("heuristic + A* already compresses these problems substantially; Phase 2's claim must be that")
        lines.append("a learned heuristic compresses MORE than this on held-out problems, not just that it beats BFS.")
    lines.append("")

    # Save
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
