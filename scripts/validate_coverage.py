r"""Coverage validation entry point: generate 500 problems and run BFS on each.

Produces a JSON report (per-depth + per-template solve rates, productive-middle
identification, per-rule application frequency).

§3.5 of `ggmr/PHASE1B_PREREG.md`: ≥ 90% solve rate at depth ≤ 10.

Usage (PowerShell):
    & .\.venv\Scripts\python.exe scripts\validate_coverage.py `
        --depths 5 10 15 20 `
        --templates linear quadratic rational polynomial mixed `
        --problems-per-bucket 25 `
        --max-nodes 5000 `
        --workers 8 `
        --output ggmr\problems\coverage_report.json

Pass `--workers N` to parallelize generation+BFS across N processes (default:
cpu_count - 1). Each problem is independent (unique seed, isolated BFS), so the
work is embarrassingly parallel. Note: each worker re-imports the rule registry
on startup (~1-2s per worker, one-time cost).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Allow running as a script from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run_one_problem(args: dict) -> dict:
    """Worker function: generate one problem and reuse its forward-BFS stats.

    Must be module-level so Windows `spawn` can pickle it. Each subprocess
    imports `ggmr.rules.core` fresh, populating its own `default_registry`.

    Optimization vs the original sequential flow: the `ReverseGenerator` already
    runs forward BFS during verification. We reuse those stats rather than re-
    running BFS in the main loop. Saves ~50% of the BFS work per problem.
    """
    # Import inside worker so each subprocess loads the registry independently.
    # (sys.path was extended in the module preamble above, which re-executes
    # on `spawn`.) Importing the package executes `__init__.py` which imports
    # every rule file, populating the registry as a side effect.
    import ggmr.rules.core  # noqa: F401  (registers rules)
    from ggmr.problems.generator import GeneratedProblem, ReverseGenerator

    depth = args["depth"]
    template = args["template"]
    problem_idx = args["problem_idx"]
    seed = args["seed"]
    max_nodes = args["max_nodes"]
    pid = f"gen_{template}_{depth}_{problem_idx:03d}"

    try:
        gen = ReverseGenerator(
            seed=seed,
            depth=depth,
            template=template,
            max_nodes=max_nodes,
        )
        problem: Optional[GeneratedProblem] = gen.generate_one(max_attempts=3)
        if problem is None:
            return {
                "id": pid,
                "depth": depth,
                "template": template,
                "problem_idx": problem_idx,
                "generated": False,
                "solved": False,
                "nodes_expanded": None,
                "time_ms": None,
                "path_length": None,
                "rule_counts": {},
            }
        # The generator already verified via forward BFS — reuse its stats.
        stats = problem.bfs_stats
        return {
            "id": pid,
            "depth": depth,
            "template": template,
            "problem_idx": problem_idx,
            "generated": True,
            "solved": True,
            "nodes_expanded": stats.get("nodes_expanded"),
            "time_ms": round(stats.get("time_ms", 0.0), 1),
            "path_length": len(problem.forward_trace),
            "rule_counts": dict(stats.get("rule_application_counts", {})),
        }
    except Exception as e:
        return {
            "id": pid,
            "depth": depth,
            "template": template,
            "problem_idx": problem_idx,
            "generated": False,
            "solved": False,
            "nodes_expanded": None,
            "time_ms": None,
            "path_length": None,
            "rule_counts": {},
            "error": f"{type(e).__name__}: {e}",
        }


def _print_progress(done: int, total: int, r: dict, t_start: float) -> None:
    elapsed = time.perf_counter() - t_start
    eta = (elapsed / done) * (total - done) if done > 0 else 0
    if r["solved"]:
        status = "SOLVED"
    elif not r["generated"]:
        status = "GEN_FAIL"
    else:
        status = "BFS_FAIL"
    print(
        f"  [{done:4d}/{total}] {r['id']:35s} {status:8s} "
        f"nodes={r.get('nodes_expanded')!s:>6s} "
        f"elapsed={elapsed:6.1f}s eta={eta:6.0f}s",
        flush=True,
    )


def run_coverage(
    depths: list[int],
    templates: list[str],
    problems_per_bucket: int,
    max_nodes: int,
    base_seed: int = 0,
    workers: Optional[int] = None,
    progress: bool = False,
) -> dict:
    """Generate problems × forward BFS × aggregate stats.

    Args:
        workers: number of parallel processes. None or 1 = sequential. Defaults
            to None for backwards compatibility (no parallelization unless asked).
    """
    # Build deterministic job list (preserves the seed assignment of the original
    # sequential version: seed_counter increments by 100 per problem in fixed order).
    jobs: list[dict] = []
    seed_counter = base_seed
    for depth in depths:
        for template in templates:
            for problem_idx in range(problems_per_bucket):
                jobs.append(
                    {
                        "depth": depth,
                        "template": template,
                        "problem_idx": problem_idx,
                        "seed": seed_counter,
                        "max_nodes": max_nodes,
                    }
                )
                seed_counter += 100

    t_start = time.perf_counter()
    results: list[dict] = []

    if not workers or workers <= 1:
        # Sequential path (preserves original behavior; tests rely on this).
        for i, job in enumerate(jobs):
            r = _run_one_problem(job)
            results.append(r)
            if progress:
                _print_progress(i + 1, len(jobs), r, t_start)
    else:
        # Parallel path: imap_unordered yields as workers finish.
        with multiprocessing.Pool(processes=workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_one_problem, jobs)):
                results.append(r)
                if progress:
                    _print_progress(i + 1, len(jobs), r, t_start)

    t_total = time.perf_counter() - t_start

    # Aggregate rule counts
    rule_counts: dict[str, int] = defaultdict(int)
    for r in results:
        for rname, count in r.get("rule_counts", {}).items():
            rule_counts[rname] += count

    solved_total = sum(1 for r in results if r["solved"])
    successful_generations = sum(1 for r in results if r["generated"])

    # per_depth
    per_depth: dict[int, dict] = {}
    for depth in depths:
        depth_results = [r for r in results if r["depth"] == depth]
        gen_count = sum(1 for r in depth_results if r["generated"])
        solv_count = sum(1 for r in depth_results if r["solved"])
        node_dist = [
            r["nodes_expanded"]
            for r in depth_results
            if r["solved"] and r["nodes_expanded"] is not None
        ]
        per_depth[depth] = {
            "total_problems": len(depth_results),
            "generated": gen_count,
            "solved": solv_count,
            "solve_rate_among_generated": solv_count / max(1, gen_count),
            "solve_rate_overall": solv_count / max(1, len(depth_results)),
            "nodes_expanded_median": (
                statistics.median(node_dist) if node_dist else None
            ),
            "nodes_expanded_p95": (
                statistics.quantiles(node_dist, n=20)[18]
                if len(node_dist) >= 5
                else None
            ),
            "productive_middle_count": sum(
                1 for n in node_dist if 100 < n < 50_000
            ),
        }

    # per_template
    per_template: dict[str, dict] = {}
    for template in templates:
        tmpl_results = [r for r in results if r["template"] == template]
        gen_count = sum(1 for r in tmpl_results if r["generated"])
        solv_count = sum(1 for r in tmpl_results if r["solved"])
        per_template[template] = {
            "total_problems": len(tmpl_results),
            "generated": gen_count,
            "solved": solv_count,
            "solve_rate_among_generated": solv_count / max(1, gen_count),
        }

    # §3.5 criterion: depth ≤ 10 (i.e., depths 5 and 10 if present)
    depth_le10 = [d for d in depths if d <= 10]
    depth_le10_total = sum(per_depth[d]["total_problems"] for d in depth_le10)
    depth_le10_solved = sum(per_depth[d]["solved"] for d in depth_le10)
    depth_le10_rate = depth_le10_solved / max(1, depth_le10_total)

    # Dead rules: registered rules with zero applications across the batch.
    # Import here (in the main process) so we get the same registry as the workers.
    import ggmr.rules.core  # noqa: F401  (registers rules)
    from ggmr.rules.registry import default_registry

    all_rule_names = set(default_registry.names())
    dead_rules = sorted(all_rule_names - set(rule_counts.keys()))

    return {
        "summary": {
            "total_problems": len(results),
            "successful_generations": successful_generations,
            "solved_total": solved_total,
            "overall_solve_rate": solved_total / max(1, len(results)),
            "wall_clock_seconds": round(t_total, 1),
            "workers": workers or 1,
        },
        "criterion_3_5": {
            "depth_le_10_total": depth_le10_total,
            "depth_le_10_solved": depth_le10_solved,
            "depth_le_10_rate": depth_le10_rate,
            "threshold": 0.90,
            "passed": depth_le10_rate >= 0.90,
        },
        "per_depth": per_depth,
        "per_template": per_template,
        "rule_application_counts": dict(rule_counts),
        "dead_rules": dead_rules,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate problems × run BFS coverage validation."
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument(
        "--templates",
        type=str,
        nargs="+",
        default=["linear", "quadratic", "rational", "polynomial", "mixed"],
    )
    parser.add_argument("--problems-per-bucket", type=int, default=25)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--output", type=str, default="ggmr/problems/coverage_report.json")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: cpu_count - 1; 1 = sequential)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print per-problem progress lines as workers complete",
    )
    args = parser.parse_args(argv)

    if args.workers is None:
        args.workers = max(1, (os.cpu_count() or 2) - 1)

    total_jobs = len(args.depths) * len(args.templates) * args.problems_per_bucket
    print(
        f"Coverage validation: depths={args.depths}, templates={args.templates}, "
        f"problems_per_bucket={args.problems_per_bucket}, max_nodes={args.max_nodes}, "
        f"workers={args.workers}, total_jobs={total_jobs}",
        flush=True,
    )

    report = run_coverage(
        depths=args.depths,
        templates=args.templates,
        problems_per_bucket=args.problems_per_bucket,
        max_nodes=args.max_nodes,
        base_seed=args.base_seed,
        workers=args.workers,
        progress=args.progress,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote {output_path}", flush=True)
    print(f"Summary: {report['summary']}", flush=True)
    print(f"§3.5: {report['criterion_3_5']}", flush=True)
    return 0 if report["criterion_3_5"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
