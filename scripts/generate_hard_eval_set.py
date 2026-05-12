r"""Hard evaluation set generator.

Produces a YAML file of problems where A* with `WeightedSumCompositeHeuristic`
expands at least `--min-astar-nodes` nodes per problem. Stratifies generation
across 4 recipes (nested_rational, complete_square, cross_side_rational,
polynomial_disguised) to ensure category balance in the final set.

Success criteria (defaults match the plan):
  - ≥ 30 of 50 problems have A* nodes ≥ 50
  - ≥ 10 of 50 problems have A* nodes ≥ 200
  - all problems BFS-solvable within `--max-bfs-nodes`
  - 0 unsound transitions (validated separately by validate_hard_set.py)

Usage (PowerShell):
    & .\.venv\Scripts\python.exe scripts\generate_hard_eval_set.py `
        --target 50 `
        --depth 20 `
        --output ggmr\problems\hard_evaluation_set.yaml `
        --workers 8 `
        --progress
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _generate_worker(args: dict) -> dict:
    """Worker: produce one HardProblem (or None) by spawning a subprocess running
    `_generate_one_hard.py`. The subprocess is killed if it exceeds
    `attempt_timeout_s` seconds — essential because some heavily-disguised
    states cause sympy.solve to hang indefinitely.

    Must be top-level so multiprocessing `spawn` can pickle it.
    """
    import json as _json
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    timeout_s = float(args.pop("attempt_timeout_s", 45.0))
    python_exe = args.pop("python_exe", _sys.executable)
    script_path = str(_Path(__file__).resolve().parent / "_generate_one_hard.py")

    cmd = [python_exe, "-u", script_path, "--job-json", _json.dumps(args)]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "recipe": args["recipe"],
            "seed": args["seed"],
            "accepted": False,
            "error": f"TIMEOUT({timeout_s:.0f}s)",
        }
    if completed.returncode != 0:
        return {
            "recipe": args["recipe"],
            "seed": args["seed"],
            "accepted": False,
            "error": f"exit_code={completed.returncode}: {completed.stderr[:200]}",
        }
    try:
        return _json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {
            "recipe": args["recipe"],
            "seed": args["seed"],
            "accepted": False,
            "error": f"parse_error: {type(e).__name__}: {e}; stdout={completed.stdout[:200]}",
        }


def _print_progress(done: int, total: int, r: dict, t_start: float) -> None:
    elapsed = time.perf_counter() - t_start
    eta = (elapsed / done) * (total - done) if done > 0 else 0
    if r["accepted"]:
        astar_n = r["record"].get("astar_nodes_expanded", -1)
        status = f"OK A*={astar_n:>5d}"
    elif "error" in r:
        status = f"ERR {r['error'][:30]}"
    else:
        status = "REJECT"
    print(
        f"  [{done:4d}/{total}] {r['recipe']:24s} seed={r['seed']:>6d} {status:30s} "
        f"elapsed={elapsed:6.1f}s eta={eta:6.0f}s",
        flush=True,
    )


def run_generation(
    target_per_recipe: int,
    depth: int,
    min_astar_nodes: int,
    max_bfs_nodes: int,
    max_bfs_depth: int,
    astar_max_nodes: int,
    astar_max_depth: int,
    max_attempts: int,
    max_seeds_per_recipe: int,
    workers: int,
    base_seed_stride: int,
    progress: bool,
    pre_bfs_complexity_max: int,
    attempt_timeout_s: float,
) -> dict:
    """Generate problems until each recipe has at least `target_per_recipe`
    accepted, or seed budget is exhausted.

    Returns: { "records": [dict, ...], "stats": {...} }
    """
    from ggmr.problems.hard_generator import RECIPES

    # Build interleaved job list so workers process all recipes in parallel.
    # (If listed per-recipe, the first recipe monopolizes all 4 workers and
    # we hit the global-target cap before touching other recipes.)
    jobs: list[dict] = []
    for i in range(max_seeds_per_recipe):
        for recipe in RECIPES:
            seed = i * base_seed_stride
            jobs.append({
                "recipe": recipe.name,
                "seed": seed,
                "depth": depth,
                "min_astar_nodes": min_astar_nodes,
                "max_bfs_nodes": max_bfs_nodes,
                "max_bfs_depth": max_bfs_depth,
                "astar_max_nodes": astar_max_nodes,
                "astar_max_depth": astar_max_depth,
                "max_attempts": max_attempts,
                "pre_bfs_complexity_max": pre_bfs_complexity_max,
                "attempt_timeout_s": attempt_timeout_s,
                "python_exe": sys.executable,
            })

    accepted_records_by_recipe: dict[str, list[dict]] = {r.name: [] for r in RECIPES}
    rejection_counts: dict[str, int] = {r.name: 0 for r in RECIPES}
    error_counts: dict[str, int] = {r.name: 0 for r in RECIPES}
    total_target = target_per_recipe * len(RECIPES)
    t_start = time.perf_counter()
    done = 0

    def _accept(r: dict) -> None:
        nonlocal done
        done += 1
        if r["accepted"]:
            accepted_records_by_recipe[r["recipe"]].append(r["record"])
        elif "error" in r:
            error_counts[r["recipe"]] += 1
        else:
            rejection_counts[r["recipe"]] += 1
        if progress:
            _print_progress(done, len(jobs), r, t_start)

    if workers <= 1:
        # Sequential
        for job in jobs:
            recipe = job["recipe"]
            if len(accepted_records_by_recipe[recipe]) >= target_per_recipe:
                continue  # skip; already met target
            r = _generate_worker(job)
            _accept(r)
            total_accepted = sum(len(v) for v in accepted_records_by_recipe.values())
            if total_accepted >= total_target:
                break
    else:
        # Parallel with imap_unordered. We can't easily skip jobs after submission,
        # so submit all and discard extra acceptances per-recipe in _accept.
        with multiprocessing.Pool(processes=workers) as pool:
            for r in pool.imap_unordered(_generate_worker, jobs):
                _accept(r)
                total_accepted = sum(len(v) for v in accepted_records_by_recipe.values())
                if total_accepted >= total_target * 2:
                    # Over-shot enough; stop
                    pool.terminate()
                    break

    t_total = time.perf_counter() - t_start

    # For each recipe, sort accepted by astar_nodes desc and keep target_per_recipe hardest.
    final_records: list[dict] = []
    for recipe in RECIPES:
        records = sorted(
            accepted_records_by_recipe[recipe.name],
            key=lambda r: r.get("astar_nodes_expanded", 0),
            reverse=True,
        )[:target_per_recipe]
        final_records.extend(records)

    # Re-id sequentially for clean output
    for i, rec in enumerate(final_records):
        recipe = rec["recipe"]
        rec["id"] = f"hard_{recipe}_{i:03d}"

    # Summary stats
    summary = {
        "wall_clock_seconds": round(t_total, 1),
        "total_accepted": len(final_records),
        "target": total_target,
        "per_recipe": {
            r.name: {
                "accepted": len(accepted_records_by_recipe[r.name]),
                "kept": min(len(accepted_records_by_recipe[r.name]), target_per_recipe),
                "rejected": rejection_counts[r.name],
                "errors": error_counts[r.name],
            }
            for r in RECIPES
        },
        "astar_distribution": _astar_distribution(final_records),
    }
    return {"records": final_records, "stats": summary}


def _astar_distribution(records: list[dict]) -> dict:
    nodes = [r["astar_nodes_expanded"] for r in records]
    if not nodes:
        return {"count": 0, "ge_50": 0, "ge_200": 0, "ge_500": 0, "ge_1000": 0}
    return {
        "count": len(nodes),
        "min": min(nodes),
        "max": max(nodes),
        "median": sorted(nodes)[len(nodes) // 2],
        "ge_50": sum(1 for n in nodes if n >= 50),
        "ge_200": sum(1 for n in nodes if n >= 200),
        "ge_500": sum(1 for n in nodes if n >= 500),
        "ge_1000": sum(1 for n in nodes if n >= 1000),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", type=int, default=50,
                   help="Total problems to generate (split equally across 4 recipes)")
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--min-astar-nodes", type=int, default=50)
    p.add_argument("--max-bfs-nodes", type=int, default=50_000)
    p.add_argument("--max-bfs-depth", type=int, default=40)
    p.add_argument("--astar-max-nodes", type=int, default=50_000)
    p.add_argument("--astar-max-depth", type=int, default=40)
    p.add_argument("--max-attempts", type=int, default=3,
                   help="Max generation attempts per seed (each attempt picks a new sub-seed)")
    p.add_argument("--max-seeds-per-recipe", type=int, default=300)
    p.add_argument("--pre-bfs-complexity-max", type=int, default=60,
                   help="Reject generations whose op_count(lhs)+op_count(rhs) exceeds this; avoids "
                        "wasting BFS budget on pathologically-disguised states")
    p.add_argument("--attempt-timeout-s", type=float, default=45.0,
                   help="Wall-clock timeout per single generate_one attempt subprocess")
    p.add_argument("--base-seed-stride", type=int, default=100)
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel processes (default: cpu_count-1; 1 = sequential)")
    p.add_argument("--output", type=str,
                   default="ggmr/problems/hard_evaluation_set.yaml")
    p.add_argument("--report", type=str,
                   default="ggmr/problems/hard_evaluation_set_generation_report.json")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args(argv)

    if args.workers is None:
        args.workers = max(1, (os.cpu_count() or 2) - 1)

    target_per_recipe = max(1, args.target // 4)
    total_target = target_per_recipe * 4

    print(
        f"Generating hard eval set: target={total_target} "
        f"({target_per_recipe} per recipe), depth={args.depth}, "
        f"min_astar_nodes={args.min_astar_nodes}, workers={args.workers}",
        flush=True,
    )

    result = run_generation(
        target_per_recipe=target_per_recipe,
        depth=args.depth,
        min_astar_nodes=args.min_astar_nodes,
        max_bfs_nodes=args.max_bfs_nodes,
        max_bfs_depth=args.max_bfs_depth,
        astar_max_nodes=args.astar_max_nodes,
        astar_max_depth=args.astar_max_depth,
        max_attempts=args.max_attempts,
        max_seeds_per_recipe=args.max_seeds_per_recipe,
        workers=args.workers,
        base_seed_stride=args.base_seed_stride,
        progress=args.progress,
        pre_bfs_complexity_max=args.pre_bfs_complexity_max,
        attempt_timeout_s=args.attempt_timeout_s,
    )

    from ggmr.problems.hard_yaml_emit import emit_hard_problems_yaml

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    emit_hard_problems_yaml(result["records"], str(output_path))
    print(f"Wrote {output_path} ({len(result['records'])} problems)", flush=True)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result["stats"], f, indent=2, default=str)
    print(f"Wrote {report_path}", flush=True)
    print(f"Summary: {json.dumps(result['stats'], indent=2)}", flush=True)

    # Exit nonzero if we didn't hit success criteria
    dist = result["stats"]["astar_distribution"]
    pass_criteria = (
        result["stats"]["total_accepted"] >= 50
        and dist["ge_50"] >= 30
        and dist["ge_200"] >= 10
    )
    return 0 if pass_criteria else 1


if __name__ == "__main__":
    sys.exit(main())
