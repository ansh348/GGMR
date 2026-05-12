r"""Hard evaluation set validator.

Loads `hard_evaluation_set.yaml`, runs BFS + A* on each problem with
`check_soundness=True`, and confirms the 4 success criteria from the plan:

  1. ≥ 30 of 50 problems have A* nodes ≥ 50
  2. ≥ 10 of 50 problems have A* nodes ≥ 200
  3. All 50 problems are BFS-solvable within `--max-bfs-nodes`
  4. 0 unsound transitions across all BFS + A* expansions

Usage (PowerShell):
    & .\.venv\Scripts\python.exe scripts\validate_hard_set.py `
        --input ggmr\problems\hard_evaluation_set.yaml `
        --output ggmr\problems\hard_evaluation_set_validation_report.json `
        --workers 8
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


def _validate_one(args: dict) -> dict:
    """Worker: validate one problem record. Top-level for multiprocessing pickling."""
    import ggmr.rules.core  # noqa: F401  (registers forward rules)
    from ggmr.expr.tree import canonical_repr
    from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
    from ggmr.problems.hard_yaml_emit import load_hard_problems_yaml
    from ggmr.search.astar import astar
    from ggmr.search.bfs import bfs
    from ggmr.soundness import IllegalStepError

    input_path = args["input_path"]
    record_idx = args["record_idx"]
    max_bfs_nodes = args["max_bfs_nodes"]
    max_bfs_depth = args["max_bfs_depth"]
    astar_max_nodes = args["astar_max_nodes"]
    astar_max_depth = args["astar_max_depth"]

    records = load_hard_problems_yaml(input_path)
    record = records[record_idx]

    target_l = canonical_repr(record.canonical_target.lhs)
    target_r = canonical_repr(record.canonical_target.rhs)

    def is_target(s) -> bool:
        return (
            canonical_repr(s.lhs) == target_l
            and canonical_repr(s.rhs) == target_r
        ) or s.is_canonical_target()

    out = {
        "id": record.id,
        "recipe": record.recipe,
        "category": record.category,
        "claimed_astar_nodes": record.astar_nodes_expanded,
        "claimed_bfs_nodes": record.bfs_nodes_expanded,
    }

    # BFS with soundness check
    bfs_unsound = False
    bfs_unsound_reason = ""
    try:
        bfs_result = bfs(
            record.initial,
            is_target,
            max_nodes=max_bfs_nodes,
            max_depth=max_bfs_depth,
            check_soundness=True,
            problem_id=record.id,
        )
        out["bfs_found"] = bool(bfs_result.found)
        out["bfs_nodes_expanded"] = int(bfs_result.stats.nodes_expanded)
        out["bfs_path_length"] = int(bfs_result.num_steps)
    except IllegalStepError as e:
        bfs_unsound = True
        bfs_unsound_reason = f"BFS unsound: {e.reason}"
        out["bfs_found"] = False
        out["bfs_nodes_expanded"] = -1
        out["bfs_unsound"] = True
    except Exception as e:
        out["bfs_found"] = False
        out["bfs_error"] = f"{type(e).__name__}: {e}"
        out["bfs_nodes_expanded"] = -1

    # A* with soundness check
    heur = WeightedSumCompositeHeuristic()
    astar_unsound = False
    astar_unsound_reason = ""
    try:
        astar_result = astar(
            record.initial,
            is_target,
            heuristic=heur,
            max_nodes=astar_max_nodes,
            max_depth=astar_max_depth,
            check_soundness=True,
            problem_id=record.id,
        )
        out["astar_found"] = bool(astar_result.found)
        out["astar_nodes_expanded"] = int(astar_result.stats.nodes_expanded)
        out["astar_path_length"] = int(astar_result.num_steps)
    except IllegalStepError as e:
        astar_unsound = True
        astar_unsound_reason = f"A* unsound: {e.reason}"
        out["astar_found"] = False
        out["astar_nodes_expanded"] = -1
        out["astar_unsound"] = True
    except Exception as e:
        out["astar_found"] = False
        out["astar_error"] = f"{type(e).__name__}: {e}"
        out["astar_nodes_expanded"] = -1

    out["unsound"] = bfs_unsound or astar_unsound
    if bfs_unsound:
        out["unsound_reason"] = bfs_unsound_reason
    elif astar_unsound:
        out["unsound_reason"] = astar_unsound_reason

    return out


def _validate_one_timed(job: dict) -> dict:
    """Wrap _validate_one in a subprocess with a hard wall-clock timeout.

    Mirrors scripts/_generate_one_hard.py's pattern. Per-problem timeout
    comes from job["per_problem_timeout_s"]. On TimeoutExpired the
    subprocess is killed and a TIME_BUDGET_EXCEEDED record is returned.
    """
    import subprocess
    import sys as _sys

    timeout_s = float(job.get("per_problem_timeout_s", 120.0))
    python_exe = job.get("python_exe", _sys.executable)
    job_id = job.get("id", "unknown")

    sub_job = {
        k: v for k, v in job.items()
        if k not in {"per_problem_timeout_s", "python_exe", "id"}
    }
    script_path = str(Path(__file__).resolve().parent / "_validate_one_subproc.py")
    cmd = [python_exe, "-u", script_path, "--job-json", json.dumps(sub_job)]

    def _err_skeleton(error_msg: str) -> dict:
        return {
            "id": job_id,
            "recipe": "unknown",
            "category": "unknown",
            "claimed_astar_nodes": 0,
            "claimed_bfs_nodes": 0,
            "bfs_found": False,
            "bfs_nodes_expanded": -1,
            "bfs_path_length": -1,
            "astar_found": False,
            "astar_nodes_expanded": -1,
            "astar_path_length": -1,
            "unsound": False,
            "error": error_msg,
        }

    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        r = _err_skeleton(f"TIME_BUDGET_EXCEEDED({timeout_s:.0f}s)")
        r["timeout"] = True
        r["timeout_s"] = timeout_s
        return r

    if completed.returncode != 0:
        return _err_skeleton(
            f"exit_code={completed.returncode}: {completed.stderr[:300]}"
        )

    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as e:
        return _err_skeleton(
            f"parse_error: {type(e).__name__}: {e}; stdout={completed.stdout[:200]}"
        )


def _print_progress(done: int, total: int, r: dict, t_start: float) -> None:
    elapsed = time.perf_counter() - t_start
    eta = (elapsed / done) * (total - done) if done > 0 else 0
    if r.get("timeout"):
        bfs_ok, astar_status = "BFS?", "TIMEOUT"
    else:
        bfs_ok = "BFS+" if r.get("bfs_found") else "BFS-"
        astar_n = r.get("astar_nodes_expanded", -1)
        astar_status = f"A*={astar_n:>5d}" if r.get("astar_found") else "A*-"
    sound = "" if not r.get("unsound") else " UNSOUND"
    print(
        f"  [{done:3d}/{total}] {r['id']:32s} {bfs_ok:5s} {astar_status:10s}{sound:9s} "
        f"elapsed={elapsed:6.1f}s eta={eta:6.0f}s",
        flush=True,
    )


def run_validation(
    input_path: str,
    max_bfs_nodes: int,
    max_bfs_depth: int,
    astar_max_nodes: int,
    astar_max_depth: int,
    workers: int,
    progress: bool,
    per_problem_timeout_s: float = 0.0,
) -> dict:
    # Just need record count; full load happens in workers
    from ggmr.problems.hard_yaml_emit import load_hard_problems_yaml

    records = load_hard_problems_yaml(input_path)
    n = len(records)

    jobs: list[dict] = [
        {
            "input_path": input_path,
            "record_idx": i,
            "id": records[i].id,
            "max_bfs_nodes": max_bfs_nodes,
            "max_bfs_depth": max_bfs_depth,
            "astar_max_nodes": astar_max_nodes,
            "astar_max_depth": astar_max_depth,
            "per_problem_timeout_s": per_problem_timeout_s,
            "python_exe": sys.executable,
        }
        for i in range(n)
    ]

    worker_fn = _validate_one_timed if per_problem_timeout_s > 0 else _validate_one

    t_start = time.perf_counter()
    results: list[dict] = []

    if workers <= 1:
        for i, job in enumerate(jobs):
            r = worker_fn(job)
            results.append(r)
            if progress:
                _print_progress(i + 1, n, r, t_start)
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            for i, r in enumerate(pool.imap_unordered(worker_fn, jobs)):
                results.append(r)
                if progress:
                    _print_progress(i + 1, n, r, t_start)

    t_total = time.perf_counter() - t_start

    # Aggregate
    bfs_solved = sum(1 for r in results if r.get("bfs_found"))
    astar_solved = sum(1 for r in results if r.get("astar_found"))
    unsound = [r for r in results if r.get("unsound")]
    timeouts = [r for r in results if r.get("timeout") or r.get("error", "").startswith("TIME_BUDGET")]
    astar_nodes = [r["astar_nodes_expanded"] for r in results if r.get("astar_found")]
    ge_50 = sum(1 for n in astar_nodes if n >= 50)
    ge_200 = sum(1 for n in astar_nodes if n >= 200)
    ge_500 = sum(1 for n in astar_nodes if n >= 500)
    ge_1000 = sum(1 for n in astar_nodes if n >= 1000)

    summary = {
        "wall_clock_seconds": round(t_total, 1),
        "total_problems": n,
        "bfs_solved": bfs_solved,
        "astar_solved": astar_solved,
        "timeout_count": len(timeouts),
        "timeout_ids": [r["id"] for r in timeouts],
        "unsound_count": len(unsound),
        "unsound_ids": [r["id"] for r in unsound],
        "astar_node_distribution": {
            "ge_50": ge_50,
            "ge_200": ge_200,
            "ge_500": ge_500,
            "ge_1000": ge_1000,
            "min": min(astar_nodes) if astar_nodes else None,
            "max": max(astar_nodes) if astar_nodes else None,
            "median": sorted(astar_nodes)[len(astar_nodes) // 2] if astar_nodes else None,
        },
        "criteria": {
            "c1_ge_50_at_least_30": ge_50 >= 30,
            "c2_ge_200_at_least_10": ge_200 >= 10,
            "c3_all_bfs_solved": bfs_solved == n,
            "c4_zero_unsound": len(unsound) == 0,
        },
    }
    summary["all_criteria_pass"] = all(summary["criteria"].values())

    return {"summary": summary, "per_problem": results}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=str,
                   default="ggmr/problems/hard_evaluation_set.yaml")
    p.add_argument("--output", type=str,
                   default="ggmr/problems/hard_evaluation_set_validation_report.json")
    p.add_argument("--max-bfs-nodes", type=int, default=50_000)
    p.add_argument("--max-bfs-depth", type=int, default=50)
    p.add_argument("--astar-max-nodes", type=int, default=50_000)
    p.add_argument("--astar-max-depth", type=int, default=50)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--per-problem-timeout-s", type=float, default=0.0,
                   help="Hard wall-clock timeout per problem in seconds (0 = disabled). "
                        "When >0, each problem runs in a subprocess and is killed on timeout, "
                        "marked TIME_BUDGET_EXCEEDED in results.")
    args = p.parse_args(argv)

    if args.workers is None:
        args.workers = max(1, (os.cpu_count() or 2) - 1)

    timeout_str = f", per-problem-timeout={args.per_problem_timeout_s}s" if args.per_problem_timeout_s > 0 else ""
    print(f"Validating {args.input} (workers={args.workers}{timeout_str})", flush=True)

    result = run_validation(
        input_path=args.input,
        max_bfs_nodes=args.max_bfs_nodes,
        max_bfs_depth=args.max_bfs_depth,
        astar_max_nodes=args.astar_max_nodes,
        astar_max_depth=args.astar_max_depth,
        workers=args.workers,
        progress=args.progress,
        per_problem_timeout_s=args.per_problem_timeout_s,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Wrote {output_path}", flush=True)
    print(f"Summary: {json.dumps(result['summary'], indent=2)}", flush=True)
    return 0 if result["summary"]["all_criteria_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
