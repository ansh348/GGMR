"""Generate (state, remaining_steps) training pairs from BFS solution traces.

Every job (easy + hard) shells out to scripts/_generate_one_training.py with a
per-source wall-clock timeout. We learned the hard way that easy jobs can also
hang inside sympy on degenerate states (~30 min on a single ReverseGenerator
attempt), and there's no reliable in-process kill on Windows.

Concurrency uses ThreadPoolExecutor since each "worker" is pure I/O (waiting
on subprocess.run) -- this avoids the 32x python.exe pool-worker overhead
that multiprocessing.Pool would add.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ggmr.training.job_planner import (  # noqa: E402
    plan_easy_jobs,
    plan_hard_jobs,
    plan_trig_jobs,
)

_SUBPROC_SCRIPT = str(Path(__file__).resolve().parent / "_generate_one_training.py")


def _run_subproc(job: dict, timeout_s: float) -> dict:
    """Spawn the subprocess worker with a wall-clock timeout."""
    cmd = [sys.executable, "-u", _SUBPROC_SCRIPT, "--job-json", json.dumps(job)]
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        return {
            "problem_id": job["problem_id"],
            "records": [],
            "skipped": True,
            "reason": f"TIMEOUT({timeout_s:.0f}s)",
        }
    if completed.returncode != 0:
        return {
            "problem_id": job["problem_id"],
            "records": [],
            "skipped": True,
            "reason": f"exit={completed.returncode}: {completed.stderr[:200].strip()}",
        }
    lines = [ln for ln in completed.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return {
            "problem_id": job["problem_id"],
            "records": [],
            "skipped": True,
            "reason": "empty_stdout",
        }
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return {
            "problem_id": job["problem_id"],
            "records": [],
            "skipped": True,
            "reason": f"parse_error: {e}",
        }


def _dispatch(job: dict, easy_timeout_s: float, hard_timeout_s: float) -> dict:
    # For trig jobs, "easy" depths (1-3) use easy_timeout; "hard" depths (4-8)
    # use hard_timeout. Detected via the template prefix set by plan_trig_jobs.
    if job.get("domain") == "trig":
        tpl = job.get("template", "")
        timeout_s = easy_timeout_s if "easy" in tpl else hard_timeout_s
    else:
        timeout_s = easy_timeout_s if job["source"] == "easy" else hard_timeout_s
    return _run_subproc(job, timeout_s)


def _load_resume_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("problem_id")
            if pid:
                ids.add(pid)
    return ids


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--domain", choices=["algebra", "trig"], default="algebra",
                        help="domain selector. algebra (default) plans easy+hard; "
                             "trig plans verify_identity jobs via TrigReverseGenerator "
                             "with training_only=True (Marcus Constraint 1).")
    parser.add_argument("--num-problems", type=int, default=None,
                        help="trig domain only: total problems (70%% easy, 30%% hard). "
                             "Algebra uses --easy-count and --hard-count instead.")
    parser.add_argument("--easy-count", type=int, default=7000)
    parser.add_argument("--hard-count", type=int, default=3000)
    parser.add_argument("--easy-max-depth", type=int, default=10)
    parser.add_argument("--easy-bfs-budget", type=int, default=5000)
    parser.add_argument("--hard-bfs-budget", type=int, default=5000)
    parser.add_argument("--easy-timeout-s", type=float, default=60.0)
    parser.add_argument("--hard-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--run-id", type=str, default="",
                        help="stamped on every record under run_id; useful for "
                             "downstream filtering of training data by run.")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.domain == "trig":
        num_problems = args.num_problems if args.num_problems is not None else 1000
        trig_jobs = plan_trig_jobs(num_problems, seed=args.seed, run_id=args.run_id)
        for j in trig_jobs:
            tpl = j.get("template", "")
            j["bfs_budget"] = (
                args.hard_bfs_budget if "hard" in tpl else args.easy_bfs_budget
            )
        jobs = trig_jobs
        easy_jobs = []
        hard_jobs = []
    else:
        easy_jobs = plan_easy_jobs(args.easy_count, args.easy_max_depth, args.seed)
        for j in easy_jobs:
            j["bfs_budget"] = args.easy_bfs_budget
        hard_jobs = plan_hard_jobs(args.hard_count, args.seed)
        for j in hard_jobs:
            j["bfs_budget"] = args.hard_bfs_budget

        # Interleave easy and hard so families/templates aren't starved when workers
        # finish a chunk.
        jobs: list[dict] = []
        i_e = i_h = 0
        while i_e < len(easy_jobs) or i_h < len(hard_jobs):
            if i_e < len(easy_jobs):
                jobs.append(easy_jobs[i_e]); i_e += 1
            if i_h < len(hard_jobs):
                jobs.append(hard_jobs[i_h]); i_h += 1

    resume_ids: set[str] = set()
    if args.resume:
        resume_ids = _load_resume_ids(output_path)
        jobs = [j for j in jobs if j["problem_id"] not in resume_ids]
        print(f"Resume: {len(resume_ids)} problem_ids already done, {len(jobs)} remaining", flush=True)

    total = len(jobs)
    if total == 0:
        print("No jobs to run.", flush=True)
        return 0

    if args.domain == "trig":
        n_easy_in_plan = sum(1 for j in jobs if "easy" in j.get("template", ""))
        n_hard_in_plan = len(jobs) - n_easy_in_plan
        print(
            f"Plan: domain=trig total={len(jobs)} "
            f"({n_easy_in_plan} easy + {n_hard_in_plan} hard) "
            f"workers={args.workers} easy_timeout={args.easy_timeout_s}s "
            f"hard_timeout={args.hard_timeout_s}s output={output_path}",
            flush=True,
        )
    else:
        print(
            f"Plan: easy={args.easy_count} hard={args.hard_count} "
            f"(after resume: {sum(1 for j in jobs if j['source'] == 'easy')} easy + "
            f"{sum(1 for j in jobs if j['source'] == 'hard')} hard) "
            f"workers={args.workers} easy_timeout={args.easy_timeout_s}s "
            f"hard_timeout={args.hard_timeout_s}s output={output_path}",
            flush=True,
        )

    counts = {"easy_done": 0, "hard_done": 0, "easy_skipped": 0, "hard_skipped": 0,
              "pairs": 0}
    family_counts: dict[str, int] = {}
    template_counts: dict[str, int] = {}
    skip_reasons: dict[str, int] = {}

    t0 = time.perf_counter()
    last_print = t0
    done_idx = 0

    with output_path.open("a", encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(_dispatch, j, args.easy_timeout_s, args.hard_timeout_s)
                for j in jobs
            ]
            for fut in as_completed(futures):
                done_idx += 1
                result = fut.result()
                pid = result["problem_id"]
                if pid.startswith("hard_") or pid.startswith("trig_hard_"):
                    source_key = "hard"
                else:
                    source_key = "easy"
                if result.get("skipped"):
                    counts[f"{source_key}_skipped"] += 1
                    reason = result.get("reason", "unknown")
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                else:
                    counts[f"{source_key}_done"] += 1
                    for rec in result["records"]:
                        fout.write(json.dumps(rec) + "\n")
                        counts["pairs"] += 1
                        if source_key == "hard":
                            fam = rec.get("family", "")
                            if fam:
                                family_counts[fam] = family_counts.get(fam, 0) + 1
                        else:
                            tpl = rec.get("template", "")
                            if tpl:
                                template_counts[tpl] = template_counts.get(tpl, 0) + 1

                now = time.perf_counter()
                if args.progress and (done_idx % 50 == 0 or now - last_print > 10
                                      or done_idx == total):
                    elapsed = now - t0
                    rate = done_idx / elapsed if elapsed > 0 else 0
                    eta = (total - done_idx) / rate if rate > 0 else 0
                    print(
                        f"  [{done_idx:5d}/{total}] easy={counts['easy_done']} "
                        f"hard={counts['hard_done']} pairs={counts['pairs']} "
                        f"skipped={counts['easy_skipped'] + counts['hard_skipped']} "
                        f"elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}",
                        flush=True,
                    )
                    last_print = now

    elapsed = time.perf_counter() - t0
    print("", flush=True)
    print(f"Done in {_format_eta(elapsed)}.", flush=True)
    print(
        f"  Easy: done={counts['easy_done']} skipped={counts['easy_skipped']}",
        flush=True,
    )
    print(
        f"  Hard: done={counts['hard_done']} skipped={counts['hard_skipped']}",
        flush=True,
    )
    print(f"  Total pairs written: {counts['pairs']}", flush=True)
    if template_counts:
        print(f"  Easy templates: {template_counts}", flush=True)
    if family_counts:
        print(f"  Hard families:  {family_counts}", flush=True)
    if skip_reasons:
        print(f"  Skip reasons:   {skip_reasons}", flush=True)
    print(f"  Output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
