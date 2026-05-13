"""Round 2 training data generator: 35 categories → ~150K-300K pairs.

Mirrors scripts/generate_training_data.py (ThreadPoolExecutor + per-job subprocess
with wall-clock timeout). Differences:
  * Plans jobs via job_planner.plan_round2_jobs (one per (category, idx))
  * Per-tier BFS budget + timeout pulled from round2_categories
  * Emits distribution report (remaining_steps histogram + category counts)
  * Default output is /data/ (Railway persistent volume)

Why subprocess-per-job: sympy can hang indefinitely on degenerate states.
The ThreadPoolExecutor wraps subprocess.run with a per-job timeout that's
the only reliable kill mechanism on Windows.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ggmr.training.job_planner import plan_round2_jobs  # noqa: E402

_SUBPROC_SCRIPT = str(Path(__file__).resolve().parent / "_generate_one_round2.py")


def _run_subproc(job: dict) -> dict:
    """Spawn the subprocess worker with the per-tier wall-clock timeout."""
    timeout_s = float(job.get("timeout_s", 120.0))
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


def _rs_bucket(rs: int) -> str:
    if rs <= 0:
        return "0"
    if rs == 1:
        return "1"
    if rs == 2:
        return "2"
    if rs <= 4:
        return "3-4"
    if rs <= 8:
        return "5-8"
    return "9-15"


def _emit_distribution_report(jsonl_path: Path, report_path: Path) -> None:
    """Histogram remaining_steps + per-category counts. Flag buckets < 5%."""
    rs_hist: Counter = Counter()
    cat_hist: Counter = Counter()
    fam_hist: Counter = Counter()
    total = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rs_hist[_rs_bucket(int(r.get("remaining_steps", 0)))] += 1
            cat_hist[r.get("category", "(none)")] += 1
            fam_hist[r.get("family", "(none)")] += 1
            total += 1
    if total == 0:
        report_path.write_text(json.dumps({"total_pairs": 0}, indent=2))
        print("  Distribution report: no rows.", flush=True)
        return
    pct = {k: round(v / total * 100, 2) for k, v in rs_hist.items()}
    flagged = [k for k in ("0", "1", "2", "3-4", "5-8", "9-15") if pct.get(k, 0) < 5.0]
    report = {
        "total_pairs": total,
        "remaining_steps_buckets": dict(rs_hist),
        "remaining_steps_pct": pct,
        "underrepresented_buckets": flagged,
        "category_counts": dict(cat_hist),
        "family_counts": dict(fam_hist),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Distribution report -> {report_path}", flush=True)
    print(f"  remaining_steps pct: {pct}", flush=True)
    if flagged:
        print(f"  WARNING: underrepresented buckets: {flagged}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="/data/training_data_round2.jsonl")
    parser.add_argument("--report", type=str, default="/data/round2_distribution_report.json")
    parser.add_argument("--jobs-per-category", type=int, default=1000)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = plan_round2_jobs(jobs_per_category=args.jobs_per_category, seed=args.seed)

    resume_ids: set[str] = set()
    if args.resume:
        resume_ids = _load_resume_ids(output_path)
        jobs = [j for j in jobs if j["problem_id"] not in resume_ids]
        print(f"Resume: {len(resume_ids)} done, {len(jobs)} remaining", flush=True)

    total = len(jobs)
    if total == 0:
        print("No jobs to run.", flush=True)
        # Still write a (possibly empty) report
        if output_path.exists():
            _emit_distribution_report(output_path, report_path)
        return 0

    print(
        f"Plan: {total} jobs across {len({j['category'] for j in jobs})} categories "
        f"(jobs_per_category={args.jobs_per_category}) workers={args.workers} "
        f"output={output_path}",
        flush=True,
    )

    counts = {"done": 0, "skipped": 0, "pairs": 0}
    cat_done: dict[str, int] = {}
    cat_skipped: dict[str, int] = {}
    skip_reasons: dict[str, int] = {}

    t0 = time.perf_counter()
    last_print = t0
    done_idx = 0

    # We need to know each job's category for accounting; build a lookup.
    job_lookup = {j["problem_id"]: j for j in jobs}

    with output_path.open("a", encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_subproc, j) for j in jobs]
            for fut in as_completed(futures):
                done_idx += 1
                result = fut.result()
                pid = result["problem_id"]
                src_job = job_lookup.get(pid, {})
                cat = src_job.get("category", "(unknown)")
                if result.get("skipped"):
                    counts["skipped"] += 1
                    cat_skipped[cat] = cat_skipped.get(cat, 0) + 1
                    reason = result.get("reason", "unknown")
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                else:
                    counts["done"] += 1
                    cat_done[cat] = cat_done.get(cat, 0) + 1
                    for rec in result["records"]:
                        fout.write(json.dumps(rec) + "\n")
                        counts["pairs"] += 1

                now = time.perf_counter()
                if args.progress and (done_idx % 100 == 0 or now - last_print > 15
                                      or done_idx == total):
                    elapsed = now - t0
                    rate = done_idx / elapsed if elapsed > 0 else 0
                    eta = (total - done_idx) / rate if rate > 0 else 0
                    print(
                        f"  [{done_idx:6d}/{total}] done={counts['done']} "
                        f"skipped={counts['skipped']} pairs={counts['pairs']} "
                        f"elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}",
                        flush=True,
                    )
                    last_print = now

    elapsed = time.perf_counter() - t0
    print("", flush=True)
    print(f"Generation finished in {_format_eta(elapsed)}.", flush=True)
    print(f"  Done: {counts['done']}  Skipped: {counts['skipped']}  Pairs: {counts['pairs']}",
          flush=True)
    if skip_reasons:
        # Print top 8 reasons by frequency
        top = sorted(skip_reasons.items(), key=lambda kv: -kv[1])[:8]
        print(f"  Top skip reasons: {dict(top)}", flush=True)
    print(f"  Per-category (done): {dict(sorted(cat_done.items()))}", flush=True)
    print(f"  Per-category (skipped): {dict(sorted(cat_skipped.items()))}", flush=True)

    # Distribution report
    _emit_distribution_report(output_path, report_path)

    # Single-line DONE marker for Railway log scanning
    print(f"DONE: {counts['pairs']} pairs in {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
