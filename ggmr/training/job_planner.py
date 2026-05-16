"""Build deterministic job lists for training data generation."""
from __future__ import annotations

from .parameter_sampling import FAMILIES

EASY_TEMPLATES = ("linear", "quadratic", "rational", "polynomial", "mixed")

# Trig (Phase 1.2): single mixed template with depth-banded easy/hard split.
TRIG_EASY_DEPTHS = (1, 2, 3)
TRIG_HARD_DEPTHS = (4, 5, 6, 7, 8)


def plan_easy_jobs(count: int, max_depth: int, seed: int) -> list[dict]:
    """Cycle through (template, depth) pairs, distributing `count` jobs evenly."""
    if count <= 0:
        return []
    depths = list(range(3, max_depth + 1))
    if not depths:
        depths = [max_depth]
    pairs = [(t, d) for t in EASY_TEMPLATES for d in depths]
    jobs: list[dict] = []
    per_pair_counters: dict[tuple[str, int], int] = {p: 0 for p in pairs}
    for i in range(count):
        template, depth = pairs[i % len(pairs)]
        idx = per_pair_counters[(template, depth)]
        per_pair_counters[(template, depth)] += 1
        # Combine the global seed with a per-pair offset so each (template,depth,idx)
        # gets a deterministic, well-separated seed.
        job_seed = seed * 1_000_003 + hash((template, depth, idx)) % 1_000_003
        jobs.append({
            "source": "easy",
            "template": template,
            "depth": depth,
            "seed": job_seed,
            "problem_id": f"easy_{template}_{depth}_{idx:05d}",
        })
    return jobs


def plan_hard_jobs(count: int, seed: int) -> list[dict]:
    """Cycle through 7 motif families, distributing `count` jobs evenly."""
    if count <= 0:
        return []
    jobs: list[dict] = []
    per_family_counters: dict[str, int] = {f: 0 for f in FAMILIES}
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        idx = per_family_counters[family]
        per_family_counters[family] += 1
        job_seed = seed * 1_000_003 + hash(("hard", family, idx)) % 1_000_003
        jobs.append({
            "source": "hard",
            "family": family,
            "seed": job_seed,
            "problem_id": f"hard_{family}_{idx:05d}",
        })
    return jobs


# Depth cycle used by round2 medium/hard categories: maps idx → advisory depth
# value carried in the JSONL `depth` field. Trivial/easy ignore this.
_ROUND2_DEPTH_CYCLE = (3, 5, 8, 10, 12, 15)


def plan_trig_jobs(num_problems: int, *, seed: int = 42, easy_frac: float = 0.7,
                   run_id: str = "") -> list[dict]:
    """Plan `num_problems` trig identity-verification jobs.

    `easy_frac` fraction get depths from TRIG_EASY_DEPTHS; remainder from
    TRIG_HARD_DEPTHS. Each job carries `domain="trig"` so worker.py dispatches
    to TrigReverseGenerator with training_only=True.
    """
    if num_problems <= 0:
        return []
    n_easy = int(num_problems * easy_frac)
    n_hard = num_problems - n_easy
    jobs: list[dict] = []
    for idx in range(n_easy):
        depth = TRIG_EASY_DEPTHS[idx % len(TRIG_EASY_DEPTHS)]
        job_seed = seed * 1_000_003 + hash(("trig_easy", depth, idx)) % 1_000_003
        jobs.append({
            "source": "trig",
            "domain": "trig",
            "template": "trig_easy",
            "depth": depth,
            "seed": job_seed,
            "problem_id": f"trig_easy_{idx:05d}",
            "run_id": run_id,
        })
    for idx in range(n_hard):
        depth = TRIG_HARD_DEPTHS[idx % len(TRIG_HARD_DEPTHS)]
        job_seed = seed * 1_000_003 + hash(("trig_hard", depth, idx)) % 1_000_003
        jobs.append({
            "source": "trig",
            "domain": "trig",
            "template": "trig_hard",
            "depth": depth,
            "seed": job_seed,
            "problem_id": f"trig_hard_{idx:05d}",
            "run_id": run_id,
        })
    return jobs


def plan_round2_jobs(*, jobs_per_category: int = 1000, seed: int = 42) -> list[dict]:
    """Plan ~35 * jobs_per_category jobs, one per (category, idx).

    Deterministic seeding: each (category, idx) pair maps to a stable job_seed.
    Categories that need a `bfs_budget` and per-tier timeout look up
    `bfs_budget_for(category)` / `timeout_for(category)` from round2_categories.
    """
    from ..problems.round2_categories import (
        CATEGORIES,
        bfs_budget_for,
        timeout_for,
    )

    if jobs_per_category <= 0:
        return []
    jobs: list[dict] = []
    for cat_name in CATEGORIES.keys():
        for idx in range(jobs_per_category):
            job_seed = (
                seed * 1_000_003 + hash(("round2", cat_name, idx)) % 1_000_003
            )
            depth = _ROUND2_DEPTH_CYCLE[idx % len(_ROUND2_DEPTH_CYCLE)]
            jobs.append({
                "source": "round2",
                "category": cat_name,
                "depth": depth,
                "seed": job_seed,
                "problem_id": f"round2_{cat_name}_{idx:05d}",
                "bfs_budget": bfs_budget_for(cat_name),
                "timeout_s": timeout_for(cat_name),
            })
    return jobs
