"""Build deterministic job lists for training data generation."""
from __future__ import annotations

from .parameter_sampling import FAMILIES

EASY_TEMPLATES = ("linear", "quadratic", "rational", "polynomial", "mixed")


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
