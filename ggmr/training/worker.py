"""Per-job worker: takes a job dict, returns records dict.

Used in-process by the orchestrator for easy jobs and as a subprocess
entry point (via scripts/_generate_one_training.py) for hard jobs.
"""
from __future__ import annotations

import random
from typing import Any

from ggmr.problems.generator import ReverseGenerator

from .extract_pairs import extract_training_pairs, pairs_from_trace
from .parameter_sampling import sample_motif_instance


def _annotate(records: list[dict], *, source: str, template: str, family: str,
              depth: int, problem_id: str) -> list[dict]:
    for r in records:
        r["source"] = source
        r["template"] = template
        r["family"] = family
        r["depth"] = depth
        r["problem_id"] = problem_id
    return records


def _generate_easy(job: dict) -> dict[str, Any]:
    template = job["template"]
    depth = job["depth"]
    seed = int(job["seed"])
    problem_id = job["problem_id"]
    bfs_budget = int(job.get("bfs_budget", 5000))

    gen = ReverseGenerator(seed=seed, depth=depth, template=template, max_nodes=bfs_budget)
    problem = gen.generate_one()
    if problem is None:
        return {
            "problem_id": problem_id,
            "records": [],
            "skipped": True,
            "reason": "generation_failed",
        }
    records = pairs_from_trace(problem.forward_trace, problem.target)
    _annotate(records, source="easy", template=template, family="",
              depth=depth, problem_id=problem_id)
    return {
        "problem_id": problem_id,
        "records": records,
        "skipped": False,
        "reason": "",
    }


def _generate_hard(job: dict) -> dict[str, Any]:
    family = job["family"]
    seed = int(job["seed"])
    problem_id = job["problem_id"]
    bfs_budget = int(job.get("bfs_budget", 5000))

    rng = random.Random(seed)
    instance = sample_motif_instance(family, rng)
    if instance is None:
        return {
            "problem_id": problem_id,
            "records": [],
            "skipped": True,
            "reason": "sampling_failed",
        }
    records = extract_training_pairs(
        instance.eq_state,
        instance.target_eq_state,
        max_nodes=bfs_budget,
    )
    if records is None:
        return {
            "problem_id": problem_id,
            "records": [],
            "skipped": True,
            "reason": "bfs_budget_exhausted",
        }
    _annotate(records, source="hard", template=instance.category, family=family,
              depth=0, problem_id=problem_id)
    return {
        "problem_id": problem_id,
        "records": records,
        "skipped": False,
        "reason": "",
    }


def _generate_round2(job: dict) -> dict[str, Any]:
    """Round 2: dispatch by `category` to the round2_categories registry."""
    from ggmr.problems.round2_categories import CATEGORIES

    category = job["category"]
    seed = int(job["seed"])
    problem_id = job["problem_id"]
    bfs_budget = int(job.get("bfs_budget", 5_000))
    depth = int(job.get("depth", 0))

    if category not in CATEGORIES:
        return {
            "problem_id": problem_id,
            "records": [],
            "skipped": True,
            "reason": f"unknown_category: {category}",
        }

    rng = random.Random(seed)
    try:
        instance = CATEGORIES[category](rng, depth)
    except Exception as e:
        return {
            "problem_id": problem_id,
            "records": [],
            "skipped": True,
            "reason": f"gen_fail: {type(e).__name__}: {e}",
        }

    records = extract_training_pairs(
        instance.eq_state,
        instance.target_eq_state,
        max_nodes=bfs_budget,
        max_depth=30,
    )
    if records is None:
        return {
            "problem_id": problem_id,
            "records": [],
            "skipped": True,
            "reason": "bfs_budget_exhausted",
        }
    _annotate(
        records,
        source="round2",
        template=instance.category,
        family=instance.motif_family,
        depth=depth,
        problem_id=problem_id,
    )
    for r in records:
        r["category"] = category
    return {
        "problem_id": problem_id,
        "records": records,
        "skipped": False,
        "reason": "",
    }


def generate_one(job: dict) -> dict[str, Any]:
    if job["source"] == "easy":
        return _generate_easy(job)
    if job["source"] == "hard":
        return _generate_hard(job)
    if job["source"] == "round2":
        return _generate_round2(job)
    return {
        "problem_id": job.get("problem_id", "<unknown>"),
        "records": [],
        "skipped": True,
        "reason": f"unknown_source: {job.get('source')}",
    }
