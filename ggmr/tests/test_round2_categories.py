"""Smoke tests for Round 2 category generators.

For each category, generate 2 problems and confirm BFS solves them within the
per-tier budget. Catches degenerate generator output and missing rule coverage.
"""
from __future__ import annotations

import random
from typing import Optional

import pytest

from ggmr.problems.round2_categories import (
    CATEGORIES,
    bfs_budget_for,
)
from ggmr.training.extract_pairs import extract_training_pairs


@pytest.mark.parametrize("cat_name", list(CATEGORIES.keys()))
def test_category_generates_and_solves(cat_name: str) -> None:
    """Each category generates 2 problems; BFS solves both within tier budget."""
    rng = random.Random(0)
    budget = bfs_budget_for(cat_name)
    for i in range(2):
        inst = CATEGORIES[cat_name](rng, depth=5)
        recs = extract_training_pairs(
            inst.eq_state,
            inst.target_eq_state,
            max_nodes=budget,
            max_depth=30,
        )
        assert recs is not None, (
            f"{cat_name} sample {i}: BFS failed within {budget} nodes\n"
            f"  initial.lhs = {inst.eq_state.lhs}\n"
            f"  initial.rhs = {inst.eq_state.rhs}\n"
            f"  target.lhs  = {inst.target_eq_state.lhs}\n"
            f"  target.rhs  = {inst.target_eq_state.rhs}\n"
            f"  params      = {inst.params}"
        )
        assert all(r["remaining_steps"] >= 0 for r in recs)
        # Final record should have remaining_steps == 0
        assert recs[-1]["remaining_steps"] == 0


def test_registry_has_35_categories() -> None:
    assert len(CATEGORIES) == 35, f"Expected 35 categories, got {len(CATEGORIES)}"


def test_all_categories_callable() -> None:
    """Every entry in CATEGORIES is callable and produces a MotifInstance."""
    from ggmr.problems.motif_templates import MotifInstance
    rng = random.Random(123)
    for cat_name, gen in CATEGORIES.items():
        inst = gen(rng, depth=5)
        assert isinstance(inst, MotifInstance), f"{cat_name} did not return MotifInstance"
        assert inst.eq_state.lhs is not None
        assert inst.target_eq_state.lhs is not None
