"""Shared fixtures: Phase 0 problem set, corpus features, common states."""

from __future__ import annotations

from pathlib import Path

import pytest
import sympy as sp
import yaml

from ggmr.state import EqState

PHASE0_PROBLEMS_PATH = (
    Path(__file__).resolve().parents[2] / "phase0" / "problems" / "problems.yaml"
)


@pytest.fixture(scope="session")
def phase0_problems() -> list[dict]:
    with open(PHASE0_PROBLEMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def phase0_states(phase0_problems) -> list[tuple[str, EqState, EqState]]:
    """List of (problem_id, initial_state, target_state)."""
    out = []
    for entry in phase0_problems:
        initial = EqState.from_strings(
            entry["initial"]["lhs"],
            entry["initial"]["rhs"],
            var_name=entry["variable"],
        )
        tgt = EqState.from_strings(
            entry["canonical_target"]["lhs"],
            entry["canonical_target"]["rhs"],
            var_name=entry["variable"],
        )
        out.append((entry["id"], initial, tgt))
    return out


@pytest.fixture(scope="session")
def x() -> sp.Symbol:
    return sp.Symbol("x")
