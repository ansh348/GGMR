"""Tests for ReplayBuffer JSONL save/load + rule_set_hash (Phase 0.4).

Marcus Constraint 2: replay buffers must persist with metadata sufficient
to detect cross-domain or cross-rule-set contamination on resume.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import torch
from torch_geometric.data import Data

import ggmr.rules.core  # noqa: F401  (register algebra rules)
from ggmr.rules.hash import rule_set_hash
from ggmr.rules.registry import Registry
from ggmr.state import EqState
from ggmr.training.exit_loop import PolicyTuple, ReplayBuffer, ValueTuple
from ggmr.training.graph import sympy_to_pyg


def _make_pair(pid: str, *, run_id: str, rule_hash: str) -> tuple[ValueTuple, PolicyTuple]:
    state = EqState.from_strings("2*x + 3", "7")
    graph = sympy_to_pyg(state.lhs, state.rhs, state.var)
    n_rules = 49
    target = np.zeros(n_rules, dtype=np.float32)
    target[0] = 1.0
    mask = np.zeros(n_rules, dtype=np.float32)
    mask[0] = 1.0
    mask[3] = 1.0
    v = ValueTuple(
        graph=Data(x=graph.x, edge_index=graph.edge_index),
        log1p_steps=1.0986,
        problem_id=pid,
        domain="algebra",
        mode="training",
        run_id=run_id,
        iteration=2,
        rule_set_hash=rule_hash,
        model_checkpoint="checkpoints/test/iter_02.pt",
        state_lhs_srepr=sp.srepr(state.lhs),
        state_rhs_srepr=sp.srepr(state.rhs),
        var_name=state.var.name,
    )
    p = PolicyTuple(
        graph=Data(x=graph.x, edge_index=graph.edge_index),
        target_distribution=target,
        legal_mask=mask,
        domain="algebra",
        mode="training",
        run_id=run_id,
        iteration=2,
        rule_set_hash=rule_hash,
        model_checkpoint="checkpoints/test/iter_02.pt",
        state_lhs_srepr=sp.srepr(state.lhs),
        state_rhs_srepr=sp.srepr(state.rhs),
        var_name=state.var.name,
    )
    return v, p


def test_jsonl_save_load_preserves_metadata(tmp_path: Path):
    h = rule_set_hash()
    buf = ReplayBuffer(max_size=100)
    for i in range(5):
        v, p = _make_pair(f"pid_{i}", run_id="testrun", rule_hash=h)
        buf.add(v, p)

    path = tmp_path / "buf.jsonl"
    buf.save(path)
    assert path.exists()

    # Header + 5 entries
    lines = path.read_text().splitlines()
    assert len(lines) == 6
    header = json.loads(lines[0])
    assert header["kind"] == "buffer_header"
    assert header["n_entries"] == 5

    reloaded = ReplayBuffer.load(path)
    assert len(reloaded) == 5
    v0 = reloaded.value_tuples()[0]
    assert v0.run_id == "testrun"
    assert v0.iteration == 2
    assert v0.rule_set_hash == h
    assert v0.domain == "algebra"
    # Graph rebuilt from srepr
    assert v0.graph.x.shape[1] == 30  # FEATURE_DIM


def test_jsonl_rule_set_hash_mismatch_raises(tmp_path: Path):
    h_current = rule_set_hash()
    # Save with a fabricated different hash
    buf = ReplayBuffer(max_size=10)
    v, p = _make_pair("pid_0", run_id="bad", rule_hash="deadbeef" * 8)
    buf.add(v, p)
    path = tmp_path / "buf_bad.jsonl"
    buf.save(path)

    with pytest.raises(ValueError, match="rule_set_hash mismatch"):
        ReplayBuffer.load(path, expected_rule_set_hash=h_current)


def test_jsonl_rule_set_hash_match_loads(tmp_path: Path):
    h = rule_set_hash()
    buf = ReplayBuffer(max_size=10)
    v, p = _make_pair("pid_match", run_id="good", rule_hash=h)
    buf.add(v, p)
    path = tmp_path / "buf_good.jsonl"
    buf.save(path)

    reloaded = ReplayBuffer.load(path, expected_rule_set_hash=h)
    assert len(reloaded) == 1


def test_legacy_pt_buffer_still_loads(tmp_path: Path):
    """Phase 3 iter 0 + 1 + 2 buffers saved as .pt must remain loadable."""
    buf = ReplayBuffer(max_size=10)
    v, p = _make_pair("pid_legacy", run_id="", rule_hash="")
    buf.add(v, p)
    path = tmp_path / "buf_legacy.pt"
    buf.save(path)
    assert path.exists()

    reloaded = ReplayBuffer.load(path)
    assert len(reloaded) == 1
    # Metadata fields filled with defaults from dataclass
    v0 = reloaded.value_tuples()[0]
    assert v0.problem_id == "pid_legacy"


def test_rule_set_hash_stable_and_distinct():
    """Same rule set: same hash. Removed rule: different hash."""
    r1 = Registry()
    r2 = Registry()
    # Empty registries: same hash
    assert rule_set_hash(r1) == rule_set_hash(r2)
    # Register a fake rule in r1 only
    from ggmr.rules.base import Action, GuardResult

    class _Fake:
        name = "FAKE"
        arity = 0
        training_safe = True

        def enumerate(self, state):
            return iter([])

        def guard(self, state, action):
            return GuardResult.passing()

        def apply(self, state, action):
            return state

    r1.register(_Fake())
    assert rule_set_hash(r1) != rule_set_hash(r2)
