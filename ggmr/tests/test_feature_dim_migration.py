"""Sanity test for FEATURE_DIM 24 -> 30 migration (Phase 0.2).

Verifies that a zero-padded 24->30 migrated checkpoint produces algebra
predictions mathematically equivalent to the original 24-dim checkpoint.
This is the load-bearing claim behind the user's "zero-pad rather than
retrain" decision.

The test is skipped if the migrated file `value_iter_00_30dim.pt` doesn't
exist (CI may not have run the migration script).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sympy as sp
import torch
from torch_geometric.data import Batch

import ggmr.rules.core  # noqa: F401  (register algebra rules)
from ggmr.state import EqState
from ggmr.training.graph import FEATURE_DIM, LEGACY_FEATURE_DIM, sympy_to_pyg
from ggmr.training.model import GINValueNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_CKPT = PROJECT_ROOT / "value_iter_00.pt"
NEW_CKPT = PROJECT_ROOT / "value_iter_00_30dim.pt"


@pytest.fixture(scope="module")
def old_model() -> GINValueNet:
    """Load the 24-dim algebra checkpoint into a 24-dim model."""
    if not OLD_CKPT.exists():
        pytest.skip(f"{OLD_CKPT} not present")
    ckpt = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
    assert ckpt["config"]["in_dim"] == LEGACY_FEATURE_DIM
    m = GINValueNet(
        in_dim=LEGACY_FEATURE_DIM,
        hidden_dim=ckpt["config"].get("hidden_dim", 128),
        num_layers=ckpt["config"].get("num_layers", 5),
        dropout=0.0,
    )
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


@pytest.fixture(scope="module")
def new_model() -> GINValueNet:
    """Load the migrated 30-dim algebra checkpoint into a 30-dim model."""
    if not NEW_CKPT.exists():
        pytest.skip(f"{NEW_CKPT} not present (run scripts/migrate_24_to_30.py)")
    ckpt = torch.load(NEW_CKPT, map_location="cpu", weights_only=False)
    assert ckpt["config"]["in_dim"] == FEATURE_DIM == 30
    m = GINValueNet(
        in_dim=FEATURE_DIM,
        hidden_dim=ckpt["config"].get("hidden_dim", 128),
        num_layers=ckpt["config"].get("num_layers", 5),
        dropout=0.0,
    )
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


_ALGEBRA_TEST_STATES = [
    ("2*x + 3", "7"),
    ("x", "5"),
    ("x**2 - 4", "0"),
    ("3*(x + 1)", "12"),
    ("(x + 1)/2", "3"),
]


@pytest.mark.parametrize("lhs, rhs", _ALGEBRA_TEST_STATES)
def test_migrated_matches_original_on_algebra(old_model, new_model, lhs, rhs):
    """Migrated 30-dim model output == 24-dim model output on algebra states.

    Zero-padded weights mean the new 6 columns contribute nothing on algebra
    inputs (whose has_trig/has_exp/... flags are 0). Forward outputs should
    match to within numerical noise.
    """
    state = EqState.from_strings(lhs, rhs)

    # 30-dim graph (current builder)
    data_30 = sympy_to_pyg(state.lhs, state.rhs, state.var)
    # Algebra states have zeros in cols [24:30]
    assert float(data_30.x[:, LEGACY_FEATURE_DIM:].sum()) == 0.0

    # 24-dim graph (slice first 24 cols)
    data_24 = data_30.clone()
    data_24.x = data_30.x[:, :LEGACY_FEATURE_DIM].contiguous()

    with torch.no_grad():
        out_old = float(old_model(Batch.from_data_list([data_24])).item())
        out_new = float(new_model(Batch.from_data_list([data_30])).item())

    assert abs(out_old - out_new) < 1e-5, (
        f"Migration drift on {lhs}={rhs}: old={out_old:.6f} new={out_new:.6f}"
    )
