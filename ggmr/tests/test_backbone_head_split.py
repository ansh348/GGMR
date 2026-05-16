"""Tests for the backbone/head split (Phase 0.3).

The split has two correctness claims:

1. `backbone_state_dict()` + `head_state_dict()` together reconstruct the
   full `state_dict()`. Nothing leaks; nothing duplicates.

2. Loading only the backbone into a fresh net leaves head weights at
   random init. Loading backbone-then-head reproduces the original
   forward output exactly (the load order doesn't change the math).

3. `load_backbone()` refuses inputs that contain head keys — protects
   against accidentally calling it with a full state_dict.

The transfer experiment depends on (1) and (2): algebra backbone loads
cleanly into a trig net with the trig head random-init.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

import ggmr.rules.core  # noqa: F401  (register algebra rules)
from ggmr.state import EqState
from ggmr.training.graph import FEATURE_DIM, sympy_to_pyg
from ggmr.training.model import GINValueNet
from ggmr.training.policy_model import GINPolicyNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALUE_CKPT_30 = PROJECT_ROOT / "value_iter_00_30dim.pt"


def _fixed_algebra_batch() -> Batch:
    """Deterministic small algebra batch for forward-output comparison."""
    states = [
        EqState.from_strings("2*x + 3", "7"),
        EqState.from_strings("x**2 - 4", "0"),
        EqState.from_strings("(x - 1)/2", "3"),
    ]
    data = [sympy_to_pyg(s.lhs, s.rhs, s.var) for s in states]
    return Batch.from_data_list(data)


def test_backbone_and_head_partition_state_dict():
    net = GINValueNet(in_dim=FEATURE_DIM)
    full = net.state_dict()
    backbone = net.backbone_state_dict()
    head = net.head_state_dict()

    assert set(backbone.keys()).isdisjoint(set(head.keys())), \
        "backbone and head must not share keys"
    union = set(backbone.keys()) | set(head.keys())
    assert union == set(full.keys()), \
        f"missing keys: {set(full.keys()) - union}; extra: {union - set(full.keys())}"


def test_load_backbone_rejects_head_keys():
    net = GINValueNet(in_dim=FEATURE_DIM)
    # Build a state that contains head keys, simulate a user mistake
    bad_state = {**net.backbone_state_dict(), "head.0.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="head keys"):
        net.load_backbone(bad_state)


def test_load_backbone_leaves_head_unchanged():
    net = GINValueNet(in_dim=FEATURE_DIM)
    head_before = {k: v.clone() for k, v in net.head_state_dict().items()}

    # Save current backbone, perturb the live net, then reload the backbone.
    saved_backbone = {k: v.clone() for k, v in net.backbone_state_dict().items()}
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))

    net.load_backbone(saved_backbone)
    head_after = net.head_state_dict()
    for k in head_before:
        # Head weights should still be the *perturbed* values, not the originals
        assert not torch.allclose(head_before[k], head_after[k]), (
            f"head key {k!r} reverted unexpectedly — load_backbone touched the head"
        )


def test_split_roundtrip_preserves_forward(tmp_path):
    """backbone+head loaded separately into a fresh net == original forward output."""
    if not VALUE_CKPT_30.exists():
        pytest.skip(f"{VALUE_CKPT_30} not present")

    ckpt = torch.load(VALUE_CKPT_30, map_location="cpu", weights_only=False)
    orig = GINValueNet(
        in_dim=ckpt["config"]["in_dim"],
        hidden_dim=ckpt["config"]["hidden_dim"],
        num_layers=ckpt["config"]["num_layers"],
        dropout=0.0,
    )
    orig.load_state_dict(ckpt["model_state"])
    orig.eval()

    backbone_state = orig.backbone_state_dict()
    head_state = orig.head_state_dict()

    # Fresh net, load backbone + head separately
    new_net = GINValueNet(
        in_dim=ckpt["config"]["in_dim"],
        hidden_dim=ckpt["config"]["hidden_dim"],
        num_layers=ckpt["config"]["num_layers"],
        dropout=0.0,
    )
    new_net.load_backbone(backbone_state)
    new_net.load_state_dict(head_state, strict=False)
    new_net.eval()

    batch = _fixed_algebra_batch()
    with torch.no_grad():
        out_orig = orig(batch)
        out_new = new_net(batch)

    assert torch.allclose(out_orig, out_new, atol=1e-6), \
        f"split roundtrip drifted: orig={out_orig.tolist()} new={out_new.tolist()}"


def test_policy_split_partition():
    """Same partition claim for GINPolicyNet."""
    net = GINPolicyNet(in_dim=FEATURE_DIM, out_dim=49)
    full = set(net.state_dict().keys())
    backbone = set(net.backbone_state_dict().keys())
    head = set(net.head_state_dict().keys())
    assert backbone.isdisjoint(head)
    assert backbone | head == full
