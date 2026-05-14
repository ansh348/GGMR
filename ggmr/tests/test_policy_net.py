"""Tests for GINPolicyNet: shape, masking, gradient flow."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch

from ggmr.rules.core import *  # noqa: F401,F403  (register rules)
from ggmr.rules.registry import default_registry
from ggmr.state import EqState
from ggmr.training.graph import sympy_to_pyg
from ggmr.training.policy_model import (
    GINPolicyNet,
    masked_log_softmax,
    masked_softmax,
    num_rules,
)


def _build_batch(states: list[EqState]) -> Batch:
    datas = [sympy_to_pyg(s.lhs, s.rhs, s.var) for s in states]
    return Batch.from_data_list(datas)


def test_policy_forward_shape():
    net = GINPolicyNet()
    net.eval()
    states = [
        EqState.from_strings("x + 1", "2"),
        EqState.from_strings("x**2 - 4", "0"),
    ]
    batch = _build_batch(states)
    with torch.no_grad():
        logits = net(batch)
    assert logits.shape == (2, num_rules()), f"got {logits.shape}, expected (2, {num_rules()})"


def test_policy_output_matches_num_rules():
    """The output dim must equal the number of registered rules so we can index by rule name."""
    net = GINPolicyNet()
    assert net.out_dim == len(default_registry.names())


def test_masked_softmax_zeros_illegal():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # only entries 0 and 2 are legal
    probs = masked_softmax(logits, mask)
    assert probs.shape == logits.shape
    assert abs(probs[0, 1].item()) < 1e-9
    assert abs(probs[0, 3].item()) < 1e-9
    assert abs(probs.sum().item() - 1.0) < 1e-6


def test_masked_softmax_uniform_when_logits_equal():
    """Equal logits over legal entries -> uniform distribution over legal entries."""
    logits = torch.zeros(1, 4)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    probs = masked_softmax(logits, mask)
    assert abs(probs[0, 0].item() - 1 / 3) < 1e-6
    assert abs(probs[0, 1].item() - 1 / 3) < 1e-6
    assert abs(probs[0, 2].item() - 1 / 3) < 1e-6
    assert abs(probs[0, 3].item()) < 1e-9


def test_masked_log_softmax_illegal_is_neg_inf():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    lp = masked_log_softmax(logits, mask)
    assert torch.isinf(lp[0, 1]) and lp[0, 1] < 0
    # log_softmax(0, 2 with masked logits 1, 3) = log_softmax([1, 3]) shifted to (-2, 0) ish
    # exp(lp[0, 0]) + exp(lp[0, 2]) should be 1
    assert abs(lp[0, 0].exp().item() + lp[0, 2].exp().item() - 1.0) < 1e-6


def test_policy_gradient_flow():
    """Backprop reaches every parameter in the network."""
    net = GINPolicyNet()
    net.train()
    states = [EqState.from_strings("x + 1", "2")]
    batch = _build_batch(states)
    logits = net(batch)
    loss = logits.sum()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        # Some grads can be 0 if path is dead, but they should be allocated
        assert p.grad.shape == p.shape
