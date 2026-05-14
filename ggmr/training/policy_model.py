"""GIN policy network: predicts a probability distribution over the 49 rule
names given an equation graph.

Same GIN backbone shape as `GINValueNet` (5 layers, 128 hidden, mean+max
readout). Separate weights — keeps gradient interference between value and
policy losses at bay during ExIt training. Output is raw logits over
`default_registry.names()` (the canonical 49-rule ordering); callers apply
legality masking + softmax via `PolicyAdvisor` (`ggmr/training/policy_heuristic.py`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

from ggmr.rules.registry import default_registry

from .graph import FEATURE_DIM


def num_rules() -> int:
    """Number of rules in the canonical registry (currently 49 for Phase 1b+)."""
    return len(default_registry.names())


class GINPolicyNet(nn.Module):
    """5-layer GIN with mean+max readout. Output is a [batch, num_rules] tensor
    of raw logits. Masking and softmax happen at inference in PolicyAdvisor.
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 5,
        dropout: float = 0.1,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else num_rules()

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(d_in, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, batch: Batch) -> Tensor:
        x = batch.x
        edge_index = batch.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = torch.cat(
            [global_mean_pool(x, batch.batch), global_max_pool(x, batch.batch)],
            dim=-1,
        )
        return self.head(pooled)


def masked_log_softmax(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Softmax over only the legal entries (mask=1.0), set illegal entries to 0.

    Args:
        logits: [batch, num_rules] raw logits
        legal_mask: [batch, num_rules] in {0.0, 1.0}, 1.0 = legal

    Returns:
        log_probs: [batch, num_rules]. Illegal entries are -inf in log space.
    """
    masked = logits.masked_fill(legal_mask < 0.5, float("-inf"))
    return F.log_softmax(masked, dim=-1)


def masked_softmax(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Softmax over legal entries; illegal entries get probability 0."""
    return masked_log_softmax(logits, legal_mask).exp()
