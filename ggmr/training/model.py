"""GIN value network: predicts log1p(remaining_steps) from (lhs, rhs) graphs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

from .graph import FEATURE_DIM


class GINValueNet(nn.Module):
    """5-layer GIN with mean+max readout. Output is non-negative (log-space prediction).

    The caller (LearnedHeuristic) applies expm1 + clip to recover step-space.
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Batch) -> Tensor:
        """Returns log-space prediction. May be negative during training;
        the inference wrapper (LearnedHeuristic) clips expm1(pred) >= 0."""
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
        return self.head(pooled).squeeze(-1)


class TreeLSTMValueNet(nn.Module):
    """Stub. Fall-back if GIN underperforms; implement on demand."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("TreeLSTMValueNet is a planned fallback; implement if GIN underperforms.")
