"""GIN value network: predicts log1p(remaining_steps) from (lhs, rhs) graphs.

The network factors structurally into two pieces:

* **Backbone** — the 5 GINConv layers + mean+max readout. State-dict keys
  are prefixed ``convs.*``. These are the weights that the cross-domain
  transfer experiment targets: structural reasoning learned on algebra
  is expected to bootstrap trig (and later calculus).

* **Head** — the 2-layer MLP that maps the [2*hidden] pooled representation
  to a single log-space step prediction. State-dict keys are prefixed
  ``head.*``. The head is domain-specific (re-initialized when transferring
  the backbone to a new domain).

`backbone_state_dict()` / `head_state_dict()` extract those subsets without
restructuring the module hierarchy — old monolithic checkpoints continue to
load with `load_state_dict()` unchanged. Marcus Constraint 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

from .graph import FEATURE_DIM

# State-dict key prefixes that identify backbone vs head weights.
# Stays in sync with the attribute names below; if either is renamed,
# `backbone_state_dict` / `head_state_dict` MUST be updated.
_BACKBONE_PREFIXES: tuple[str, ...] = ("convs.",)
_VALUE_HEAD_PREFIXES: tuple[str, ...] = ("head.",)


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

    # --- transfer helpers ----------------------------------------------------

    def backbone_state_dict(self) -> dict[str, Tensor]:
        """Subset of state_dict containing only the GIN backbone weights
        (transferable across domains). Used by `save_checkpoints` to emit
        a `backbone_iter_NN.pt` file alongside the full checkpoint, and by
        the transfer experiment to load algebra weights into a fresh trig net.
        """
        full = self.state_dict()
        return {k: v for k, v in full.items()
                if any(k.startswith(p) for p in _BACKBONE_PREFIXES)}

    def head_state_dict(self) -> dict[str, Tensor]:
        """Subset of state_dict containing only the value-head weights
        (domain-specific). For symmetry with `backbone_state_dict`."""
        full = self.state_dict()
        return {k: v for k, v in full.items()
                if any(k.startswith(p) for p in _VALUE_HEAD_PREFIXES)}

    def load_backbone(self, state: dict[str, Tensor]) -> tuple[list[str], list[str]]:
        """Load backbone weights only (non-strict). The head stays at its current
        initialization. Returns (missing_keys, unexpected_keys) like `load_state_dict`.

        Marcus transfer-experiment entry point: initialize a trig value net's
        backbone from an algebra checkpoint, then train both backbone and head
        from there (or freeze the backbone for a zero-shot probe).
        """
        # `strict=False` allows the head's own keys to remain at their init
        # rather than error on "missing keys: head.*". We do verify the user
        # didn't accidentally pass a full state by checking no head.* keys
        # leak through.
        leaked = [k for k in state.keys()
                  if any(k.startswith(p) for p in _VALUE_HEAD_PREFIXES)]
        if leaked:
            raise ValueError(
                f"load_backbone refusing to load: state contains head keys {leaked[:3]}; "
                "use `load_state_dict` for the full checkpoint instead"
            )
        return self.load_state_dict(state, strict=False)


class TreeLSTMValueNet(nn.Module):
    """Stub. Fall-back if GIN underperforms; implement on demand."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("TreeLSTMValueNet is a planned fallback; implement if GIN underperforms.")
