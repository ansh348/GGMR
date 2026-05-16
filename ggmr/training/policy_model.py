"""GIN policy network: predicts a probability distribution over the rule
names given an equation graph.

Same GIN backbone shape as `GINValueNet` (5 layers, 128 hidden, mean+max
readout). Separate weights — keeps gradient interference between value and
policy losses at bay during ExIt training. Output is raw logits over
`default_registry.names()` (canonical rule ordering); callers apply
legality masking + softmax via `PolicyAdvisor` (`ggmr/training/policy_heuristic.py`).

Shares the backbone/head split convention with `GINValueNet`: state-dict
keys ``convs.*`` are the transferable backbone, ``head.*`` are the
domain-specific policy head. See `backbone_state_dict` / `load_backbone`.
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

# Same prefix convention as GINValueNet. Keep in sync with the attribute
# names below.
_BACKBONE_PREFIXES: tuple[str, ...] = ("convs.",)
_POLICY_HEAD_PREFIXES: tuple[str, ...] = ("head.",)


def num_rules() -> int:
    """Number of rules in the canonical registry."""
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

    # --- transfer helpers ----------------------------------------------------

    def backbone_state_dict(self) -> dict[str, Tensor]:
        """Subset of state_dict containing only the GIN backbone weights.

        Identical convention to `GINValueNet.backbone_state_dict`. The
        backbone module shape (5 GINConv layers + readout) matches across
        value and policy nets, so a single saved `backbone.pt` can seed
        either head.
        """
        full = self.state_dict()
        return {k: v for k, v in full.items()
                if any(k.startswith(p) for p in _BACKBONE_PREFIXES)}

    def head_state_dict(self) -> dict[str, Tensor]:
        full = self.state_dict()
        return {k: v for k, v in full.items()
                if any(k.startswith(p) for p in _POLICY_HEAD_PREFIXES)}

    def load_backbone(self, state: dict[str, Tensor]) -> tuple[list[str], list[str]]:
        """Load backbone weights only (non-strict); policy head stays at init."""
        leaked = [k for k in state.keys()
                  if any(k.startswith(p) for p in _POLICY_HEAD_PREFIXES)]
        if leaked:
            raise ValueError(
                f"load_backbone refusing to load: state contains head keys {leaked[:3]}; "
                "use `load_state_dict` for the full checkpoint instead"
            )
        return self.load_state_dict(state, strict=False)


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
