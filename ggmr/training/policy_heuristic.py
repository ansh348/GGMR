"""Network wrappers that adapt `GINValueNet` / `GINPolicyNet` to MCTS and A*.

`ValueAdvisor` reads a value net checkpoint and exposes a `value_fn(state)` that
returns a Q in [0, 1] via `steps_to_q(predicted_steps)`. Suitable to plug into
`mcts_search(value_fn=...)`.

`PolicyAdvisor` reads a policy net checkpoint and exposes:
  - `policy_fn(state, legal_rule_names) -> dict[str, float]` for MCTS priors
  - `action_ordering_key(state, actions) -> list[Action]` for A* with policy
    ordering (used in the policy-ordered eval mode in `ggmr.training.evaluate`).

Both wrappers cache per-state network outputs by `canonical_repr` to amortize
across repeated MCTS visits to the same node.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from ggmr.expr.tree import canonical_repr
from ggmr.rules.base import Action
from ggmr.rules.registry import default_registry
from ggmr.search.mcts import steps_to_q
from ggmr.state import EqState
from ggmr.training.graph import sympy_to_pyg
from ggmr.training.model import GINValueNet
from ggmr.training.policy_model import GINPolicyNet, masked_softmax, num_rules

logger = logging.getLogger(__name__)


def _state_cache_key(state: EqState) -> tuple:
    return (canonical_repr(state.lhs), canonical_repr(state.rhs), state.var.name)


class ValueAdvisor:
    """Adapter: value-net checkpoint -> MCTS value_fn returning Q in [0, 1]."""

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str = "cpu",
        cache_size: int = 50_000,
    ):
        self.device = device
        self.cache_size = cache_size
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        self._model = GINValueNet(
            in_dim=cfg.get("in_dim", 24),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 5),
            dropout=0.0,
        )
        self._model.load_state_dict(ckpt["model_state"])
        self._model.to(device).eval()
        self._target_transform = ckpt.get("target_transform", "log1p")
        self._cache: OrderedDict[tuple, float] = OrderedDict()

    def predicted_steps(self, state: EqState) -> float:
        """Run the value net forward and return predicted_steps clipped to [0, 30]."""
        try:
            key = _state_cache_key(state)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            data = sympy_to_pyg(state.lhs, state.rhs, state.var)
            batch = Batch.from_data_list([data]).to(self.device)
            with torch.no_grad():
                pred_raw = float(self._model(batch).item())
            if self._target_transform == "log1p":
                pred = float(np.clip(np.expm1(pred_raw), 0.0, 30.0))
            else:
                pred = float(np.clip(pred_raw, 0.0, 30.0))
            self._cache[key] = pred
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return pred
        except Exception as e:
            logger.warning(f"ValueAdvisor fallback for {type(e).__name__}: {e}")
            return 5.0  # neutral default; far enough from 0 to avoid spurious "solved"

    def value_fn(self, state: EqState) -> float:
        """Return Q in [0, 1] via `steps_to_q(predicted_steps)`."""
        return steps_to_q(self.predicted_steps(state))


class PolicyAdvisor:
    """Adapter: policy-net checkpoint -> MCTS policy_fn and A* action-ordering key.

    The policy net outputs `num_rules` raw logits in registry order. At inference,
    we mask illegal rules, softmax, and return a `rule_name -> probability` dict
    over the legal rules.
    """

    def __init__(
        self,
        ckpt_path: str | Path | None,
        device: str = "cpu",
        cache_size: int = 50_000,
    ):
        self.device = device
        self.cache_size = cache_size
        self._rule_names: list[str] = list(default_registry.names())
        self._name_to_idx: dict[str, int] = {n: i for i, n in enumerate(self._rule_names)}
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        if ckpt_path is None:
            self._model = None
            return
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        self._model = GINPolicyNet(
            in_dim=cfg.get("in_dim", 24),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 5),
            dropout=0.0,
            out_dim=cfg.get("out_dim", num_rules()),
        )
        self._model.load_state_dict(ckpt["model_state"])
        self._model.to(device).eval()

    def _logits(self, state: EqState) -> np.ndarray:
        """Return raw logits as a numpy array of shape [num_rules]."""
        if self._model is None:
            return np.zeros(num_rules(), dtype=np.float32)
        key = _state_cache_key(state)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        try:
            data = sympy_to_pyg(state.lhs, state.rhs, state.var)
            batch = Batch.from_data_list([data]).to(self.device)
            with torch.no_grad():
                logits = self._model(batch).squeeze(0).cpu().numpy()
        except Exception as e:
            logger.warning(f"PolicyAdvisor fallback for {type(e).__name__}: {e}")
            logits = np.zeros(num_rules(), dtype=np.float32)
        self._cache[key] = logits
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return logits

    def policy_fn(self, state: EqState, legal_rule_names: list[str]) -> dict[str, float]:
        """Return masked-softmax distribution over the legal rule names."""
        if not legal_rule_names:
            return {}
        logits = self._logits(state)
        legal_idx = [self._name_to_idx[n] for n in legal_rule_names if n in self._name_to_idx]
        if not legal_idx:
            p = 1.0 / len(legal_rule_names)
            return {n: p for n in legal_rule_names}
        legal_logits = logits[legal_idx]
        m = legal_logits.max()
        exp_logits = np.exp(legal_logits - m)
        denom = exp_logits.sum()
        if denom <= 0:
            p = 1.0 / len(legal_rule_names)
            return {n: p for n in legal_rule_names}
        probs = exp_logits / denom
        out = {n: 0.0 for n in legal_rule_names}
        for rn, p in zip([self._rule_names[i] for i in legal_idx], probs):
            out[rn] = float(p)
        return out

    def action_ordering_key(self, state: EqState, action: Action) -> float:
        """Return a sort key for A* action ordering: higher policy logit = explored first.

        Caller passes individual actions; sort with `key=lambda a: -advisor.action_ordering_key(state, a)`
        for descending order.
        """
        logits = self._logits(state)
        idx = self._name_to_idx.get(action.rule_name)
        if idx is None:
            return float("-inf")
        return float(logits[idx])
