"""LearnedHeuristic: GIN-backed value network wrapped as a Heuristic Protocol.

Drop-in replacement for `WeightedSumCompositeHeuristic` in A* / beam search.
On any internal failure (graph build, model forward, ckpt load), falls back
to the hand heuristic so A* never crashes -- critical because A* line 79
calls heuristic.evaluate(initial) outside try/except.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from ggmr.expr.tree import canonical_repr
from ggmr.state import EqState
from ggmr.training.graph import sympy_to_pyg
from ggmr.training.model import GINValueNet

from .composite import WeightedSumCompositeHeuristic

logger = logging.getLogger(__name__)


class LearnedHeuristic:
    """Implements the Heuristic Protocol from `ggmr/heuristics/composite.py`."""

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str = "cpu",
        cache_size: int = 50_000,
    ):
        self.device = device
        self.cache_size = cache_size
        self._fallback = WeightedSumCompositeHeuristic()
        self._cache: OrderedDict[tuple, float] = OrderedDict()

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
        logger.info(f"LearnedHeuristic loaded {ckpt_path} on {device}")

    def evaluate(self, state: EqState) -> float:
        try:
            key = (canonical_repr(state.lhs), canonical_repr(state.rhs), state.var.name)
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
            logger.warning(
                f"LearnedHeuristic fallback for {type(e).__name__}: {e}"
            )
            return self._fallback.evaluate(state)
