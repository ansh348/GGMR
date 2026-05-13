"""JSONL -> list of PyG Data with split-by-problem_id.

Schema (per `ggmr/training/extract_pairs.py`): each JSONL row has fields
`problem_id`, `remaining_steps`, `state_lhs_srepr`, `state_rhs_srepr`,
`var`, `excluded_srepr`, `source`, `family`, `template`, `depth`.

Split policy: dedupe by (lhs_srepr, rhs_srepr, remaining_steps) at parse
time, then group by problem_id, stratify by `source`, random split 80/10/10.
This avoids the leakage that would come from putting near-duplicate states
of the same problem in both train and val.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import sympy as sp
import torch
from torch_geometric.data import Data

from .graph import sympy_to_pyg
from .srepr_parse import parse_srepr

logger = logging.getLogger(__name__)


@dataclass
class RowMeta:
    problem_id: str
    source: str
    family: str | None
    template: str | None
    remaining_steps: int
    depth: int


class GGMRDataset:
    """Holds PyG Data + parallel meta. Indexing returns Data with .y attached."""

    def __init__(self, data_list: list[Data], meta: list[RowMeta]):
        if len(data_list) != len(meta):
            raise ValueError(f"data_list ({len(data_list)}) and meta ({len(meta)}) must align")
        self.data_list = data_list
        self.meta = meta

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]

    @classmethod
    def from_jsonl(cls, path: Path | str, *, dedupe: bool = True) -> "GGMRDataset":
        """Parse JSONL, dedupe at parse time, build graphs.

        If `dedupe=True`, rows with duplicate (state_lhs_srepr, state_rhs_srepr,
        remaining_steps) are skipped — keeps the first occurrence.
        Returns synthetic fallback if file is empty or missing.
        """
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            logger.warning(f"JSONL {p} missing or empty - using synthetic fallback (50 random graphs)")
            return cls._synthetic(50)

        data_list: list[Data] = []
        meta: list[RowMeta] = []
        seen_keys: set[tuple] = set()
        parse_fails = 0
        graph_fails = 0
        dedup_skips = 0
        total = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    parse_fails += 1
                    continue
                lhs_srepr = row.get("state_lhs_srepr", "")
                rhs_srepr = row.get("state_rhs_srepr", "")
                rs = int(row["remaining_steps"])
                if dedupe:
                    key = (lhs_srepr, rhs_srepr, rs)
                    if key in seen_keys:
                        dedup_skips += 1
                        continue
                    seen_keys.add(key)
                try:
                    lhs = parse_srepr(lhs_srepr)
                    rhs = parse_srepr(rhs_srepr)
                    var = sp.Symbol(row.get("var", "x"))
                    data = sympy_to_pyg(lhs, rhs, var)
                except Exception as e:
                    graph_fails += 1
                    if graph_fails <= 5:
                        logger.warning(f"graph build failed for problem {row.get('problem_id')}: {e}")
                    continue
                data.y = torch.tensor([math.log1p(rs)], dtype=torch.float32)
                data_list.append(data)
                meta.append(RowMeta(
                    problem_id=str(row.get("problem_id", "")),
                    source=str(row.get("source", "unknown")),
                    family=row.get("family"),
                    template=row.get("template"),
                    remaining_steps=rs,
                    depth=int(row.get("depth", 0)),
                ))
        logger.info(
            f"loaded {len(data_list)} rows from {p} "
            f"(total: {total}, json fails: {parse_fails}, graph fails: {graph_fails}, dedup skips: {dedup_skips})"
        )
        if not data_list:
            logger.warning("All rows failed to parse - falling back to synthetic")
            return cls._synthetic(50)
        return cls(data_list, meta)

    @classmethod
    def _synthetic(cls, n: int) -> "GGMRDataset":
        """Generate n random toy graphs for pipeline smoke-testing."""
        x_sym = sp.Symbol("x")
        rng = random.Random(42)
        data_list: list[Data] = []
        meta: list[RowMeta] = []
        for i in range(n):
            a = rng.randint(-5, 5)
            b = rng.randint(-10, 10)
            lhs = sp.Integer(a if a != 0 else 1) * x_sym + sp.Integer(b)
            rhs = sp.Integer(rng.randint(-10, 10))
            data = sympy_to_pyg(lhs, rhs, x_sym)
            rs = rng.randint(0, 5)
            data.y = torch.tensor([math.log1p(rs)], dtype=torch.float32)
            data_list.append(data)
            meta.append(RowMeta(
                problem_id=f"synthetic_{i:04d}",
                source="synthetic",
                family=None,
                template="linear",
                remaining_steps=rs,
                depth=1,
            ))
        return cls(data_list, meta)

    def split_by_problem_id(
        self, *, train: float = 0.8, val: float = 0.1, seed: int = 42
    ) -> tuple["GGMRDataset", "GGMRDataset", "GGMRDataset"]:
        """Group rows by problem_id, stratify by source, random split per stratum."""
        if train + val >= 1.0:
            raise ValueError(f"train + val must be < 1.0, got {train + val}")

        by_pid: dict[str, list[int]] = {}
        pid_source: dict[str, str] = {}
        for i, m in enumerate(self.meta):
            by_pid.setdefault(m.problem_id, []).append(i)
            pid_source[m.problem_id] = m.source

        strata: dict[str, list[str]] = {}
        for pid, src in pid_source.items():
            strata.setdefault(src, []).append(pid)

        rng = random.Random(seed)
        train_pids: set[str] = set()
        val_pids: set[str] = set()
        test_pids: set[str] = set()
        for src, pids in strata.items():
            pids_shuf = list(pids)
            rng.shuffle(pids_shuf)
            n = len(pids_shuf)
            n_train = max(1, int(round(n * train))) if n > 0 else 0
            n_val = max(1, int(round(n * val))) if n > 1 else 0
            # Edge case: tiny strata might end up empty in val/test
            n_train = min(n_train, n)
            n_val = min(n_val, n - n_train)
            train_pids.update(pids_shuf[:n_train])
            val_pids.update(pids_shuf[n_train:n_train + n_val])
            test_pids.update(pids_shuf[n_train + n_val:])

        def gather(pids: Iterable[str]) -> "GGMRDataset":
            idxs: list[int] = []
            for pid in pids:
                idxs.extend(by_pid.get(pid, []))
            return GGMRDataset(
                [self.data_list[i] for i in idxs],
                [self.meta[i] for i in idxs],
            )

        return gather(train_pids), gather(val_pids), gather(test_pids)
