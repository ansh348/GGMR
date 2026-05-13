"""Regression and ranking metrics for Phase 2 training/evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def pearsonr(x: Iterable[float], y: Iterable[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 if either is constant."""
    x = np.asarray(list(x), dtype=float).flatten()
    y = np.asarray(list(y), dtype=float).flatten()
    if len(x) < 2:
        return 0.0
    xm, ym = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((xm * xm).sum() * (ym * ym).sum()))
    if denom < 1e-12:
        return 0.0
    return float((xm * ym).sum() / denom)


def _ranks_avg_ties(a: np.ndarray) -> np.ndarray:
    """Return ranks with average rank for ties (1-indexed)."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and a[order[j]] == a[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearmanr(x: Iterable[float], y: Iterable[float]) -> float:
    """Spearman rank correlation. Average ranks for ties."""
    x_arr = np.asarray(list(x), dtype=float).flatten()
    y_arr = np.asarray(list(y), dtype=float).flatten()
    if len(x_arr) < 2:
        return 0.0
    return pearsonr(_ranks_avg_ties(x_arr), _ranks_avg_ties(y_arr))


def mae(preds: Iterable[float], targets: Iterable[float]) -> float:
    p = np.asarray(list(preds), dtype=float).flatten()
    t = np.asarray(list(targets), dtype=float).flatten()
    if len(p) == 0:
        return 0.0
    return float(np.mean(np.abs(p - t)))


def per_family_mae(
    preds: Iterable[float], targets: Iterable[float], families: Iterable[str]
) -> dict[str, float]:
    """Mean absolute error broken down by family label."""
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p, t, f in zip(preds, targets, families):
        buckets[str(f)].append((float(p), float(t)))
    return {
        fam: float(np.mean([abs(p - t) for p, t in rows])) if rows else 0.0
        for fam, rows in buckets.items()
    }


def geomean_ratio(numers: Iterable[float], denoms: Iterable[float]) -> float:
    """Geometric mean of (numer/denom) ratios over positive pairs.

    Skips pairs where numer<=0 or denom<=0. Returns 1.0 if no valid pairs.
    """
    logs: list[float] = []
    for n, d in zip(numers, denoms):
        n_f, d_f = float(n), float(d)
        if n_f > 0 and d_f > 0:
            logs.append(np.log(n_f) - np.log(d_f))
    if not logs:
        return 1.0
    return float(np.exp(np.mean(logs)))
