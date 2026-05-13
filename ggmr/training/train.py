"""Phase 2 GIN training CLI.

    python -m ggmr.training.train --data <path> --epochs 100 \
        --batch-size 64 --device cuda --output checkpoints/<run>

Trains GINValueNet on log1p(remaining_steps), MSE loss. Early stops on
val_loss with patience 15. Saves best.pt with full metadata (config,
git_sha, dataset_sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from .dataset import GGMRDataset
from .graph import FEATURE_DIM, NODE_TYPE_VOCAB
from .metrics import mae, pearsonr, per_family_mae, spearmanr
from .model import GINValueNet

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to training_data.jsonl")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", required=True, help="checkpoint directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return "unavailable"
    return h.hexdigest()


def evaluate(model, loader, device, dataset: GGMRDataset, indices: list[int] | None = None):
    """Run forward over loader, return (val_loss, mae_step, mae_step_cond, pearson, spearman, preds, targets)."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    preds_log: list[float] = []
    targets_log: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(-1)
            loss = F.mse_loss(pred, y, reduction="sum")
            total_loss += loss.item()
            total_n += y.numel()
            preds_log.extend(pred.cpu().tolist())
            targets_log.extend(y.cpu().tolist())
    val_loss = total_loss / max(total_n, 1)
    preds_step = np.clip(np.expm1(np.array(preds_log)), 0.0, 30.0)
    targets_step = np.expm1(np.array(targets_log))
    mae_step = mae(preds_step, targets_step)
    # MAE conditional on remaining_steps > 0
    mask = targets_step > 0.5
    mae_step_cond = mae(preds_step[mask], targets_step[mask]) if mask.any() else 0.0
    r = pearsonr(preds_log, targets_log)
    rho = spearmanr(preds_log, targets_log)
    return val_loss, mae_step, mae_step_cond, r, rho, preds_log, targets_log


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    dataset = GGMRDataset.from_jsonl(args.data)
    train_ds, val_ds, test_ds = dataset.split_by_problem_id(seed=args.seed)
    logger.info(f"split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("Empty train or val split - cannot train.")
        return 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = args.device
    model = GINValueNet(
        in_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model: {n_params:,} params on {device}")

    opt = Adam(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    best_val = float("inf")
    patience = 0
    history: list[dict] = []
    config = {
        "in_dim": FEATURE_DIM,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    metadata = {
        "git_sha": _git_sha(),
        "dataset_path": str(args.data),
        "dataset_sha256": _file_sha256(Path(args.data)),
        "target_transform": "log1p",
        "node_vocab": list(NODE_TYPE_VOCAB),
        "config": config,
        "seed": args.seed,
    }

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            y = batch.y.view(-1)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.numel()
            total_n += y.numel()
        train_loss = total_loss / max(total_n, 1)

        val_loss, val_mae, val_mae_cond, val_r, val_rho, _, _ = evaluate(
            model, val_loader, device, val_ds
        )
        sched.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae_step": val_mae,
            "val_mae_step_cond": val_mae_cond,
            "val_pearson_log": val_r,
            "val_spearman_log": val_rho,
            "lr": opt.param_groups[0]["lr"],
        })
        logger.info(
            f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"mae_step {val_mae:.3f}  mae_cond {val_mae_cond:.3f}  r {val_r:.3f}  rho {val_rho:.3f}"
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience = 0
            ckpt = {
                "model_state": model.state_dict(),
                **metadata,
                "epoch": epoch,
                "val_loss": val_loss,
            }
            torch.save(ckpt, out / "best.pt")
        else:
            patience += 1
            if patience >= args.patience:
                logger.info(f"early stopping at epoch {epoch} (best val {best_val:.4f})")
                break

    elapsed = time.perf_counter() - t0
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Per-family MAE on val (after expm1)
    val_loss, val_mae, val_mae_cond, val_r, val_rho, preds_log, targets_log = evaluate(
        model, val_loader, device, val_ds
    )
    preds_step = np.clip(np.expm1(np.array(preds_log)), 0.0, 30.0).tolist()
    targets_step = np.expm1(np.array(targets_log)).tolist()
    fam_keys = [m.family or m.source for m in val_ds.meta]
    pf_mae = per_family_mae(preds_step, targets_step, fam_keys)

    print("=" * 60)
    print("training summary")
    print("=" * 60)
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  best val_loss: {best_val:.4f}")
    print(f"  final val_mae_step: {val_mae:.3f}")
    print(f"  final val_mae_step_cond: {val_mae_cond:.3f}  (remaining_steps > 0)")
    print(f"  final val_pearson_log: {val_r:.3f}")
    print(f"  final val_spearman_log: {val_rho:.3f}")
    print(f"  per-family MAE:")
    for fam, m in sorted(pf_mae.items()):
        print(f"    {fam}: {m:.3f}")
    print(f"  checkpoint: {out / 'best.pt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
