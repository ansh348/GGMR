"""Transfer Experiment 1 (Marcus Phase 2): does an algebra-trained GIN backbone
bootstrap trig faster than scratch?

Three conditions on trig identity-verification data (`trig_training.jsonl`,
generated under `training_only=True` — Marcus Constraint 1):

    A. Trig scratch              — backbone + head random init
    B. Trig from algebra backbone — backbone loaded from algebra_backbone.pt,
                                     head random init
    C. Zero-shot probe           — backbone + value head BOTH loaded from
                                     algebra checkpoints; NO trig fine-tuning.
                                     Single forward pass on trig val split.

A and B use identical random seeds, identical data splits, identical batch
order, identical hyperparameters. The only difference is initialization,
isolating the contribution of transferred weights.

Outputs:
- transfer_exp_1_results.csv (one row per (condition, epoch))
- {output_dir}/A_scratch/best.pt          — condition A best
- {output_dir}/B_transfer/best.pt         — condition B best
- {output_dir}/A_scratch_backbone.pt      — backbone snapshot for reverse transfer
- {output_dir}/B_transfer_backbone.pt
- {output_dir}/summary.json               — headline numbers

Marcus primary metric: `val_pearson_log`. Headline finding: epochs-to-0.85
for B vs A (transfer efficiency) plus C's zero-shot value (structural transfer).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ggmr.rules.core  # noqa: F401  (register rules)
from ggmr.expr.tree import canonical_repr
from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
from ggmr.problems.loader import load_trig_evaluation_set
from ggmr.search.astar import astar
from ggmr.state import EqState
from ggmr.training.dataset import GGMRDataset
from ggmr.training.graph import FEATURE_DIM, sympy_to_pyg
from ggmr.training.metrics import mae, pearsonr, spearmanr
from ggmr.training.model import GINValueNet


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory heuristic adapter for A* eval
# ---------------------------------------------------------------------------


class _ModelHeuristic:
    """Heuristic Protocol adapter wrapping an in-memory GINValueNet.

    Mirrors LearnedHeuristic but skips the disk round-trip. Caches predictions
    by canonical_repr to avoid re-evaluating the same state.
    """

    def __init__(self, model: GINValueNet, device: str, cache_size: int = 50_000):
        self.model = model.eval()
        self.device = device
        self.cache_size = cache_size
        self._fallback = WeightedSumCompositeHeuristic()
        self._cache: OrderedDict[tuple, float] = OrderedDict()

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
                pred_raw = float(self.model(batch).item())
            pred = float(np.clip(np.expm1(pred_raw), 0.0, 30.0))
            self._cache[key] = pred
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return pred
        except Exception as e:  # noqa: BLE001
            logger.warning(f"_ModelHeuristic fallback: {type(e).__name__}: {e}")
            return self._fallback.evaluate(state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model: GINValueNet, loader: DataLoader, device: str) -> dict:
    """Compute val_loss, val_mae_step, val_pearson_log, val_spearman_log."""
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
    val_mae_step = mae(preds_step, targets_step)
    val_pearson_log = pearsonr(preds_log, targets_log)
    val_spearman_log = spearmanr(preds_log, targets_log)
    return {
        "val_loss": float(val_loss),
        "val_mae_step": float(val_mae_step),
        "val_pearson_log": float(val_pearson_log),
        "val_spearman_log": float(val_spearman_log),
    }


def _run_astar_eval(
    model: GINValueNet,
    problems: list,
    device: str,
    *,
    max_nodes: int = 5_000,
    max_depth: int = 20,
    training_only: bool = True,
) -> float:
    """Run A* on `problems` using `model` as the heuristic. Returns solve_rate.

    `training_only=True` honors Marcus Constraint 1 (no oracle in the search
    that the heuristic is being measured against).
    """
    if not problems:
        return 0.0
    heuristic = _ModelHeuristic(model, device)
    solved = 0
    for prob in problems:
        try:
            result = astar(
                prob.initial,
                prob.is_target,
                heuristic=heuristic,
                max_nodes=max_nodes,
                max_depth=max_depth,
                check_soundness=False,
                training_only=training_only,
                problem_id=prob.id,
            )
            if result.found:
                solved += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(f"A* eval crashed on {prob.id}: {type(e).__name__}: {e}")
    return solved / len(problems)


def _build_model(device: str, hidden: int = 128, num_layers: int = 5,
                 dropout: float = 0.1) -> GINValueNet:
    return GINValueNet(
        in_dim=FEATURE_DIM,
        hidden_dim=hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)


def _load_backbone_into(model: GINValueNet, backbone_path: Path, device: str) -> None:
    """Load `algebra_backbone.pt` (a {model_state: ...} dict) into model.convs.*
    weights. Head stays at its init.
    """
    ckpt = torch.load(backbone_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    missing, unexpected = model.load_backbone(state)
    # Missing keys for head.* are EXPECTED (we only loaded backbone).
    real_missing = [k for k in missing if not k.startswith("head.")]
    if real_missing:
        raise RuntimeError(f"backbone load missing keys: {real_missing[:5]}")
    if unexpected:
        logger.warning(f"backbone load unexpected keys (ignored): {unexpected[:5]}")


def _load_head_into(model: GINValueNet, head_path: Path, device: str) -> None:
    """Load `algebra_value_head.pt` (a {model_state: head.* keys}) into model.head.*"""
    ckpt = torch.load(head_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    # Use non-strict load — backbone keys would be missing here.
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith("convs.")]
    if real_missing:
        raise RuntimeError(f"head load missing keys: {real_missing[:5]}")


# ---------------------------------------------------------------------------
# Training driver (shared by conditions A and B)
# ---------------------------------------------------------------------------


def _train_one_condition(
    *,
    name: str,
    model: GINValueNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_problems: list,
    epochs: int,
    lr: float,
    device: str,
    eval_every: int,
    eval_astar_max_nodes: int,
    save_dir: Path,
    csv_writer,
    t_origin: float,
    bn_warmup: bool = False,
) -> list[dict]:
    """Train `model` for `epochs` epochs, log per-epoch metrics, save best.

    If `bn_warmup=True`, run one warmup pass over `train_loader` with the
    model in train() mode but with NO gradient updates — only the BatchNorm
    running stats are updated. Used by Condition B with `--warmup-b` to
    isolate the BN-stats-mismatch confound from the weights-mismatch confound.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    if bn_warmup:
        logger.info(f"[{name}] BN warmup: one pass through train_loader, lr=0")
        model.train()
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                _ = model(batch)  # forward only; BN stats update via train() mode

    opt = Adam(model.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=max(1, epochs))

    history: list[dict] = []
    best_val = float("inf")

    for epoch in range(epochs):
        # Train
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
        sched.step()

        # Eval (val_pearson_log every epoch)
        metrics = _evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["condition"] = name
        metrics["train_loss"] = float(train_loss)
        metrics["wall_time_s"] = round(time.perf_counter() - t_origin, 2)

        # A* solve rate at intervals (and final epoch)
        do_astar = (epoch % eval_every == 0) or (epoch == epochs - 1)
        if do_astar and eval_problems:
            metrics["solve_rate"] = _run_astar_eval(
                model, eval_problems, device, max_nodes=eval_astar_max_nodes
            )
        else:
            metrics["solve_rate"] = None

        # Persist best
        if metrics["val_loss"] < best_val - 1e-6:
            best_val = metrics["val_loss"]
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "in_dim": FEATURE_DIM,
                    "hidden_dim": model.hidden_dim,
                    "num_layers": model.num_layers,
                    "dropout": model.dropout,
                },
                "epoch": epoch,
                "val_loss": metrics["val_loss"],
                "val_pearson_log": metrics["val_pearson_log"],
                "target_transform": "log1p",
                "condition": name,
            }, save_dir / "best.pt")

        history.append(metrics)
        csv_writer.writerow({
            "condition": metrics["condition"],
            "epoch": metrics["epoch"],
            "val_pearson_log": f"{metrics['val_pearson_log']:.5f}",
            "val_mae_step": f"{metrics['val_mae_step']:.5f}",
            "solve_rate": f"{metrics['solve_rate']:.4f}" if metrics["solve_rate"] is not None else "",
            "wall_time_s": metrics["wall_time_s"],
        })
        logger.info(
            f"[{name}] ep {epoch:3d}  train {train_loss:.4f}  val {metrics['val_loss']:.4f}  "
            f"r {metrics['val_pearson_log']:.3f}  mae_step {metrics['val_mae_step']:.3f}"
            + (f"  solve_rate {metrics['solve_rate']:.3f}" if metrics["solve_rate"] is not None else "")
        )

    # Save backbone snapshot for reverse-transfer experiments
    torch.save({
        "model_state": model.backbone_state_dict(),
        "config": {
            "in_dim": FEATURE_DIM,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
        },
        "kind": "backbone",
        "source": f"{name}_final",
    }, save_dir.parent / f"{name}_backbone.pt")

    return history


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def _run_condition_a(
    *, dataset_path: Path, output_dir: Path, epochs: int, batch_size: int,
    seed: int, device: str, eval_every: int, eval_astar_max_nodes: int,
    eval_problems: list, csv_writer, t_origin: float,
) -> list[dict]:
    """Trig scratch."""
    _set_seed(seed)
    dataset = GGMRDataset.from_jsonl(dataset_path)
    train_ds, val_ds, _ = dataset.split_by_problem_id(seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    logger.info(f"[A] train={len(train_ds)} val={len(val_ds)}")

    _set_seed(seed)  # re-seed AFTER data split so weight init is deterministic
    model = _build_model(device)
    return _train_one_condition(
        name="A_scratch", model=model, train_loader=train_loader,
        val_loader=val_loader, eval_problems=eval_problems, epochs=epochs,
        lr=1e-3, device=device, eval_every=eval_every,
        eval_astar_max_nodes=eval_astar_max_nodes,
        save_dir=output_dir / "A_scratch",
        csv_writer=csv_writer, t_origin=t_origin,
    )


def _run_condition_b(
    *, dataset_path: Path, output_dir: Path, epochs: int, batch_size: int,
    seed: int, device: str, eval_every: int, eval_astar_max_nodes: int,
    algebra_backbone_path: Path, eval_problems: list, csv_writer, t_origin: float,
    bn_warmup: bool = False, name_suffix: str = "",
) -> list[dict]:
    """Trig from algebra backbone."""
    _set_seed(seed)
    dataset = GGMRDataset.from_jsonl(dataset_path)
    train_ds, val_ds, _ = dataset.split_by_problem_id(seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    logger.info(f"[B] train={len(train_ds)} val={len(val_ds)}")

    _set_seed(seed)
    model = _build_model(device)
    _load_backbone_into(model, algebra_backbone_path, device)
    logger.info(f"[B] loaded backbone from {algebra_backbone_path}")
    # Head stays at the post-`_set_seed(seed)` random init — identical to A's head init.

    name = f"B_transfer{name_suffix}"
    return _train_one_condition(
        name=name, model=model, train_loader=train_loader,
        val_loader=val_loader, eval_problems=eval_problems, epochs=epochs,
        lr=1e-3, device=device, eval_every=eval_every,
        eval_astar_max_nodes=eval_astar_max_nodes,
        save_dir=output_dir / name,
        csv_writer=csv_writer, t_origin=t_origin,
        bn_warmup=bn_warmup,
    )


def _run_condition_c(
    *, dataset_path: Path, algebra_backbone_path: Path, algebra_value_head_path: Path,
    seed: int, batch_size: int, device: str, eval_astar_max_nodes: int,
    eval_problems: list, csv_writer, t_origin: float,
) -> dict:
    """Zero-shot probe: load both backbone and value head from algebra; no
    trig training. Single forward pass on val split."""
    _set_seed(seed)
    dataset = GGMRDataset.from_jsonl(dataset_path)
    _, val_ds, _ = dataset.split_by_problem_id(seed=seed)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    logger.info(f"[C] val={len(val_ds)} (no training)")

    model = _build_model(device)
    _load_backbone_into(model, algebra_backbone_path, device)
    _load_head_into(model, algebra_value_head_path, device)
    logger.info(f"[C] loaded backbone + head from algebra checkpoints")

    metrics = _evaluate(model, val_loader, device)
    solve_rate = _run_astar_eval(
        model, eval_problems, device, max_nodes=eval_astar_max_nodes,
    ) if eval_problems else 0.0

    row = {
        "condition": "C_zero_shot",
        "epoch": 0,
        **metrics,
        "solve_rate": solve_rate,
        "wall_time_s": round(time.perf_counter() - t_origin, 2),
    }
    csv_writer.writerow({
        "condition": row["condition"],
        "epoch": row["epoch"],
        "val_pearson_log": f"{row['val_pearson_log']:.5f}",
        "val_mae_step": f"{row['val_mae_step']:.5f}",
        "solve_rate": f"{row['solve_rate']:.4f}",
        "wall_time_s": row["wall_time_s"],
    })
    logger.info(
        f"[C_zero_shot] val_pearson_log={metrics['val_pearson_log']:.3f}  "
        f"val_mae_step={metrics['val_mae_step']:.3f}  solve_rate={solve_rate:.3f}"
    )
    return row


# ---------------------------------------------------------------------------
# Headline summary
# ---------------------------------------------------------------------------


def _epochs_to_pearson(history: list[dict], target: float = 0.85) -> Optional[int]:
    for row in history:
        if row.get("val_pearson_log", -1.0) >= target:
            return int(row["epoch"])
    return None


def _final_pearson(history: list[dict]) -> float:
    return float(history[-1]["val_pearson_log"]) if history else float("nan")


def _summarize(history_a: list[dict], history_b: list[dict],
               row_c: dict, output_dir: Path) -> dict:
    epochs_a_to_85 = _epochs_to_pearson(history_a, 0.85)
    epochs_b_to_85 = _epochs_to_pearson(history_b, 0.85)
    summary = {
        "condition_A_final_pearson": _final_pearson(history_a),
        "condition_B_final_pearson": _final_pearson(history_b),
        "condition_C_pearson": float(row_c["val_pearson_log"]),
        "condition_A_solve_rate_final": float(history_a[-1].get("solve_rate") or 0.0) if history_a else 0.0,
        "condition_B_solve_rate_final": float(history_b[-1].get("solve_rate") or 0.0) if history_b else 0.0,
        "condition_C_solve_rate": float(row_c["solve_rate"]),
        "A_epochs_to_0.85": epochs_a_to_85,
        "B_epochs_to_0.85": epochs_b_to_85,
    }
    if epochs_a_to_85 is not None and epochs_b_to_85 is not None:
        summary["transfer_speedup_at_0.85"] = epochs_a_to_85 / max(1, epochs_b_to_85)
        summary["headline"] = (
            f"B reached val_pearson_log >= 0.85 in {epochs_b_to_85} epochs vs "
            f"A in {epochs_a_to_85} epochs "
            f"(speedup {summary['transfer_speedup_at_0.85']:.2f}x). "
            f"C zero-shot pearson: {summary['condition_C_pearson']:.3f}"
        )
    else:
        summary["headline"] = (
            f"Neither A nor B reached 0.85 within {len(history_a)} epochs. "
            f"A final={summary['condition_A_final_pearson']:.3f}, "
            f"B final={summary['condition_B_final_pearson']:.3f}. "
            f"C zero-shot pearson: {summary['condition_C_pearson']:.3f}"
        )
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="trig_training.jsonl")
    parser.add_argument("--algebra-backbone", type=Path, required=True)
    parser.add_argument("--algebra-value-head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="run A* solve-rate eval every N epochs (also at final epoch)")
    parser.add_argument("--eval-astar-max-nodes", type=int, default=5000,
                        help="A* node budget per problem during periodic eval")
    parser.add_argument("--eval-difficulty", default="easy_medium",
                        choices=["easy", "easy_medium", "all"],
                        help="trig eval split for A* solve_rate")
    parser.add_argument("--skip-a", action="store_true",
                        help="skip Condition A (scratch); useful for re-running just B/C")
    parser.add_argument("--skip-b", action="store_true")
    parser.add_argument("--skip-c", action="store_true")
    parser.add_argument("--no-astar", action="store_true",
                        help="skip A* solve-rate eval entirely (val_pearson_log only). "
                             "Trained-model A* heuristic explores wide on trig when the "
                             "model predicts uniform values, making eval slow. Useful for "
                             "fast iteration when val_pearson_log is the only metric needed.")
    parser.add_argument("--also-warmup-b", action="store_true",
                        help="In addition to plain Condition B, run B_warmup variant: "
                             "1 pass through train data with no gradient updates (BN "
                             "running stats recalibrate to trig) BEFORE normal training. "
                             "Isolates the BN-mismatch confound from weight-mismatch.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load A* eval problems (subset of trig eval set)
    if args.no_astar:
        eval_problems: list = []
        logger.info("A* eval DISABLED (--no-astar)")
    else:
        all_problems = load_trig_evaluation_set()
        if args.eval_difficulty == "easy":
            eval_problems = [p for p in all_problems if p.id.startswith("trig_e")]
        elif args.eval_difficulty == "easy_medium":
            eval_problems = [p for p in all_problems
                             if p.id.startswith("trig_e") or p.id.startswith("trig_m")]
        else:
            eval_problems = all_problems
        logger.info(f"A* eval set: {len(eval_problems)} problems ({args.eval_difficulty})")

    csv_path = args.output_dir / "transfer_exp_1_results.csv"
    csv_fields = ["condition", "epoch", "val_pearson_log", "val_mae_step",
                  "solve_rate", "wall_time_s"]
    t_origin = time.perf_counter()

    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
        writer.writeheader()

        # ---- Condition C (fast — runs first) -------------------------------
        row_c = {}
        if not args.skip_c:
            row_c = _run_condition_c(
                dataset_path=args.data,
                algebra_backbone_path=args.algebra_backbone,
                algebra_value_head_path=args.algebra_value_head,
                seed=args.seed, batch_size=args.batch_size, device=args.device,
                eval_astar_max_nodes=args.eval_astar_max_nodes,
                eval_problems=eval_problems,
                csv_writer=writer, t_origin=t_origin,
            )
            csv_f.flush()

        # ---- Condition A (scratch baseline) --------------------------------
        history_a: list[dict] = []
        if not args.skip_a:
            history_a = _run_condition_a(
                dataset_path=args.data, output_dir=args.output_dir,
                epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                device=args.device, eval_every=args.eval_every,
                eval_astar_max_nodes=args.eval_astar_max_nodes,
                eval_problems=eval_problems,
                csv_writer=writer, t_origin=t_origin,
            )
            csv_f.flush()

        # ---- Condition B (transfer) ----------------------------------------
        history_b: list[dict] = []
        history_b_warmup: list[dict] = []
        if not args.skip_b:
            history_b = _run_condition_b(
                dataset_path=args.data, output_dir=args.output_dir,
                epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                device=args.device, eval_every=args.eval_every,
                eval_astar_max_nodes=args.eval_astar_max_nodes,
                algebra_backbone_path=args.algebra_backbone,
                eval_problems=eval_problems,
                csv_writer=writer, t_origin=t_origin,
            )
            csv_f.flush()

            if args.also_warmup_b:
                history_b_warmup = _run_condition_b(
                    dataset_path=args.data, output_dir=args.output_dir,
                    epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                    device=args.device, eval_every=args.eval_every,
                    eval_astar_max_nodes=args.eval_astar_max_nodes,
                    algebra_backbone_path=args.algebra_backbone,
                    eval_problems=eval_problems,
                    csv_writer=writer, t_origin=t_origin,
                    bn_warmup=True, name_suffix="_warmup",
                )
                csv_f.flush()

    summary = _summarize(history_a, history_b, row_c, args.output_dir)
    if history_b_warmup:
        summary["condition_B_warmup_final_pearson"] = _final_pearson(history_b_warmup)
        summary["B_warmup_epochs_to_0.85"] = _epochs_to_pearson(history_b_warmup, 0.85)
    print("=" * 70)
    print("Transfer Experiment 1 — summary")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"Results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
