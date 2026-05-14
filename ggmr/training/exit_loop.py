"""Expert Iteration (ExIt) training loop: MCTS -> trajectories -> SL on value + policy.

The main entry point is `run_exit_iteration`, which:
  1. Runs MCTS on a list of (initial, target, is_target) problems
  2. Collects per-step (state, MCTS-discovered-remaining-steps, visit-distribution) tuples
  3. Appends to a `ReplayBuffer`
  4. Trains value net (MSE on log1p targets) and policy net (KL vs MCTS distribution)
  5. Returns the updated checkpoint paths

`pre_train_policy_on_bfs` consumes the warm-start JSONL from
`scripts/extract_policy_warmstart_data.py` and trains the policy net with
cross-entropy against the BFS-optimal rule_name.

`run_exit` orchestrates `pre_train` -> N iterations -> final eval.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import random
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import sympy as sp
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from ggmr.expr.tree import canonical_repr
from ggmr.problems.round2_categories import CATEGORIES
from ggmr.rules.registry import default_registry
from ggmr.search.mcts import MCTSResult, mcts_search, steps_to_q
from ggmr.state import EqState
from ggmr.training.extract_pairs import _build_is_target
from ggmr.training.graph import sympy_to_pyg
from ggmr.training.model import GINValueNet
from ggmr.training.policy_heuristic import PolicyAdvisor, ValueAdvisor
from ggmr.training.policy_model import GINPolicyNet, masked_log_softmax, num_rules
from ggmr.training.srepr_parse import parse_srepr

logger = logging.getLogger(__name__)


# -------------------------- problem + tuple types ---------------------------


@dataclass
class ExitProblem:
    """One ExIt training problem: an (initial, target) pair plus metadata."""
    problem_id: str
    initial: EqState
    target: EqState
    category: str = ""

    @property
    def is_target(self):
        return _build_is_target(self.target)


@dataclass
class ValueTuple:
    """(state_graph, log1p_target) for value training."""
    graph: Data
    log1p_steps: float
    problem_id: str


@dataclass
class PolicyTuple:
    """(state_graph, distribution_over_rule_names) for policy training.

    `legal_mask` is a [num_rules] vector with 1.0 on legal entries and 0.0
    elsewhere — used by the masked-softmax loss.
    """
    graph: Data
    target_distribution: np.ndarray  # [num_rules], sums to 1.0 over legal rules
    legal_mask: np.ndarray  # [num_rules], 0/1


# ----------------------------- replay buffer --------------------------------


class ReplayBuffer:
    """FIFO buffer of (ValueTuple, PolicyTuple) tuples, capped at `max_size`.

    Both halves are kept in lockstep so they share the same state ordering.
    `policy_tuple` may be None if a record only contributes a value target
    (e.g., warm-start data has policy but no MCTS value; we still mirror the
    rule-name as a one-hot policy target).
    """

    def __init__(self, max_size: int = 50_000, held_out_pids: set[str] | None = None):
        self.max_size = max_size
        self._held_out_pids: set[str] = set(held_out_pids or ())
        self._value_buf: deque[ValueTuple] = deque()
        self._policy_buf: deque[PolicyTuple | None] = deque()
        self._pids: deque[str] = deque()

    def add(self, value_tuple: ValueTuple, policy_tuple: PolicyTuple | None) -> None:
        if value_tuple.problem_id in self._held_out_pids:
            raise AssertionError(
                f"refusing to add held-out problem_id {value_tuple.problem_id} to buffer"
            )
        self._value_buf.append(value_tuple)
        self._policy_buf.append(policy_tuple)
        self._pids.append(value_tuple.problem_id)
        while len(self._value_buf) > self.max_size:
            self._value_buf.popleft()
            self._policy_buf.popleft()
            self._pids.popleft()

    def __len__(self) -> int:
        return len(self._value_buf)

    def value_tuples(self) -> list[ValueTuple]:
        return list(self._value_buf)

    def policy_tuples(self) -> list[PolicyTuple]:
        return [p for p in self._policy_buf if p is not None]

    def sanity_check_no_leak(self, eval_pids: set[str]) -> None:
        """Assert that no evaluation problem_id leaked into training data."""
        train_pids = set(self._pids)
        leak = train_pids & eval_pids
        if leak:
            raise AssertionError(f"replay buffer leaked eval pids: {sorted(leak)[:5]}")


# ----------------------- warm-start data loading ----------------------------


def _build_warmstart_graph(record: dict) -> Data | None:
    try:
        lhs = parse_srepr(record["state_lhs_srepr"])
        rhs = parse_srepr(record["state_rhs_srepr"])
        var = sp.Symbol(record.get("var", "x"))
        return sympy_to_pyg(lhs, rhs, var)
    except Exception as e:
        logger.debug(f"warmstart graph build failed: {e}")
        return None


def load_warmstart_jsonl(path: Path | str) -> tuple[list[dict], list[dict]]:
    """Load warmstart JSONL, return (train_records, held_out_records).

    Each record has fields including `state_lhs_srepr`, `state_rhs_srepr`,
    `var`, `rule_name`, `split`, `problem_id`.
    """
    train, held_out = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("split") == "held_out":
                held_out.append(r)
            else:
                train.append(r)
    return train, held_out


# ----------------------------- training steps -------------------------------


def _rule_name_to_idx() -> dict[str, int]:
    return {n: i for i, n in enumerate(default_registry.names())}


def train_policy_step(
    policy_net: GINPolicyNet,
    optimizer: Adam,
    batch: Batch,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    loss_type: str = "kl",
) -> float:
    """One optimizer step. `targets` is either:
      - long [B] of class indices (loss_type='ce'), or
      - float [B, num_rules] of probabilities (loss_type='kl').
    `legal_mask` is float [B, num_rules]. Illegal entries get -inf logits.
    """
    policy_net.train()
    logits = policy_net(batch)
    log_probs = masked_log_softmax(logits, legal_mask)
    if loss_type == "ce":
        loss = F.nll_loss(log_probs, targets)
    elif loss_type == "kl":
        # KL(target || pred) = sum target * (log target - log pred). Drop constant term.
        target_dist = targets
        # Avoid log(0) by masking the contribution where target=0
        mask = target_dist > 0
        contrib = torch.where(mask, target_dist * (-log_probs), torch.zeros_like(target_dist))
        loss = contrib.sum(dim=-1).mean()
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_value_step(value_net: GINValueNet, optimizer: Adam, batch: Batch) -> float:
    value_net.train()
    pred = value_net(batch)
    y = batch.y.view(-1)
    loss = F.mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


# ------------------------- policy pre-training ------------------------------


def _build_policy_dataset_from_warmstart(records: list[dict]) -> tuple[list[Data], list[int], list[np.ndarray]]:
    """Build PyG datas, integer rule labels, and per-state legal masks.

    The legal mask is derived from `default_registry.enumerate_actions(state)`
    to keep the cross-entropy/masked-softmax loss honest.
    """
    rn_to_idx = _rule_name_to_idx()
    n_rules = num_rules()
    datas: list[Data] = []
    labels: list[int] = []
    masks: list[np.ndarray] = []
    for r in records:
        graph = _build_warmstart_graph(r)
        if graph is None:
            continue
        rule_name = r.get("rule_name")
        if rule_name not in rn_to_idx:
            continue
        try:
            lhs = parse_srepr(r["state_lhs_srepr"])
            rhs = parse_srepr(r["state_rhs_srepr"])
            var = sp.Symbol(r.get("var", "x"))
            state = EqState(lhs=lhs, rhs=rhs, var=var)
        except Exception:
            continue
        mask = np.zeros(n_rules, dtype=np.float32)
        any_legal = False
        for rule, action in default_registry.enumerate_actions(state):
            if rule.guard(state, action).ok:
                mask[rn_to_idx[rule.name]] = 1.0
                any_legal = True
        if not any_legal or mask[rn_to_idx[rule_name]] == 0.0:
            continue
        datas.append(graph)
        labels.append(rn_to_idx[rule_name])
        masks.append(mask)
    return datas, labels, masks


def _collate_policy_batch(
    datas: list[Data],
    labels: list[int],
    masks: list[np.ndarray],
    indices: list[int],
    device: str,
) -> tuple[Batch, torch.Tensor, torch.Tensor]:
    batch = Batch.from_data_list([datas[i] for i in indices]).to(device)
    y = torch.tensor([labels[i] for i in indices], dtype=torch.long).to(device)
    m = torch.tensor(np.stack([masks[i] for i in indices], axis=0)).to(device)
    return batch, y, m


def pre_train_policy_on_bfs(
    *,
    warmstart_path: Path | str,
    output_ckpt: Path | str,
    device: str = "cpu",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Pre-train GINPolicyNet on warm-start (state, optimal_rule_name) pairs.

    Returns a stats dict with held-out top-1 accuracy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_recs, held_out_recs = load_warmstart_jsonl(warmstart_path)
    logger.info(f"warmstart: {len(train_recs)} train + {len(held_out_recs)} held_out")

    train_datas, train_labels, train_masks = _build_policy_dataset_from_warmstart(train_recs)
    held_datas, held_labels, held_masks = _build_policy_dataset_from_warmstart(held_out_recs)
    logger.info(f"built train={len(train_datas)} held_out={len(held_datas)} examples")

    net = GINPolicyNet().to(device)
    opt = Adam(net.parameters(), lr=lr)

    n = len(train_datas)
    epoch_losses: list[float] = []
    held_top1: list[float] = []
    for epoch in range(epochs):
        net.train()
        order = list(range(n))
        random.shuffle(order)
        epoch_loss = 0.0
        n_steps = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            if len(idx) < 2:  # BatchNorm needs >=2 samples
                continue
            batch, y, mask = _collate_policy_batch(
                train_datas, train_labels, train_masks, idx, device
            )
            loss = train_policy_step(net, opt, batch, y, mask, loss_type="ce")
            epoch_loss += loss
            n_steps += 1
        avg_loss = epoch_loss / max(n_steps, 1)
        epoch_losses.append(avg_loss)

        # Held-out top-1 accuracy
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, len(held_datas), batch_size):
                idx = list(range(start, min(start + batch_size, len(held_datas))))
                if len(idx) < 2:
                    continue
                batch, y, mask = _collate_policy_batch(
                    held_datas, held_labels, held_masks, idx, device
                )
                logits = net(batch)
                masked = logits.masked_fill(mask < 0.5, float("-inf"))
                pred = masked.argmax(dim=-1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        acc = correct / max(total, 1)
        held_top1.append(acc)
        logger.info(f"epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} held_top1={acc:.3f}")

    # Save checkpoint
    output_ckpt = Path(output_ckpt)
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": net.state_dict(),
        "config": {
            "in_dim": net.in_dim,
            "hidden_dim": net.hidden_dim,
            "num_layers": net.num_layers,
            "dropout": net.dropout,
            "out_dim": net.out_dim,
        },
        "epoch": epochs,
        "held_out_top1": held_top1[-1] if held_top1 else 0.0,
        "rule_names": list(default_registry.names()),
    }, output_ckpt)
    logger.info(f"saved policy ckpt to {output_ckpt}")

    return {
        "epoch_losses": epoch_losses,
        "held_top1": held_top1,
        "final_held_top1": held_top1[-1] if held_top1 else 0.0,
        "train_size": len(train_datas),
        "held_out_size": len(held_datas),
    }


# ----------------------- MCTS trajectory collection -------------------------


def _verify_trajectory_soundness(
    initial: EqState, path: list, target_is_target: Callable[[EqState], bool]
) -> bool:
    """Replay the trajectory step by step, applying each action's rule, and confirm
    each transition matches the expected child state (under canonical_repr)."""
    from ggmr.expr.tree import normalize
    from ggmr.rules.base import merge_guard_into_state

    state = initial
    for parent_state, action in path:
        if canonical_repr(parent_state.lhs) != canonical_repr(state.lhs) or \
           canonical_repr(parent_state.rhs) != canonical_repr(state.rhs):
            return False
        rule = default_registry.get(action.rule_name)
        guard = rule.guard(state, action)
        if not guard.ok:
            return False
        try:
            child = rule.apply(state, action)
        except Exception:
            return False
        if guard.new_excluded or guard.new_side_conditions:
            child = merge_guard_into_state(child, guard)
        child = child.with_lhs_rhs(normalize(child.lhs), normalize(child.rhs))
        state = child
    return target_is_target(state)


# --- helpers for ProcessPoolExecutor IPC (sympy expressions don't pickle cleanly;
#     we serialize via srepr in the main process and reconstruct in workers).

def _serialize_state(state: EqState) -> dict:
    return {
        "lhs": sp.srepr(state.lhs),
        "rhs": sp.srepr(state.rhs),
        "var": state.var.name,
        "excluded": sorted(sp.srepr(e) for e in state.excluded),
    }


def _deserialize_state(d: dict) -> EqState:
    excluded = tuple(parse_srepr(s) for s in d.get("excluded", ()))
    return EqState(
        lhs=parse_srepr(d["lhs"]),
        rhs=parse_srepr(d["rhs"]),
        var=sp.Symbol(d.get("var", "x")),
        excluded=frozenset(excluded),
    )


def _serialize_problem(prob: ExitProblem) -> dict:
    return {
        "problem_id": prob.problem_id,
        "category": prob.category,
        "initial": _serialize_state(prob.initial),
        "target": _serialize_state(prob.target),
    }


def _set_alarm(seconds: int) -> None:
    """Arm a SIGALRM (Unix only). On Windows, no-op — MCTS is bounded by num_simulations × max_moves."""
    if platform.system() == "Windows" or seconds <= 0:
        return
    import signal

    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"MCTS worker exceeded {seconds}s timeout")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)


def _clear_alarm() -> None:
    if platform.system() == "Windows":
        return
    import signal
    signal.alarm(0)


def _mcts_worker_fn(args: tuple) -> dict:
    """Worker process: load model copies from disk, run MCTS on one problem,
    return a serializable result dict. Trajectory soundness is verified here
    (before serialization) so the main process can trust ``found=True`` results."""
    (problem_dict, value_ckpt_path, policy_ckpt_path,
     num_simulations, max_moves, c_puct, timeout_s) = args

    # Register rules in this subprocess
    import ggmr.rules.core  # noqa: F401

    pid = problem_dict.get("problem_id", "<unknown>")

    _set_alarm(timeout_s)
    try:
        initial = _deserialize_state(problem_dict["initial"])
        target = _deserialize_state(problem_dict["target"])
        is_target = _build_is_target(target)

        va = ValueAdvisor(value_ckpt_path, device="cpu")
        pa = PolicyAdvisor(policy_ckpt_path, device="cpu")

        result = mcts_search(
            initial, is_target,
            value_fn=va.value_fn, policy_fn=pa.policy_fn,
            num_simulations=num_simulations, max_moves=max_moves, c_puct=c_puct,
        )
    except TimeoutError:
        return {"problem_id": pid, "found": False, "reason": "timeout",
                "total_simulations": 0}
    except Exception as e:  # noqa: BLE001
        return {"problem_id": pid, "found": False,
                "reason": f"{type(e).__name__}: {e}", "total_simulations": 0}
    finally:
        _clear_alarm()

    if not result.found:
        return {"problem_id": pid, "found": False, "reason": "not_solved",
                "total_simulations": result.stats.total_simulations}

    try:
        sound = _verify_trajectory_soundness(initial, result.path, is_target)
    except Exception:  # noqa: BLE001
        sound = False
    if not sound:
        return {"problem_id": pid, "found": False, "reason": "unsound",
                "total_simulations": result.stats.total_simulations}

    path_records: list[dict] = []
    for (state, action), dist in zip(result.path, result.visit_distributions):
        path_records.append({
            "state": _serialize_state(state),
            "action_rule": action.rule_name,
            "visit_distribution": dict(dist),
        })

    return {
        "problem_id": pid,
        "found": True,
        "path_records": path_records,
        "total_simulations": result.stats.total_simulations,
    }


def _build_tuples_from_worker_result(
    result: dict,
    rn_to_idx: dict[str, int],
    n_rules: int,
) -> tuple[list[ValueTuple], list[PolicyTuple]]:
    """In the main process, turn a worker result's serialized trajectory into
    (ValueTuple, PolicyTuple) lists. We rebuild PyG graphs here because Data
    objects don't pickle reliably across processes."""
    pid = result["problem_id"]
    path_records = result["path_records"]
    n_steps = len(path_records)
    value_tuples: list[ValueTuple] = []
    policy_tuples: list[PolicyTuple] = []
    for i, rec in enumerate(path_records):
        try:
            state = _deserialize_state(rec["state"])
            graph = sympy_to_pyg(state.lhs, state.rhs, state.var)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"failed to rebuild graph for {pid}/{i}: {e}")
            continue
        remaining = n_steps - i
        v_data = Data(x=graph.x, edge_index=graph.edge_index)
        v_data.y = torch.tensor([math.log1p(remaining)], dtype=torch.float32)
        value_tuples.append(ValueTuple(
            graph=v_data, log1p_steps=math.log1p(remaining), problem_id=pid,
        ))

        # Legal mask + visit-derived policy target (recomputed; cheap & avoids
        # trusting the worker to enumerate legals).
        mask = np.zeros(n_rules, dtype=np.float32)
        target = np.zeros(n_rules, dtype=np.float32)
        for rule, action in default_registry.enumerate_actions(state):
            if rule.guard(state, action).ok:
                mask[rn_to_idx[rule.name]] = 1.0
        for rn, p in rec.get("visit_distribution", {}).items():
            if rn in rn_to_idx:
                target[rn_to_idx[rn]] = float(p)
        s = target.sum()
        if s > 0:
            target /= s
        elif mask.sum() > 0:
            target = mask / mask.sum()
        p_graph = Data(x=graph.x, edge_index=graph.edge_index)
        policy_tuples.append(PolicyTuple(
            graph=p_graph, target_distribution=target, legal_mask=mask,
        ))
    return value_tuples, policy_tuples


def collect_mcts_trajectories(
    problems: list[ExitProblem],
    *,
    value_ckpt_path: str | Path,
    policy_ckpt_path: str | Path,
    num_simulations: int,
    max_moves: int,
    c_puct: float = 1.5,
    max_workers: int | None = None,
    timeout_per_problem: int = 120,
    progress_every: int = 50,
    collection_timeout_s: int | None = 3600,
) -> tuple[list[ValueTuple], list[PolicyTuple], dict]:
    """Run MCTS on each problem in parallel via ProcessPoolExecutor.

    Each worker loads its own copy of the value + policy nets from disk
    (`value_ckpt_path`, `policy_ckpt_path` must be saved BEFORE this call —
    `run_one_iteration` writes "inflight" checkpoints to satisfy this contract).
    Workers verify trajectory soundness before returning. Per-problem timeout
    is enforced via SIGALRM on Unix; on Windows MCTS is bounded only by
    num_simulations × max_moves.
    """
    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 4)
    value_ckpt_path = str(value_ckpt_path)
    policy_ckpt_path = str(policy_ckpt_path)

    rn_to_idx = _rule_name_to_idx()
    n_rules = num_rules()

    job_args = [
        (
            _serialize_problem(prob), value_ckpt_path, policy_ckpt_path,
            num_simulations, max_moves, c_puct, timeout_per_problem,
        )
        for prob in problems
    ]

    value_tuples: list[ValueTuple] = []
    policy_tuples: list[PolicyTuple] = []
    n_solved = 0
    n_failed = 0
    n_timeouts = 0
    n_unsound = 0
    total_sims = 0
    t0 = time.perf_counter()

    def _consume(result: dict) -> None:
        nonlocal n_solved, n_failed, n_timeouts, n_unsound, total_sims
        total_sims += int(result.get("total_simulations", 0) or 0)
        if result.get("found"):
            n_solved += 1
            vts, pts = _build_tuples_from_worker_result(result, rn_to_idx, n_rules)
            value_tuples.extend(vts)
            policy_tuples.extend(pts)
        else:
            reason = result.get("reason", "unknown")
            if reason == "timeout":
                n_timeouts += 1
            elif reason == "unsound":
                n_unsound += 1
            else:
                n_failed += 1

    def _log_progress(done: int) -> None:
        rate = done / max(time.perf_counter() - t0, 1e-9)
        remaining = max(0, len(job_args) - done)
        eta_min = (remaining / rate / 60.0) if rate > 0 else 0.0
        logger.info(
            f"MCTS progress: {done}/{len(job_args)} "
            f"solved={n_solved} failed={n_failed} timeout={n_timeouts} unsound={n_unsound} "
            f"value_tuples={len(value_tuples)} rate={rate:.2f}/s eta={eta_min:.1f}min"
        )

    if max_workers <= 1:
        # Sequential fallback for tests / debugging
        for i, args in enumerate(job_args):
            _consume(_mcts_worker_fn(args))
            if (i + 1) % progress_every == 0:
                _log_progress(i + 1)
    else:
        # spawn context: avoid Linux fork() copy-on-write amplification of parent
        # state (trained nets + replay buffer + PyTorch state) across 32 workers,
        # which can cause silent OOM-kill cascades and broken-pool hangs.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            futures = [pool.submit(_mcts_worker_fn, args) for args in job_args]
            try:
                for i, fut in enumerate(as_completed(futures, timeout=collection_timeout_s)):
                    try:
                        result = fut.result()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"worker future raised: {type(e).__name__}: {e}")
                        n_failed += 1
                        result = None
                    if result is not None:
                        _consume(result)
                    if (i + 1) % progress_every == 0:
                        _log_progress(i + 1)
            except FuturesTimeout:
                n_pending = sum(1 for f in futures if not f.done())
                logger.error(
                    f"MCTS collection exceeded {collection_timeout_s}s; cancelling "
                    f"{n_pending} pending futures. solved={n_solved} timeouts={n_timeouts} "
                    f"unsound={n_unsound} failed={n_failed}"
                )
                for fut in futures:
                    if not fut.done():
                        fut.cancel()
            except BrokenProcessPool as e:
                logger.error(
                    f"ProcessPoolExecutor broken (likely OOM-killed workers): {e}. "
                    f"solved={n_solved} before crash."
                )
                raise

    # Final progress line so we always see a summary even if not on a 50-mark
    _log_progress(len(job_args))

    return value_tuples, policy_tuples, {
        "num_problems": len(problems),
        "num_solved": n_solved,
        "num_sound": n_solved,  # solved == sound here (workers verify before reporting found)
        "num_failed": n_failed,
        "num_timeouts": n_timeouts,
        "num_unsound": n_unsound,
        "total_simulations": total_sims,
        "num_value_tuples": len(value_tuples),
        "num_policy_tuples": len(policy_tuples),
        "elapsed_s": time.perf_counter() - t0,
        "max_workers": max_workers,
    }


# ---------------------- one ExIt iteration on a buffer ----------------------


def _attach_y_to_datas(value_tuples: list[ValueTuple]) -> list[Data]:
    out: list[Data] = []
    for t in value_tuples:
        d = Data(x=t.graph.x, edge_index=t.graph.edge_index)
        d.y = torch.tensor([t.log1p_steps], dtype=torch.float32)
        out.append(d)
    return out


def train_value_on_buffer(
    value_net: GINValueNet,
    buffer: ReplayBuffer,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> list[float]:
    if len(buffer) == 0:
        return []
    datas = _attach_y_to_datas(buffer.value_tuples())
    loader = DataLoader(datas, batch_size=batch_size, shuffle=True)
    opt = Adam(value_net.parameters(), lr=lr)
    losses: list[float] = []
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            loss = train_value_step(value_net, opt, batch)
            total += loss
            n += 1
        avg = total / max(n, 1)
        losses.append(avg)
        logger.info(f"value_train epoch {epoch + 1}/{epochs}: loss={avg:.4f}")
    return losses


def train_policy_on_buffer(
    policy_net: GINPolicyNet,
    buffer: ReplayBuffer,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> list[float]:
    pt = buffer.policy_tuples()
    if not pt:
        return []
    # Pre-compute torch tensors per record
    losses: list[float] = []
    opt = Adam(policy_net.parameters(), lr=lr)
    for epoch in range(epochs):
        order = list(range(len(pt)))
        random.shuffle(order)
        total = 0.0
        n = 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            if len(idx) < 2:  # BatchNorm needs >=2
                continue
            sub = [pt[i] for i in idx]
            batch = Batch.from_data_list([t.graph for t in sub]).to(device)
            target = torch.tensor(
                np.stack([t.target_distribution for t in sub], axis=0)
            ).to(device)
            mask = torch.tensor(
                np.stack([t.legal_mask for t in sub], axis=0)
            ).to(device)
            loss = train_policy_step(policy_net, opt, batch, target, mask, loss_type="kl")
            total += loss
            n += 1
        avg = total / max(n, 1)
        losses.append(avg)
        logger.info(f"policy_train epoch {epoch + 1}/{epochs}: loss={avg:.4f}")
    return losses


@dataclass
class IterationResult:
    iteration: int
    value_losses: list[float]
    policy_losses: list[float]
    mcts_stats: dict
    eval_stats: dict
    value_ckpt: Path
    policy_ckpt: Path
    buffer_size: int


def _save_inflight_checkpoints(
    value_net: GINValueNet,
    policy_net: GINPolicyNet,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist current model weights so MCTS workers (subprocesses) can load them.

    Overwritten each iteration; not the same as `save_checkpoints` which writes
    per-iteration archival snapshots.
    """
    inflight = output_dir / "inflight"
    inflight.mkdir(parents=True, exist_ok=True)
    value_path = inflight / "value.pt"
    policy_path = inflight / "policy.pt"
    torch.save({
        "model_state": value_net.state_dict(),
        "config": {
            "in_dim": value_net.in_dim,
            "hidden_dim": value_net.hidden_dim,
            "num_layers": value_net.num_layers,
            "dropout": value_net.dropout,
        },
        "target_transform": "log1p",
    }, value_path)
    torch.save({
        "model_state": policy_net.state_dict(),
        "config": {
            "in_dim": policy_net.in_dim,
            "hidden_dim": policy_net.hidden_dim,
            "num_layers": policy_net.num_layers,
            "dropout": policy_net.dropout,
            "out_dim": policy_net.out_dim,
        },
    }, policy_path)
    return value_path, policy_path


def run_one_iteration(
    *,
    iteration: int,
    problems: list[ExitProblem],
    eval_problem_ids: set[str],
    value_net: GINValueNet,
    policy_net: GINPolicyNet,
    buffer: ReplayBuffer,
    num_simulations: int,
    max_moves: int,
    c_puct: float,
    value_train_epochs: int,
    policy_train_epochs: int,
    value_lr: float,
    policy_lr: float,
    batch_size: int,
    output_dir: Path,
    device: str,
    eval_fn: Callable[[GINValueNet, GINPolicyNet], dict] | None = None,
    max_workers: int | None = None,
    timeout_per_problem: int = 120,
    progress_every: int = 50,
    collection_timeout_s: int | None = 3600,
) -> IterationResult:
    """One ExIt cycle: MCTS (parallel) -> append buffer -> train both nets -> eval -> checkpoint."""
    logger.info(f"=== iteration {iteration}: MCTS on {len(problems)} problems ===")
    gc.collect()
    value_path, policy_path = _save_inflight_checkpoints(value_net, policy_net, output_dir)
    value_tuples, policy_tuples, mcts_stats = collect_mcts_trajectories(
        problems,
        value_ckpt_path=value_path,
        policy_ckpt_path=policy_path,
        num_simulations=num_simulations,
        max_moves=max_moves,
        c_puct=c_puct,
        max_workers=max_workers,
        timeout_per_problem=timeout_per_problem,
        progress_every=progress_every,
        collection_timeout_s=collection_timeout_s,
    )
    logger.info(f"MCTS stats: {mcts_stats}")
    for v, p in zip(value_tuples, policy_tuples):
        buffer.add(v, p)
    buffer.sanity_check_no_leak(eval_problem_ids)
    logger.info(f"buffer size: {len(buffer)}")

    value_losses = train_value_on_buffer(
        value_net, buffer,
        epochs=value_train_epochs, batch_size=batch_size, lr=value_lr, device=device,
    )
    policy_losses = train_policy_on_buffer(
        policy_net, buffer,
        epochs=policy_train_epochs, batch_size=batch_size, lr=policy_lr, device=device,
    )

    eval_stats = eval_fn(value_net, policy_net) if eval_fn is not None else {}

    value_ckpt, policy_ckpt = save_checkpoints(
        value_net, policy_net, output_dir, iteration,
        extra={"mcts_stats": mcts_stats, "eval_stats": eval_stats},
    )
    return IterationResult(
        iteration=iteration,
        value_losses=value_losses,
        policy_losses=policy_losses,
        mcts_stats=mcts_stats,
        eval_stats=eval_stats,
        value_ckpt=value_ckpt,
        policy_ckpt=policy_ckpt,
        buffer_size=len(buffer),
    )


def generate_exit_problems(
    *,
    num_problems: int,
    base_seed: int,
    held_out_frac: float = 0.10,
) -> tuple[list[ExitProblem], list[ExitProblem]]:
    """Stratify-sample fresh problems via the Round 2 category generators.

    Returns (train_problems, held_out_problems). Uses different seeds from the
    warm-start extraction so train problems are fresh per iteration if desired
    (caller controls base_seed). For deterministic re-runs, use a fixed seed.
    """
    cats = list(CATEGORIES.keys())
    per_cat = max(1, num_problems // len(cats))
    held_out_per_cat = max(1, int(per_cat * held_out_frac))
    train: list[ExitProblem] = []
    held: list[ExitProblem] = []
    for cat in cats:
        for idx in range(per_cat):
            rng = random.Random(base_seed * 1_000_003 + hash(("exit", cat, idx)) % 1_000_003)
            depth = _DEPTH_CYCLE[idx % len(_DEPTH_CYCLE)]
            try:
                mi = CATEGORIES[cat](rng, depth)
            except Exception as e:
                logger.debug(f"exit gen_fail {cat}/{idx}: {e}")
                continue
            prob = ExitProblem(
                problem_id=f"exit_{cat}_{idx:05d}",
                initial=mi.eq_state, target=mi.target_eq_state, category=cat,
            )
            if idx < held_out_per_cat:
                held.append(prob)
            else:
                train.append(prob)
    return train, held


_DEPTH_CYCLE = (3, 5, 8, 10, 12, 15)


def save_checkpoints(
    value_net: GINValueNet,
    policy_net: GINPolicyNet,
    out_dir: Path,
    iteration: int,
    extra: dict | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    value_path = out_dir / f"value_iter_{iteration:02d}.pt"
    policy_path = out_dir / f"policy_iter_{iteration:02d}.pt"
    torch.save({
        "model_state": value_net.state_dict(),
        "config": {
            "in_dim": value_net.in_dim,
            "hidden_dim": value_net.hidden_dim,
            "num_layers": value_net.num_layers,
            "dropout": value_net.dropout,
        },
        "iteration": iteration,
        "target_transform": "log1p",
        **(extra or {}),
    }, value_path)
    torch.save({
        "model_state": policy_net.state_dict(),
        "config": {
            "in_dim": policy_net.in_dim,
            "hidden_dim": policy_net.hidden_dim,
            "num_layers": policy_net.num_layers,
            "dropout": policy_net.dropout,
            "out_dim": policy_net.out_dim,
        },
        "iteration": iteration,
        "rule_names": list(default_registry.names()),
        **(extra or {}),
    }, policy_path)
    return value_path, policy_path
