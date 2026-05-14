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

import json
import logging
import math
import random
from collections import defaultdict, deque
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


def collect_mcts_trajectories(
    problems: list[ExitProblem],
    *,
    value_advisor: ValueAdvisor,
    policy_advisor: PolicyAdvisor,
    num_simulations: int,
    max_moves: int,
    c_puct: float = 1.5,
) -> tuple[list[ValueTuple], list[PolicyTuple], dict]:
    """Run MCTS on each problem. Return value tuples, policy tuples, and stats."""
    value_tuples: list[ValueTuple] = []
    policy_tuples: list[PolicyTuple] = []
    rn_to_idx = _rule_name_to_idx()
    n_rules = num_rules()
    n_solved = 0
    n_sound = 0
    total_sims = 0

    for prob in problems:
        try:
            result = mcts_search(
                prob.initial,
                prob.is_target,
                value_fn=value_advisor.value_fn,
                policy_fn=policy_advisor.policy_fn,
                num_simulations=num_simulations,
                max_moves=max_moves,
                c_puct=c_puct,
            )
        except Exception as e:
            logger.warning(f"MCTS error on {prob.problem_id}: {e}")
            continue
        total_sims += result.stats.total_simulations
        if not result.found:
            continue
        # Verify trajectory soundness once before extracting tuples
        if not _verify_trajectory_soundness(prob.initial, result.path, prob.is_target):
            logger.warning(f"unsound trajectory for {prob.problem_id}; skipping")
            continue
        n_solved += 1
        n_sound += 1
        # Extract per-step value+policy tuples. visit_distributions[i] is the
        # root visit distribution at move i (state = path[i][0]).
        n_steps = len(result.path)
        for i, ((state, _action), dist) in enumerate(zip(result.path, result.visit_distributions)):
            try:
                graph = sympy_to_pyg(state.lhs, state.rhs, state.var)
            except Exception:
                continue
            remaining = n_steps - i  # MCTS-discovered distance from this state
            data = Data(x=graph.x, edge_index=graph.edge_index)
            data.y = torch.tensor([math.log1p(remaining)], dtype=torch.float32)
            value_tuples.append(ValueTuple(
                graph=data, log1p_steps=math.log1p(remaining), problem_id=prob.problem_id,
            ))
            # Build full-length distribution + legal mask
            mask = np.zeros(n_rules, dtype=np.float32)
            target = np.zeros(n_rules, dtype=np.float32)
            for rule, action in default_registry.enumerate_actions(state):
                if rule.guard(state, action).ok:
                    mask[rn_to_idx[rule.name]] = 1.0
            for rn, p in dist.items():
                if rn in rn_to_idx:
                    target[rn_to_idx[rn]] = float(p)
            s = target.sum()
            if s > 0:
                target /= s
            else:
                # Fallback: uniform over legal
                if mask.sum() > 0:
                    target = mask / mask.sum()
            policy_graph = Data(x=graph.x, edge_index=graph.edge_index)
            policy_tuples.append(PolicyTuple(
                graph=policy_graph, target_distribution=target, legal_mask=mask,
            ))

    return value_tuples, policy_tuples, {
        "num_problems": len(problems),
        "num_solved": n_solved,
        "num_sound": n_sound,
        "total_simulations": total_sims,
        "num_value_tuples": len(value_tuples),
        "num_policy_tuples": len(policy_tuples),
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


def run_one_iteration(
    *,
    iteration: int,
    problems: list[ExitProblem],
    eval_problem_ids: set[str],
    value_net: GINValueNet,
    policy_net: GINPolicyNet,
    value_advisor: ValueAdvisor,
    policy_advisor: PolicyAdvisor,
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
) -> IterationResult:
    """One ExIt cycle: MCTS -> append buffer -> train both nets -> eval -> checkpoint."""
    logger.info(f"=== iteration {iteration}: MCTS on {len(problems)} problems ===")
    value_tuples, policy_tuples, mcts_stats = collect_mcts_trajectories(
        problems,
        value_advisor=value_advisor,
        policy_advisor=policy_advisor,
        num_simulations=num_simulations,
        max_moves=max_moves,
        c_puct=c_puct,
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
