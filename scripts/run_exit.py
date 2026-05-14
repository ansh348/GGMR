"""CLI: run Phase 3 ExIt training (warm-start + N iterations + eval).

Default flow:
  1. If warmstart JSONL missing, generate it (via extract_policy_warmstart_data)
  2. If policy checkpoint missing, pre-train on warm-start data
  3. Load value + policy nets; build replay buffer (seeded with warm-start tuples)
  4. For each iteration: generate fresh problems -> MCTS -> append buffer ->
     train both nets -> save checkpoint -> simple held-out eval
  5. Dump per-iteration results JSON.

Final A* + MCTS comparison eval is in `ggmr.training.evaluate` (step 8); this
script saves the checkpoints that eval consumes.

    python scripts/run_exit.py \
        --value-ckpt checkpoints/round2/best.pt \
        --output-dir checkpoints/exit_v1 \
        --iterations 3 \
        --simulations 400 \
        --num-problems 1000 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import sympy as sp
import torch

import ggmr.rules.core  # noqa: F401  (register rules)
from ggmr.training.exit_loop import (
    ExitProblem,
    IterationResult,
    ReplayBuffer,
    collect_mcts_trajectories,
    generate_exit_problems,
    load_warmstart_jsonl,
    pre_train_policy_on_bfs,
    run_one_iteration,
)
from ggmr.training.model import GINValueNet
from ggmr.training.policy_heuristic import PolicyAdvisor, ValueAdvisor
from ggmr.training.policy_model import GINPolicyNet, num_rules
from ggmr.training.srepr_parse import parse_srepr
from ggmr.training.graph import sympy_to_pyg
from ggmr.training.exit_loop import ValueTuple, PolicyTuple
from ggmr.search.mcts import mcts_search
from ggmr.training.extract_pairs import _build_is_target
from ggmr.problems.loader import load_hard_evaluation_set, load_phase0_problems
from ggmr.state import EqState

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_value_net(ckpt_path: Path, device: str) -> GINValueNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    net = GINValueNet(
        in_dim=cfg.get("in_dim", 24),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 5),
        dropout=cfg.get("dropout", 0.1),
    )
    net.load_state_dict(ckpt["model_state"])
    return net.to(device)


def _load_policy_net(ckpt_path: Path, device: str) -> GINPolicyNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    net = GINPolicyNet(
        in_dim=cfg.get("in_dim", 24),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 5),
        dropout=cfg.get("dropout", 0.1),
        out_dim=cfg.get("out_dim", num_rules()),
    )
    net.load_state_dict(ckpt["model_state"])
    return net.to(device)


def _seed_buffer_from_warmstart(
    warmstart_path: Path,
    buffer: ReplayBuffer,
) -> int:
    """Seed the replay buffer with warm-start tuples (one-hot policy, log1p(1) value).

    Value target is log1p(1) ≈ 0.693 (one BFS step away from target). This is
    a placeholder — the warm-start records don't carry the full remaining_steps
    along the BFS path, only the next-action. For policy training this is fine;
    for value training we'd prefer richer signal. Iteration 1+ uses MCTS-derived
    remaining_steps which are accurate.
    """
    import math
    from ggmr.rules.registry import default_registry
    from torch_geometric.data import Data

    rn_to_idx = {n: i for i, n in enumerate(default_registry.names())}
    n_rules = num_rules()
    count = 0
    train_recs, _ = load_warmstart_jsonl(warmstart_path)
    for r in train_recs:
        try:
            lhs = parse_srepr(r["state_lhs_srepr"])
            rhs = parse_srepr(r["state_rhs_srepr"])
            var = sp.Symbol(r.get("var", "x"))
            graph = sympy_to_pyg(lhs, rhs, var)
        except Exception:
            continue
        rule_name = r.get("rule_name")
        if rule_name not in rn_to_idx:
            continue
        state = EqState(lhs=lhs, rhs=rhs, var=var)
        legal_mask = np.zeros(n_rules, dtype=np.float32)
        any_legal = False
        for rule, action in default_registry.enumerate_actions(state):
            if rule.guard(state, action).ok:
                legal_mask[rn_to_idx[rule.name]] = 1.0
                any_legal = True
        if not any_legal:
            continue
        # Approximate remaining_steps via path_length - step_index (when present)
        path_len = r.get("path_length", 0)
        step_idx = r.get("step_index", 0)
        remaining = max(path_len - step_idx, 1)
        v_data = Data(x=graph.x, edge_index=graph.edge_index)
        v_data.y = torch.tensor([math.log1p(remaining)], dtype=torch.float32)
        target = np.zeros(n_rules, dtype=np.float32)
        target[rn_to_idx[rule_name]] = 1.0
        v_tuple = ValueTuple(
            graph=v_data, log1p_steps=math.log1p(remaining),
            problem_id=r.get("problem_id", "warmstart"),
        )
        p_graph = Data(x=graph.x, edge_index=graph.edge_index)
        p_tuple = PolicyTuple(graph=p_graph, target_distribution=target, legal_mask=legal_mask)
        buffer.add(v_tuple, p_tuple)
        count += 1
    return count


def _quick_eval_held_out(
    value_net: GINValueNet,
    policy_net: GINPolicyNet,
    held_out: list[ExitProblem],
    *,
    num_simulations: int,
    max_moves: int,
    device: str,
) -> dict:
    """Run MCTS on held-out problems with the current networks. Report solve count
    and median simulations-to-solve. Used as a quick per-iteration health check."""
    value_net.eval()
    policy_net.eval()
    # Create advisors that wrap the live nets (not from disk)
    va = _live_value_advisor(value_net, device)
    pa = _live_policy_advisor(policy_net, device)
    solved = 0
    sims_to_solve: list[int] = []
    for p in held_out:
        try:
            result = mcts_search(
                p.initial, p.is_target,
                value_fn=va.value_fn, policy_fn=pa.policy_fn,
                num_simulations=num_simulations, max_moves=max_moves,
            )
        except Exception:
            continue
        if result.found:
            solved += 1
            sims_to_solve.append(result.stats.total_simulations)
    return {
        "held_out_size": len(held_out),
        "solved": solved,
        "solve_rate": solved / max(len(held_out), 1),
        "median_sims": float(np.median(sims_to_solve)) if sims_to_solve else None,
        "mean_sims": float(np.mean(sims_to_solve)) if sims_to_solve else None,
    }


def _live_value_advisor(value_net: GINValueNet, device: str) -> ValueAdvisor:
    """Build a ValueAdvisor that wraps an in-memory value net (no file IO)."""
    va = ValueAdvisor.__new__(ValueAdvisor)
    va.device = device
    va.cache_size = 50_000
    va._model = value_net.eval()
    va._target_transform = "log1p"
    va._cache = __import__("collections").OrderedDict()
    return va


def _live_policy_advisor(policy_net: GINPolicyNet, device: str) -> PolicyAdvisor:
    pa = PolicyAdvisor.__new__(PolicyAdvisor)
    pa.device = device
    pa.cache_size = 50_000
    from ggmr.rules.registry import default_registry
    pa._rule_names = list(default_registry.names())
    pa._name_to_idx = {n: i for i, n in enumerate(pa._rule_names)}
    pa._model = policy_net.eval()
    pa._cache = __import__("collections").OrderedDict()
    return pa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-ckpt", type=Path, required=True,
                        help="initial value net checkpoint (e.g., checkpoints/round2/best.pt)")
    parser.add_argument("--policy-ckpt-init", type=Path, default=None,
                        help="pre-trained policy checkpoint; if missing, pre-train from warmstart")
    parser.add_argument("--warmstart-data", type=Path, default=Path("policy_warmstart.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--num-problems", type=int, default=1000)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--value-epochs", type=int, default=10)
    parser.add_argument("--policy-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--policy-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--held-out-eval-sims", type=int, default=200)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)

    # Step 1: ensure warmstart data exists
    if not args.warmstart_data.exists():
        logger.error(
            f"warmstart data not found at {args.warmstart_data}. "
            f"Generate with: python scripts/extract_policy_warmstart_data.py "
            f"--num-problems 900 --output {args.warmstart_data}"
        )
        return 1

    # Step 2: ensure policy ckpt exists (pre-train if not)
    if args.policy_ckpt_init is None or not args.policy_ckpt_init.exists():
        pre_ckpt = args.output_dir / "policy_warmstart.pt"
        if not pre_ckpt.exists():
            logger.info(f"pre-training policy on {args.warmstart_data} -> {pre_ckpt}")
            pre_stats = pre_train_policy_on_bfs(
                warmstart_path=args.warmstart_data,
                output_ckpt=pre_ckpt,
                device=args.device,
                epochs=10,
                batch_size=args.batch_size,
                lr=1e-3,
                seed=args.seed,
            )
            with (args.output_dir / "pre_train_stats.json").open("w") as f:
                json.dump(pre_stats, f, indent=2)
            if pre_stats["final_held_top1"] < 0.5:
                logger.warning(
                    f"pre-train held-out top1 = {pre_stats['final_held_top1']:.3f} < 0.5; "
                    f"continuing but criterion not met"
                )
        else:
            logger.info(f"reusing existing warm-start policy ckpt at {pre_ckpt}")
        args.policy_ckpt_init = pre_ckpt

    # Step 3: load nets
    logger.info(f"loading value net from {args.value_ckpt}")
    value_net = _load_value_net(args.value_ckpt, args.device)
    logger.info(f"loading policy net from {args.policy_ckpt_init}")
    policy_net = _load_policy_net(args.policy_ckpt_init, args.device)

    # Step 4: collect eval problem IDs for no-leak check
    hard = load_hard_evaluation_set()
    phase0 = load_phase0_problems()
    eval_pids: set[str] = {p.id for p in hard} | {p.id for p in phase0}
    logger.info(f"eval pids (hard+phase0): {len(eval_pids)}")

    # Step 5: build replay buffer; seed with warm-start tuples
    buffer = ReplayBuffer(max_size=args.buffer_size, held_out_pids=set())
    n_seeded = _seed_buffer_from_warmstart(args.warmstart_data, buffer)
    logger.info(f"seeded buffer with {n_seeded} warm-start tuples")

    # Step 6: run iterations
    all_results: list[dict] = []
    t_start = time.perf_counter()
    for it in range(args.iterations):
        logger.info(f"\n===== ExIt iteration {it + 1}/{args.iterations} =====")
        train_problems, held_out = generate_exit_problems(
            num_problems=args.num_problems,
            base_seed=args.seed + it * 1000,
        )
        # Filter out any problem_id that could overlap with eval set (defensive)
        train_problems = [p for p in train_problems if p.problem_id not in eval_pids]
        held_pids = {p.problem_id for p in held_out}
        logger.info(f"problems: train={len(train_problems)} held_out={len(held_out)}")

        # Live advisors wrap current nets (no disk reload)
        va = _live_value_advisor(value_net, args.device)
        pa = _live_policy_advisor(policy_net, args.device)

        def eval_fn(vn, pn):
            return _quick_eval_held_out(
                vn, pn, held_out,
                num_simulations=args.held_out_eval_sims,
                max_moves=args.max_moves,
                device=args.device,
            )

        # Buffer's held-out set: union of MCTS held-out + hard/phase0 IDs
        buffer._held_out_pids = eval_pids | held_pids

        result = run_one_iteration(
            iteration=it,
            problems=train_problems,
            eval_problem_ids=eval_pids | held_pids,
            value_net=value_net, policy_net=policy_net,
            value_advisor=va, policy_advisor=pa,
            buffer=buffer,
            num_simulations=args.simulations,
            max_moves=args.max_moves,
            c_puct=args.c_puct,
            value_train_epochs=args.value_epochs,
            policy_train_epochs=args.policy_epochs,
            value_lr=args.value_lr,
            policy_lr=args.policy_lr,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            device=args.device,
            eval_fn=eval_fn,
        )
        all_results.append({
            "iteration": result.iteration,
            "buffer_size": result.buffer_size,
            "value_losses": result.value_losses,
            "policy_losses": result.policy_losses,
            "mcts_stats": result.mcts_stats,
            "eval_stats": result.eval_stats,
            "value_ckpt": str(result.value_ckpt),
            "policy_ckpt": str(result.policy_ckpt),
        })
        with (args.output_dir / "results.json").open("w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"iteration {it} eval: {result.eval_stats}")

    elapsed = time.perf_counter() - t_start
    logger.info(f"all {args.iterations} iterations done in {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
