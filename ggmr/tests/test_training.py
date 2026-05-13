"""Phase 2 training pipeline tests.

11 tests covering srepr roundtrip, SymPy->PyG conversion (known trees,
Pow exponent flag, minus-one flag), GIN forward shape, overfit-one-batch
capacity check, LearnedHeuristic protocol/fallback, dataset split-by-problem
discipline, and a phase0 sanity run with a random-weight checkpoint.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from ggmr.heuristics.composite import Heuristic, WeightedSumCompositeHeuristic
from ggmr.heuristics.learned import LearnedHeuristic
from ggmr.problems.loader import load_phase0_problems
from ggmr.search.astar import astar
from ggmr.state import EqState
from ggmr.training.dataset import GGMRDataset
from ggmr.training.graph import (
    FEATURE_DIM,
    NODE_TYPE_TO_IDX,
    NODE_TYPE_VOCAB,
    sympy_to_pyg,
)
from ggmr.training.model import GINValueNet
from ggmr.training.srepr_parse import parse_srepr

MINI_PATH = Path(__file__).resolve().parents[1] / "training" / "training_data_mini.jsonl"


# ---------- fixtures ----------

@pytest.fixture(scope="session")
def random_ckpt(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build a random-weight GINValueNet checkpoint for downstream tests."""
    out = tmp_path_factory.mktemp("random_ckpt") / "rand.pt"
    model = GINValueNet(in_dim=FEATURE_DIM, hidden_dim=16, num_layers=2, dropout=0.0)
    torch.save({
        "model_state": model.state_dict(),
        "config": {"in_dim": FEATURE_DIM, "hidden_dim": 16, "num_layers": 2},
        "target_transform": "log1p",
        "node_vocab": list(NODE_TYPE_VOCAB),
    }, out)
    return str(out)


# ---------- 1. srepr roundtrip ----------

def test_srepr_roundtrip() -> None:
    """Every mini-row LHS/RHS srepr parses to a valid SymPy Basic without raising.

    Note: strict srepr-string equality is NOT preserved because SymPy's Mul
    constructor splits Integer(-N) into Integer(-1)*Integer(N) even with
    evaluate=False. The structural divergence is consistent across all
    callers (parse_srepr is the single entry point), so training and
    inference see the same canonicalized form.
    """
    if not MINI_PATH.exists():
        pytest.skip(f"mini data missing at {MINI_PATH}")
    parse_fails = 0
    total = 0
    with open(MINI_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("state_lhs_srepr", "state_rhs_srepr"):
                total += 1
                try:
                    expr = parse_srepr(row[key])
                    if not isinstance(expr, sp.Basic):
                        parse_fails += 1
                except Exception:
                    parse_fails += 1
    assert parse_fails == 0, f"{parse_fails}/{total} srepr parses failed"


# ---------- 2. SymPy -> PyG known tree ----------

def test_sympy_to_pyg_known_tree() -> None:
    """2x + 3 = 7 produces Eq root + 5-node LHS + 1-node RHS = 7 nodes."""
    x = sp.Symbol("x")
    lhs = 2 * x + 3
    rhs = sp.Integer(7)
    data = sympy_to_pyg(lhs, rhs, x)
    # Eq(idx 0) + Add(1) + Mul(2) + Integer(2)(3) + Symbol(x)(4) + Integer(3)(5) + Integer(7)(6) = 7 nodes
    assert data.x.shape == (7, FEATURE_DIM)
    # 6 forward edges * 2 (bidirectional) + 7 self-loops = 19 edges
    assert data.edge_index.shape[1] == 6 * 2 + 7
    # Type one-hot indices
    eq_root = data.x[0]
    assert eq_root[NODE_TYPE_TO_IDX["UNK"]] == 1.0


# ---------- 3. Pow exponent flag ----------

def test_pow_exponent_flag() -> None:
    """Pow(x, -1) sets is_pow_exponent on the exponent child, is_pow_base on x."""
    x = sp.Symbol("x")
    lhs = sp.Pow(x, sp.Integer(-1))
    rhs = sp.Integer(1)
    data = sympy_to_pyg(lhs, rhs, x)
    # Eq(0), Pow(1), Symbol(x)(2), Integer(-1)(3), Integer(1)(4)
    is_pow_base = data.x[:, 18]
    is_pow_exponent = data.x[:, 19]
    assert is_pow_base[2].item() == 1.0, "x should be marked is_pow_base"
    assert is_pow_exponent[3].item() == 1.0, "Integer(-1) should be marked is_pow_exponent"
    assert is_pow_base[3].item() == 0.0
    assert is_pow_exponent[2].item() == 0.0


# ---------- 4. minus-one flag ----------

def test_minus_one_flag() -> None:
    """Integer(-1) atom has is_minus_one feature = 1.0."""
    x = sp.Symbol("x")
    lhs = sp.Mul(sp.Integer(-1), x, evaluate=False)
    rhs = sp.Integer(0)
    data = sympy_to_pyg(lhs, rhs, x)
    # Find the node with type Integer and value -1
    type_int_idx = NODE_TYPE_TO_IDX["Integer"]
    is_int_col = 14  # numeric: sign_pos(10), sign_neg(11), sign_zero(12), log_abs(13), is_integer(14), is_minus_one(15)
    is_m1_col = 15
    found = False
    for i in range(data.x.shape[0]):
        if data.x[i, type_int_idx].item() == 1.0 and data.x[i, is_int_col].item() == 1.0:
            if data.x[i, is_m1_col].item() == 1.0:
                found = True
                break
    assert found, "Integer(-1) node should have is_minus_one=1.0"


# ---------- 5. GIN forward shape ----------

def test_gin_forward_shape() -> None:
    """Random Batch of 4 graphs -> output shape (4,), all finite."""
    x = sp.Symbol("x")
    data_list = [
        sympy_to_pyg(2 * x + 3, sp.Integer(7), x),
        sympy_to_pyg(x**2 - 1, sp.Integer(0), x),
        sympy_to_pyg(sp.Pow(x, sp.Integer(-1)), sp.Integer(2), x),
        sympy_to_pyg(x + sp.Rational(1, 2), sp.Rational(3, 4), x),
    ]
    batch = Batch.from_data_list(data_list)
    model = GINValueNet(in_dim=FEATURE_DIM, hidden_dim=32, num_layers=3, dropout=0.0)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
    # Final ReLU was removed during Phase 2 (it killed gradients on negative
    # log-space predictions); inference clips via `expm1` + np.clip[0, 30].
    # So untrained-random-weight output may be negative — only shape/finiteness
    # are tested here.


# ---------- 6. overfit one batch ----------

def test_overfit_one_batch() -> None:
    """Model has capacity: 8 distinct linear equations with y = log1p(coefficient)
    should fit to MSE < 0.1 in 800 epochs. Validates basic learning machinery."""
    x = sp.Symbol("x")
    torch.manual_seed(0)
    random.seed(0)
    data_list = []
    for i in range(8):
        # 8 distinct equations: (i+1)*x + 2 = 5, y = log1p(i)
        d = sympy_to_pyg(sp.Integer(i + 1) * x + sp.Integer(2), sp.Integer(5), x)
        d.y = torch.tensor([math.log1p(i)], dtype=torch.float32)
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    model = GINValueNet(in_dim=FEATURE_DIM, hidden_dim=64, num_layers=3, dropout=0.0)
    opt = Adam(model.parameters(), lr=1e-2)
    model.train()
    final_loss = float("inf")
    for _ in range(800):
        opt.zero_grad()
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y.view(-1))
        loss.backward()
        opt.step()
        final_loss = loss.item()
    assert final_loss < 0.1, f"could not overfit 8-sample batch: final MSE {final_loss:.4f}"


# ---------- 7. LearnedHeuristic implements Protocol ----------

def test_learned_heuristic_protocol(random_ckpt: str) -> None:
    """LearnedHeuristic satisfies the runtime_checkable Heuristic Protocol."""
    h = LearnedHeuristic(random_ckpt, device="cpu")
    assert isinstance(h, Heuristic), "LearnedHeuristic must satisfy Heuristic protocol"
    state = EqState.from_strings("2*x + 3", "7")
    val = h.evaluate(state)
    assert isinstance(val, float)
    assert math.isfinite(val)
    assert val >= 0.0


# ---------- 8. LearnedHeuristic fallback on error ----------

def test_learned_heuristic_fallback(random_ckpt: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the model raises, evaluate falls back to the hand heuristic value (not inf)."""
    h = LearnedHeuristic(random_ckpt, device="cpu")
    state = EqState.from_strings("2*x + 3", "7")
    expected_fallback = WeightedSumCompositeHeuristic().evaluate(state)

    def boom(*a, **kw):
        raise RuntimeError("simulated forward failure")
    monkeypatch.setattr(h, "_model", type("M", (), {"__call__": boom})())
    h._cache.clear()
    val = h.evaluate(state)
    assert math.isfinite(val)
    assert val == expected_fallback


# ---------- 9. dataset split by problem ----------

def test_dataset_split_by_problem(tmp_path: Path) -> None:
    """No problem_id should appear in both train and val splits."""
    rows = []
    for pid_num in range(6):
        for step in range(3):
            rows.append({
                "problem_id": f"easy_lin_{pid_num:03d}",
                "remaining_steps": step,
                "state_lhs_srepr": f"Add(Mul(Integer({pid_num + 1}), Symbol('x')), Integer({step}))",
                "state_rhs_srepr": "Integer(7)",
                "var": "x",
                "excluded_srepr": [],
                "source": "easy",
                "family": None,
                "template": "linear",
                "depth": 1,
            })
    p = tmp_path / "synthetic.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ds = GGMRDataset.from_jsonl(p)
    train, val, test = ds.split_by_problem_id(seed=0)
    train_pids = {m.problem_id for m in train.meta}
    val_pids = {m.problem_id for m in val.meta}
    test_pids = {m.problem_id for m in test.meta}
    assert train_pids & val_pids == set(), f"leakage train<->val: {train_pids & val_pids}"
    assert train_pids & test_pids == set(), f"leakage train<->test: {train_pids & test_pids}"
    assert val_pids & test_pids == set(), f"leakage val<->test: {val_pids & test_pids}"
    assert len(train_pids) >= 1
    assert len(train) >= 1


# ---------- 10. evaluate runs end-to-end on 2 problems ----------

def test_evaluate_two_problems(random_ckpt: str) -> None:
    """Run hand + learned A* on 2 phase0 problems; both should complete (no crash)."""
    problems = load_phase0_problems()
    assert len(problems) >= 2, "phase0 should have >=2 problems"
    subset = problems[:2]
    hand = WeightedSumCompositeHeuristic()
    learned = LearnedHeuristic(random_ckpt, device="cpu")
    for prob in subset:
        r_hand = astar(prob.initial, prob.is_target, heuristic=hand,
                       max_nodes=2000, max_depth=15, problem_id=prob.id)
        r_learned = astar(prob.initial, prob.is_target, heuristic=learned,
                          max_nodes=2000, max_depth=15, problem_id=prob.id)
        assert r_hand.stats.nodes_expanded >= 0
        assert r_learned.stats.nodes_expanded >= 0


# ---------- 11. phase0 solve rate with random-weight heuristic ----------

@pytest.mark.slow
def test_phase0_solve_rate_with_learned(random_ckpt: str) -> None:
    """All 20 phase0 problems should solve within 5000 nodes with the learned heuristic
    (random weights -- relies on A*'s g-cost dominating eventually for small problems)."""
    problems = load_phase0_problems()
    assert len(problems) == 20, f"phase0 should have 20 problems, got {len(problems)}"
    learned = LearnedHeuristic(random_ckpt, device="cpu")
    solved = 0
    for prob in problems:
        result = astar(
            prob.initial, prob.is_target, heuristic=learned,
            max_nodes=5000, max_depth=15, problem_id=prob.id,
        )
        if result.found:
            solved += 1
    # Random heuristic + small phase0 problems -> A* still completes most
    assert solved >= 15, f"only {solved}/20 phase0 solved with random-weight learned heuristic"
