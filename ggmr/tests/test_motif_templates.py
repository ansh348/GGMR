"""Tests for ggmr/problems/motif_templates.py.

Fast tests (default): algebraic correctness, op-count budget, YAML roundtrip,
template-specific rejection of degenerate parameter choices.

Slow tests (`pytest -m slow`): actually run BFS at 50k nodes on the default
parameters of each motif family — verifies the constructed problems are BFS-
solvable. Excluded from default `pytest` via pyproject.toml addopts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sympy as sp

from ggmr.expr.tree import canonical_repr, op_count
from ggmr.problems.hard_yaml_emit import (
    emit_hard_problems_yaml,
    load_hard_problems_yaml,
)
from ggmr.problems.motif_templates import (
    MotifInstance,
    OP_COUNT_BUDGET,
    motif_l1,
    motif_l3,
    motif_p3,
    motif_p4,
    motif_r1,
    motif_r2,
    motif_v1_ex1,
    verify_instance,
)

# Bring PARAM_SWEEPS / TEMPLATE_FNS in from the generator script
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from generate_hard_eval_set_v2 import PARAM_SWEEPS, TEMPLATE_FNS  # noqa: E402


FAMILIES = list(TEMPLATE_FNS.keys())


# ---------------------------------------------------------------------------
# Default-build / verification / op-count / solution-set tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
def test_default_builds(family, x):
    """Validated default parameters of every family produce a MotifInstance."""
    inst = TEMPLATE_FNS[family](var=x, **PARAM_SWEEPS[family][0])
    assert isinstance(inst, MotifInstance)
    assert inst.motif_family == family


@pytest.mark.parametrize("family", FAMILIES)
def test_all_sweep_variations_verify(family, x):
    """At least 8 of each family's sweep dicts must build and pass verify_instance."""
    fn = TEMPLATE_FNS[family]
    valid = 0
    failures: list[tuple[int, str]] = []
    for i, params in enumerate(PARAM_SWEEPS[family]):
        try:
            inst = fn(var=x, **params)
        except (ValueError, AssertionError) as e:
            failures.append((i, f"build: {e}"))
            continue
        ok, reason = verify_instance(inst)
        if not ok:
            failures.append((i, f"verify: {reason}"))
            continue
        valid += 1
    assert valid >= 8, (
        f"{family}: only {valid}/{len(PARAM_SWEEPS[family])} variations valid.\n"
        f"Failures: {failures}"
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_op_count_budget(family, x):
    """No variation exceeds the per-problem op-count budget."""
    fn = TEMPLATE_FNS[family]
    for params in PARAM_SWEEPS[family]:
        try:
            inst = fn(var=x, **params)
        except (ValueError, AssertionError):
            continue
        total = op_count(inst.eq_state.lhs) + op_count(inst.eq_state.rhs)
        assert total <= OP_COUNT_BUDGET, f"{family} {params}: ops={total}"


@pytest.mark.parametrize("family", FAMILIES)
def test_target_solution_matches(family, x):
    """Initial state's solution set equals target's solution set for every valid variation."""
    fn = TEMPLATE_FNS[family]
    for params in PARAM_SWEEPS[family]:
        try:
            inst = fn(var=x, **params)
        except (ValueError, AssertionError):
            continue
        actual = inst.eq_state.solution_set()
        expected = inst.target_eq_state.solution_set()
        assert actual == expected, f"{family} {params}: actual={actual} expected={expected}"


@pytest.mark.parametrize("family", FAMILIES)
def test_yaml_roundtrip(family, x, tmp_path):
    """Build → emit_hard_problems_yaml → load_hard_problems_yaml preserves the
    correctness invariants (solution set, excluded, variable) and the
    factored-twin disguise on the LHS.

    Note: `sp.sympify(srepr)` re-evaluates simpler subtrees (the RHS expanded
    twin form, for example, collapses to its canonical polynomial). The
    asymmetry between LHS factored and RHS evaluated is what BFS+A* sees,
    and is what makes validation A* numbers reproducible across runs.
    """
    fn = TEMPLATE_FNS[family]
    inst = fn(var=x, **PARAM_SWEEPS[family][0])
    record = inst.to_record(f"hard_motif_{family}_test")
    out_path = tmp_path / f"{family}.yaml"
    emit_hard_problems_yaml([record], str(out_path))
    loaded = load_hard_problems_yaml(str(out_path))
    assert len(loaded) == 1
    rt = loaded[0]
    assert rt.initial.solution_set() == inst.eq_state.solution_set()
    assert rt.initial.excluded == inst.eq_state.excluded
    assert rt.initial.var.name == inst.eq_state.var.name
    # LHS factored-twin structure should survive (we check it's still complex,
    # i.e. roughly comparable op_count, not fully collapsed to canonical poly).
    orig_lhs_ops = op_count(inst.eq_state.lhs)
    rt_lhs_ops = op_count(rt.initial.lhs)
    assert rt_lhs_ops >= orig_lhs_ops // 2, (
        f"{family}: LHS lost factored structure on roundtrip "
        f"(orig ops={orig_lhs_ops}, rt ops={rt_lhs_ops})"
    )


# ---------------------------------------------------------------------------
# Template-specific rejection of degenerate parameters
# ---------------------------------------------------------------------------

def test_p3_rejects_reducible_quadratic(x):
    with pytest.raises(ValueError, match="not irreducible"):
        motif_p3(var=x, roots=(1, 2, 3), irreducible_p=4, irreducible_q=3,
                 linear_decoy_pair=(5, -1))


def test_p3_rejects_duplicate_roots(x):
    with pytest.raises(ValueError, match="distinct"):
        motif_p3(var=x, roots=(1, 1, 3), irreducible_p=-2, irreducible_q=7,
                 linear_decoy_pair=(5, -1))


def test_r1_rejects_integer_target(x):
    """(rhs_const - lhs_const) divisible by (lhs_linear - rhs_linear) → integer target."""
    with pytest.raises(ValueError, match="integer"):
        motif_r1(var=x, lhs_linear=8, rhs_linear=5, lhs_const=3, rhs_const=3,
                 twin1=(1, -2), twin2=(3, -4))


def test_r2_rejects_integer_target(x):
    with pytest.raises(ValueError, match="integer"):
        motif_r2(var=x, scalar=2, inner_coef=3, inner_const=1, rhs_const=14,
                 inner_twin=(1, -2), outer_twin=(3, -4))


def test_p4_rejects_denominator_overlap(x):
    with pytest.raises(ValueError, match="coincides"):
        motif_p4(var=x, target_roots=(5, 2, -3), denom1=5, denom2=4, scalar=5,
                 twin=(7, -1), free_d=3, free_e=0)


def test_p4_rejects_equal_denoms(x):
    with pytest.raises(ValueError, match="distinct"):
        motif_p4(var=x, target_roots=(5, 2, -3), denom1=-1, denom2=-1, scalar=5,
                 twin=(7, -1), free_d=3, free_e=0)


def test_v1_ex1_rejects_target_equals_rational_root(x):
    """target_val == rational_root would make excluded value satisfy target."""
    with pytest.raises(ValueError, match="equals target"):
        motif_v1_ex1(var=x, linear_coef=2, lhs_const=3, rhs_const=7,
                     twin_a=1, twin_b=-4, rational_root=2)


def test_polynomial_twin_rejects_zero_factor(x):
    with pytest.raises(ValueError, match="zero factor"):
        motif_l1(var=x, linear_coef=2, lhs_const=0, rhs_const=4,
                 twin1=(0, -3), twin2=(1, -2))


def test_polynomial_twin_rejects_square(x):
    with pytest.raises(ValueError, match="degenerate square"):
        motif_l1(var=x, linear_coef=2, lhs_const=0, rhs_const=4,
                 twin1=(2, 2), twin2=(1, -3))


# ---------------------------------------------------------------------------
# Slow tests: actually run BFS to confirm BFS-solvability of default params
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_default_bfs_solves_50k(family, x):
    from ggmr.search.bfs import bfs
    inst = TEMPLATE_FNS[family](var=x, **PARAM_SWEEPS[family][0])
    target = inst.target_eq_state
    target_lhs_key = canonical_repr(target.lhs)
    target_rhs_key = canonical_repr(target.rhs)
    target_sol = target.solution_set()

    def is_target(s):
        if canonical_repr(s.lhs) == target_lhs_key and canonical_repr(s.rhs) == target_rhs_key:
            return True
        return s.is_canonical_target() and s.solution_set() == target_sol

    result = bfs(inst.eq_state, is_target, max_nodes=50_000, max_depth=50)
    assert result.found, (
        f"{family} default did not solve in 50k BFS nodes "
        f"(expanded={result.stats.nodes_expanded})"
    )
