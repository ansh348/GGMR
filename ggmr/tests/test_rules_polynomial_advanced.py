"""Tests for cubic-identity factoring rules and the nth-root rule.

Covers:
- FACTOR_DIFFERENCE_OF_CUBES
- FACTOR_SUM_OF_CUBES
- NTH_ROOT_BOTH_SIDES

Plus 2 end-to-end A* integration tests using `WeightedSumCompositeHeuristic`.
"""

from __future__ import annotations

import sympy as sp
from sympy import Integer

from ggmr.rules.core import *  # noqa: F401,F403  (registers all rules)
from ggmr.rules.core.arithmetic import NthRootBothSides, SplitAbsoluteValue
from ggmr.rules.core.polynomial_advanced import (
    FactorDifferenceOfCubes,
    FactorSumOfCubes,
)
from ggmr.rules.base import Action
from ggmr.soundness import VERIFY_PASS, verify_transition
from ggmr.state import EqState


def _solset(s: EqState) -> frozenset:
    return s.solution_set()


def _verify(parent: EqState, child: EqState) -> str:
    verdict, _ = verify_transition(
        parent.lhs,
        parent.rhs,
        child.lhs,
        child.rhs,
        parent.var,
        parent_excluded=parent.excluded,
        child_excluded=child.excluded,
    )
    return verdict


# --- FACTOR_DIFFERENCE_OF_CUBES -------------------------------------------


def test_diff_cubes_x3_minus_8_lhs():
    s = EqState.from_strings("x**3 - 8", "0")
    rule = FactorDifferenceOfCubes()
    actions = list(rule.enumerate(s))
    assert len(actions) >= 1, "Expected at least one enumerated action on x^3 - 8 = 0"
    child = rule.apply(s, actions[0])
    assert _verify(s, child) == VERIFY_PASS


def test_diff_cubes_x3_minus_27_rhs_side():
    """walk_with_side must cover BOTH lhs and rhs subtrees."""
    s = EqState.from_strings("0", "x**3 - 27")
    rule = FactorDifferenceOfCubes()
    actions = list(rule.enumerate(s))
    assert len(actions) >= 1
    # At least one of those actions should target the rhs (where the diff-of-cubes lives)
    assert any(a.target_side == "rhs" for a in actions)
    rhs_action = next(a for a in actions if a.target_side == "rhs")
    child = rule.apply(s, rhs_action)
    assert _verify(s, child) == VERIFY_PASS


def test_diff_cubes_nonsymbol_base():
    """`(2*x)**3 - 1` should fire with a = 2*x, b = 1."""
    s = EqState.from_strings("(2*x)**3 - 1", "0")
    rule = FactorDifferenceOfCubes()
    actions = list(rule.enumerate(s))
    assert len(actions) >= 1


def test_diff_cubes_skips_non_cube():
    """7 is not a perfect cube; rule should NOT fire."""
    s = EqState.from_strings("x**3 - 7", "0")
    rule = FactorDifferenceOfCubes()
    # Must not yield any action where the top-level sub matches diff-of-cubes
    # The walk visits all subtrees; we check the LHS root subtree specifically.
    actions = list(rule.enumerate(s))
    root_actions = [a for a in actions if a.target_path == () and a.target_side == "lhs"]
    assert root_actions == [], "x^3 - 7 should not be detected as diff-of-cubes at root"


# --- FACTOR_SUM_OF_CUBES --------------------------------------------------


def test_sum_cubes_x3_plus_8():
    s = EqState.from_strings("x**3 + 8", "0")
    rule = FactorSumOfCubes()
    actions = list(rule.enumerate(s))
    assert len(actions) >= 1
    child = rule.apply(s, actions[0])
    assert _verify(s, child) == VERIFY_PASS


def test_sum_cubes_x3_plus_27():
    s = EqState.from_strings("x**3 + 27", "0")
    rule = FactorSumOfCubes()
    actions = list(rule.enumerate(s))
    assert len(actions) >= 1
    child = rule.apply(s, actions[0])
    assert _verify(s, child) == VERIFY_PASS


def test_sum_cubes_skips_diff_form():
    """`x^3 - 8` is a difference-of-cubes, NOT sum-of-cubes; sum rule must skip it at root."""
    s = EqState.from_strings("x**3 - 8", "0")
    rule = FactorSumOfCubes()
    actions = list(rule.enumerate(s))
    root_actions = [a for a in actions if a.target_path == () and a.target_side == "lhs"]
    assert root_actions == []


# --- NTH_ROOT_BOTH_SIDES --------------------------------------------------


def test_nth_root_x3_eq_8():
    s = EqState.from_strings("x**3", "8")
    rule = NthRootBothSides()
    actions = list(rule.enumerate(s))
    assert len(actions) == 1
    assert actions[0].params == (Integer(3),)
    assert actions[0].target_side == "lhs"
    child = rule.apply(s, actions[0])
    assert child.lhs == sp.Symbol("x")
    assert child.rhs == Integer(2)
    assert _verify(s, child) == VERIFY_PASS


def test_nth_root_x3_eq_neg_27_odd_no_block():
    """Odd root of negative is allowed (real_root gives the real branch)."""
    s = EqState.from_strings("x**3", "-27")
    rule = NthRootBothSides()
    actions = list(rule.enumerate(s))
    assert len(actions) == 1
    child = rule.apply(s, actions[0])
    # sp.real_root(-27, 3) = -3
    assert child.rhs == Integer(-3)
    assert _verify(s, child) == VERIFY_PASS


def test_nth_root_x4_eq_16_even_principal():
    s = EqState.from_strings("x**4", "16")
    rule = NthRootBothSides()
    actions = list(rule.enumerate(s))
    assert len(actions) == 1
    assert actions[0].params == (Integer(4),)
    child = rule.apply(s, actions[0])
    # Principal: x = 16^(1/4) = 2. SymPy may keep as Pow(16, 1/4) until simplified.
    assert sp.simplify(child.rhs - Integer(2)) == 0
    assert _verify(s, child) == VERIFY_PASS


def test_nth_root_blocks_even_root_of_negative():
    s = EqState.from_strings("x**4", "-16")
    rule = NthRootBothSides()
    assert list(rule.enumerate(s)) == []


def test_nth_root_skips_n2():
    """n=2 belongs to SqrtBothSides; NthRoot must not duplicate."""
    s = EqState.from_strings("x**2", "9")
    rule = NthRootBothSides()
    assert list(rule.enumerate(s)) == []


def test_nth_root_skips_non_pow():
    s = EqState.from_strings("x + 1", "5")
    rule = NthRootBothSides()
    assert list(rule.enumerate(s)) == []


# --- SPLIT_ABSOLUTE_VALUE -------------------------------------------------


def test_split_abs_x_eq_5():
    s = EqState.from_strings("Abs(x)", "5")
    rule = SplitAbsoluteValue()
    actions = list(rule.enumerate(s))
    assert len(actions) == 2
    params = sorted([a.params for a in actions], key=lambda p: int(p[0]))
    assert params == [(Integer(-1),), (Integer(1),)]
    # +branch: x = 5
    plus = next(a for a in actions if a.params == (Integer(1),))
    child_plus = rule.apply(s, plus)
    assert child_plus.lhs == sp.Symbol("x")
    assert sp.simplify(child_plus.rhs - Integer(5)) == 0
    assert _verify(s, child_plus) == VERIFY_PASS
    # -branch: x = -5
    minus = next(a for a in actions if a.params == (Integer(-1),))
    child_minus = rule.apply(s, minus)
    assert child_minus.lhs == sp.Symbol("x")
    assert sp.simplify(child_minus.rhs - Integer(-5)) == 0
    assert _verify(s, child_minus) == VERIFY_PASS


def test_split_abs_x_minus_1_eq_3():
    s = EqState.from_strings("Abs(x - 1)", "3")
    rule = SplitAbsoluteValue()
    actions = list(rule.enumerate(s))
    assert len(actions) == 2
    plus = next(a for a in actions if a.params == (Integer(1),))
    child_plus = rule.apply(s, plus)
    # Inner f(x) = x - 1 ; new rhs = 3
    assert sp.simplify(child_plus.lhs - (sp.Symbol("x") - 1)) == 0
    assert sp.simplify(child_plus.rhs - Integer(3)) == 0
    assert _verify(s, child_plus) == VERIFY_PASS
    minus = next(a for a in actions if a.params == (Integer(-1),))
    child_minus = rule.apply(s, minus)
    assert sp.simplify(child_minus.rhs - Integer(-3)) == 0
    assert _verify(s, child_minus) == VERIFY_PASS


def test_split_abs_blocks_negative_rhs():
    """|x| = -5 has no real solutions; rule should not fire."""
    s = EqState.from_strings("Abs(x)", "-5")
    rule = SplitAbsoluteValue()
    assert list(rule.enumerate(s)) == []


def test_split_abs_blocks_symbolic_rhs():
    """|x| = x + 1 — other side is symbolic; cannot verify nonneg; skip."""
    s = EqState.from_strings("Abs(x)", "x + 1")
    rule = SplitAbsoluteValue()
    assert list(rule.enumerate(s)) == []


def test_split_abs_skips_non_abs():
    s = EqState.from_strings("x + 1", "5")
    rule = SplitAbsoluteValue()
    assert list(rule.enumerate(s)) == []


# --- Integration: A* end-to-end with new rules ----------------------------


def test_astar_solves_cubic_x3_minus_8():
    from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
    from ggmr.search.astar import astar
    from ggmr.training.extract_pairs import _build_is_target

    s = EqState.from_strings("x**3 - 8", "0")
    target = EqState.from_strings("x", "2")
    result = astar(
        s,
        _build_is_target(target),
        heuristic=WeightedSumCompositeHeuristic(),
        max_nodes=5_000,
        max_depth=15,
    )
    assert result.found, (
        f"x^3 - 8 = 0 should solve with 48-rule library; "
        f"failed after {result.stats.nodes_expanded} nodes"
    )


def test_astar_solves_x3_eq_8_via_nth_root():
    from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
    from ggmr.search.astar import astar
    from ggmr.training.extract_pairs import _build_is_target

    s = EqState.from_strings("x**3", "8")
    target = EqState.from_strings("x", "2")
    result = astar(
        s,
        _build_is_target(target),
        heuristic=WeightedSumCompositeHeuristic(),
        max_nodes=1_000,
        max_depth=10,
    )
    assert result.found


def test_astar_solves_abs_value():
    """|x| = 5 should solve via SPLIT_ABSOLUTE_VALUE branching."""
    from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
    from ggmr.search.astar import astar
    from ggmr.training.extract_pairs import _build_is_target

    s = EqState.from_strings("Abs(x)", "5")
    target = EqState.from_strings("x", "5")
    result = astar(
        s,
        _build_is_target(target),
        heuristic=WeightedSumCompositeHeuristic(),
        max_nodes=1_000,
        max_depth=10,
    )
    assert result.found, (
        f"|x|=5 should solve via SPLIT_ABSOLUTE_VALUE; "
        f"failed after {result.stats.nodes_expanded} nodes"
    )
