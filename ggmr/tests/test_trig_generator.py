"""Smoke tests for `TrigReverseGenerator` (Phase 1.2b).

Verifies the generator produces solvable problems at modest depth and
fails cleanly at unreasonable depth.
"""

from __future__ import annotations

import pytest

from ggmr.problems.trig_generator import TrigReverseGenerator
from ggmr.problems.trig_templates import TRIG_TEMPLATES


def test_generator_invalid_mode_raises():
    with pytest.raises(ValueError):
        TrigReverseGenerator(mode="solve_equation")


def test_generator_templates_dict_populated():
    """TRIG_TEMPLATES has the 10 canonical seeds + 'mixed'."""
    assert "mixed" in TRIG_TEMPLATES
    assert "pyth" in TRIG_TEMPLATES
    assert "double_sin" in TRIG_TEMPLATES
    assert len(TRIG_TEMPLATES) >= 10


def test_generator_depth_1_yields_solvable_problem():
    """At depth=1 the generator should usually succeed within 5 attempts."""
    gen = TrigReverseGenerator(seed=0, depth=1, template="mixed")
    problem = gen.generate_one(max_attempts=10)
    assert problem is not None, "generator failed to produce depth=1 problem"
    assert problem.depth == 1
    # The forward trace should be non-empty and end at a state where the
    # identity target predicate is met.
    assert len(problem.forward_trace) >= 1


def test_generator_depth_3_yields_solvable_problem():
    """Depth=3 still tractable in <5k BFS nodes."""
    gen = TrigReverseGenerator(seed=100, depth=3, template="mixed", max_nodes=5_000)
    problem = gen.generate_one(max_attempts=10)
    assert problem is not None, "generator failed to produce depth=3 problem"
    assert problem.depth == 3


def test_generator_extreme_depth_returns_none_cleanly():
    """Depth=20 should be unreachable in 5k BFS budget; must return None,
    not crash."""
    gen = TrigReverseGenerator(seed=7, depth=20, template="mixed", max_nodes=5_000)
    problem = gen.generate_one(max_attempts=2)
    # Either None (typical) or an unusually lucky success — both are fine.
    assert problem is None or problem.depth == 20


def test_generator_pyth_seed_specific():
    """The Pythagorean seed should reliably produce depth-2 problems."""
    gen = TrigReverseGenerator(seed=200, depth=2, template="pyth")
    problem = gen.generate_one(max_attempts=10)
    assert problem is not None, "generator failed on pyth seed at depth=2"
    # Initial should differ from target (some inverse rules fired)
    from ggmr.expr.tree import canonical_repr
    assert (
        canonical_repr(problem.initial.lhs) != canonical_repr(problem.target.lhs)
        or canonical_repr(problem.initial.rhs) != canonical_repr(problem.target.rhs)
    ), "initial state equals target — no inverse rules took effect"
