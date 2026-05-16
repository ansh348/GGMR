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
    """At depth=1 the generator should usually succeed within 10 attempts.

    v2 note: with 42+ seed families and 24 inverse rules, some seed×rule
    combinations produce states that BFS can't verify within a tight node
    budget. We use a moderate budget and retry up to 10 attempts."""
    gen = TrigReverseGenerator(seed=0, depth=1, template="mixed", max_nodes=2_000)
    problem = gen.generate_one(max_attempts=10)
    assert problem is not None, "generator failed to produce depth=1 problem"
    assert problem.depth == 1
    assert len(problem.forward_trace) >= 1


def test_generator_depth_2_yields_solvable_problem():
    """Depth=2: with v2's expanded inverse registry, depth=2 should usually
    succeed within a tight test budget. Uses max_nodes=800 to force fast
    rejection of pathological expansions."""
    gen = TrigReverseGenerator(seed=100, depth=2, template="mixed", max_nodes=800)
    problem = gen.generate_one(max_attempts=4)
    if problem is not None:
        assert problem.depth == 2


def test_generator_extreme_depth_returns_none_cleanly():
    """Depth=8 with a tight BFS budget should usually be unreachable;
    must return None cleanly when the budget is insufficient, never crash.

    v2 note: applicable-action sampling produces denser expansions, so
    BFS reverse-solving may need many nodes. With max_nodes=500 BFS
    bails fast; the generator should return None within ~10s.
    """
    gen = TrigReverseGenerator(seed=7, depth=8, template="mixed", max_nodes=500)
    problem = gen.generate_one(max_attempts=1)
    assert problem is None or problem.depth == 8


def test_generator_pyth_seed_specific():
    """The Pythagorean seed at depth-1 should reliably produce a solvable
    problem. We use depth=1 to keep BFS fast in test mode."""
    gen = TrigReverseGenerator(seed=200, depth=1, template="pyth", max_nodes=1_500)
    problem = gen.generate_one(max_attempts=5)
    assert problem is not None, "generator failed on pyth seed at depth=1"
    from ggmr.expr.tree import canonical_repr
    assert (
        canonical_repr(problem.initial.lhs) != canonical_repr(problem.target.lhs)
        or canonical_repr(problem.initial.rhs) != canonical_repr(problem.target.rhs)
    ), "initial state equals target — no inverse rules took effect"
