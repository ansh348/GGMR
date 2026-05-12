"""Tests for the ReverseGenerator: produces solvable problems at controlled depth."""

from __future__ import annotations

import pytest

from ggmr.problems.generator import ReverseGenerator
from ggmr.problems.yaml_emit import emit_problems_yaml, problem_to_dict


def test_generator_depth_5_linear():
    """Generate a depth-5 linear problem; assert it solves via BFS."""
    gen = ReverseGenerator(seed=42, depth=5, template="linear", max_nodes=5000)
    problem = gen.generate_one()
    assert problem is not None
    assert problem.template == "linear"
    assert problem.depth == 5
    assert problem.bfs_stats["nodes_expanded"] > 0


def test_generator_depth_10_linear():
    gen = ReverseGenerator(seed=43, depth=10, template="linear", max_nodes=5000)
    problem = gen.generate_one()
    assert problem is not None
    assert problem.template == "linear"


def test_generator_quadratic():
    gen = ReverseGenerator(seed=44, depth=3, template="quadratic", max_nodes=5000)
    problem = gen.generate_one()
    assert problem is not None


def test_generator_determinism():
    """Same seed → byte-identical problem."""
    gen1 = ReverseGenerator(seed=100, depth=5, template="linear")
    gen2 = ReverseGenerator(seed=100, depth=5, template="linear")
    p1 = gen1.generate_one()
    p2 = gen2.generate_one()
    assert p1 is not None and p2 is not None
    # Compare structurally
    assert str(p1.initial.lhs) == str(p2.initial.lhs)
    assert str(p1.initial.rhs) == str(p2.initial.rhs)


def test_generator_yaml_roundtrip(tmp_path):
    gen = ReverseGenerator(seed=7, depth=5, template="linear", max_nodes=5000)
    problem = gen.generate_one()
    assert problem is not None
    out = tmp_path / "test_problems.yaml"
    emit_problems_yaml([problem], str(out))
    assert out.exists()
    # Roundtrip via yaml
    import yaml
    with open(out, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert isinstance(loaded, list) and len(loaded) == 1
    entry = loaded[0]
    assert entry["category"] == "linear"
    assert entry["variable"] == "x"
    assert "initial" in entry and "lhs" in entry["initial"]
    assert "canonical_target" in entry
    assert "trace" in entry
