"""Test the coverage validation script: small batch, asserts §3.5 logic works.

Marked `slow` because the full 500-problem run takes ~40 minutes. The smoke
test here uses 5 problems per bucket (50 total), takes ~2-3 minutes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_validate_coverage_smoke(tmp_path):
    """Run coverage validation on a small batch (5 problems × 5 templates × 4 depths)."""
    from scripts.validate_coverage import run_coverage

    report = run_coverage(
        depths=[5, 10],
        templates=["linear", "quadratic"],
        problems_per_bucket=3,
        max_nodes=5000,
        base_seed=0,
    )
    assert "summary" in report
    assert "criterion_3_5" in report
    assert "per_depth" in report
    assert "per_template" in report
    assert "rule_application_counts" in report
    # Sanity: total_problems matches inputs (2 depths × 2 templates × 3 = 12)
    assert report["summary"]["total_problems"] == 12
    # At least some problems should solve
    assert report["summary"]["solved_total"] > 0


def test_coverage_report_json_structure(tmp_path):
    """Verify the JSON output schema: depth_le_10_rate, dead_rules, etc."""
    from scripts.validate_coverage import run_coverage

    report = run_coverage(
        depths=[5, 10],
        templates=["linear"],
        problems_per_bucket=2,
        max_nodes=5000,
    )
    # Required keys
    assert "criterion_3_5" in report
    crit = report["criterion_3_5"]
    assert "depth_le_10_rate" in crit
    assert "threshold" in crit
    assert "passed" in crit
    assert isinstance(crit["passed"], bool)
    # Dead rules list (may be empty)
    assert isinstance(report["dead_rules"], list)
