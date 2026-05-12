"""Cross-cutting soundness test: every registered rule, applied to every Phase 0
problem's initial state, must produce children whose solution sets are a subset
of the parent's effective solution set.

This is the predicate from `ggmr/PHASE1A_PREREG.md` §3.2 scaled to the full
rule × problem matrix at depth 1.
"""

from __future__ import annotations

import pytest

from ggmr.rules.core import *  # noqa: F401,F403
from ggmr.rules.registry import default_registry
from ggmr.soundness import (
    VERIFY_PASS,
    VERIFY_UNVERIFIABLE,
    verify_transition,
)


def test_all_registered_rules_sound_at_depth_1(phase0_states):
    """For every (Phase 0 initial state, rule, action) at depth 1:
    verify the produced child is sound (PASS) or unverifiable (skipped).
    Confirmed unsound transitions cause the test to fail.
    """
    sound, unverifiable, total = 0, 0, 0
    failures: list[str] = []
    for problem_id, initial, _target in phase0_states:
        for rule, action in default_registry.enumerate_actions(initial):
            g = rule.guard(initial, action)
            if not g.ok:
                continue
            try:
                child = rule.apply(initial, action)
            except Exception:
                continue
            if g.new_excluded or g.new_side_conditions:
                child = child.with_excluded(*g.new_excluded).with_side_conditions(
                    *g.new_side_conditions
                )
            verdict, reason = verify_transition(
                initial.lhs,
                initial.rhs,
                child.lhs,
                child.rhs,
                initial.var,
                parent_excluded=initial.excluded,
                child_excluded=child.excluded,
            )
            total += 1
            if verdict == VERIFY_PASS:
                sound += 1
            elif verdict == VERIFY_UNVERIFIABLE:
                unverifiable += 1
            else:
                failures.append(f"{problem_id} | {rule.name} | {action.params} | {reason}")
    if failures:
        pytest.fail(
            f"\nUnsound transitions detected ({len(failures)} of {total}):\n"
            + "\n".join(failures[:10])
            + ("\n...\n" if len(failures) > 10 else "")
        )
    assert sound + unverifiable == total
    # Sanity: at least some transitions should pass; we expect hundreds across 20 problems.
    assert sound > 50, f"expected many sound transitions, got {sound}"
