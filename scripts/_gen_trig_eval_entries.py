"""One-off: generate trig_evaluation_set_v1.yaml extension entries.

Produces deduplicated trig identity problems via TrigReverseGenerator across
all canonical templates. Writes YAML fragment to stdout.

Not committed long-term; one-off generator.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ggmr.expr.tree import canonical_repr  # noqa: E402
from ggmr.problems.trig_generator import TrigReverseGenerator  # noqa: E402
from ggmr.problems.trig_templates import TRIG_TEMPLATES  # noqa: E402


def _expr_to_yaml_str(expr) -> str:
    s = str(expr)
    needs_quote = any(c in s for c in '*/:-+()')
    return f'"{s}"' if needs_quote else s


def emit_entry(out, pid: str, category: str, difficulty: str, lhs, rhs, depth: int):
    out.write(f"- id: {pid}\n")
    out.write(f"  category: {category}\n")
    out.write(f"  difficulty: {difficulty}\n")
    out.write(f"  variable: x\n")
    out.write(f'  source: "TrigReverseGenerator depth={depth}"\n')
    out.write(f"  initial:\n")
    out.write(f"    lhs: {_expr_to_yaml_str(lhs)}\n")
    out.write(f"    rhs: {_expr_to_yaml_str(rhs)}\n")
    out.write(f"  canonical_target:\n")
    out.write(f'    lhs: "0"\n')
    out.write(f'    rhs: "0"\n')
    out.write("\n")


# Don't include "mixed" — iterate concrete templates for diversity
_CONCRETE_TEMPLATES = [t for t in TRIG_TEMPLATES.keys() if t != "mixed"]


def generate_unique(start_seed: int, depths: list[int], count: int, diff_label: str,
                    pid_prefix: str, start_pid_idx: int,
                    seen_initials: set[str], out, max_attempts_per_pid: int = 40):
    accepted = 0
    seed = start_seed
    attempts = 0
    n_dupes = 0
    template_cycle = list(_CONCRETE_TEMPLATES)
    while accepted < count and attempts < count * max_attempts_per_pid:
        attempts += 1
        depth = depths[attempts % len(depths)]
        template = template_cycle[attempts % len(template_cycle)]
        gen = TrigReverseGenerator(seed=seed, depth=depth, template=template,
                                   max_nodes=5000)
        problem = gen.generate_one(max_attempts=2)
        seed += 1
        if problem is None:
            continue
        # Dedup: canonical_repr of (lhs, rhs) tuple
        key = canonical_repr(problem.initial.lhs) + "|" + canonical_repr(problem.initial.rhs)
        if key in seen_initials:
            n_dupes += 1
            continue
        seen_initials.add(key)
        pid_idx = start_pid_idx + accepted
        emit_entry(
            out,
            pid=f"{pid_prefix}{pid_idx:02d}",
            category=problem.template,
            difficulty=diff_label,
            lhs=problem.initial.lhs,
            rhs=problem.initial.rhs,
            depth=depth,
        )
        accepted += 1
    print(
        f"# {pid_prefix}*: accepted={accepted}/{count} dupes_skipped={n_dupes} "
        f"({attempts} attempts)",
        file=sys.stderr,
    )


def main():
    out = sys.stdout
    seen: set[str] = set()
    out.write("# AUTO-GENERATED extension entries\n")
    out.write("# Append to trig_evaluation_set_v1.yaml\n\n")

    out.write("# ---- easy programmatic (trig_e11 - e20) -------------------------------\n")
    generate_unique(
        start_seed=10_000, depths=[1, 2], count=10, diff_label="easy",
        pid_prefix="trig_e", start_pid_idx=11,
        seen_initials=seen, out=out,
    )

    out.write("# ---- medium programmatic (trig_m11 - m30) -----------------------------\n")
    generate_unique(
        start_seed=20_000, depths=[3, 4, 5], count=20, diff_label="medium",
        pid_prefix="trig_m", start_pid_idx=11,
        seen_initials=seen, out=out,
    )

    out.write("# ---- hard programmatic (trig_h11 - h40) -------------------------------\n")
    generate_unique(
        start_seed=30_000, depths=[6, 7, 8, 9, 10], count=30, diff_label="hard",
        pid_prefix="trig_h", start_pid_idx=11,
        seen_initials=seen, out=out,
    )


if __name__ == "__main__":
    main()
