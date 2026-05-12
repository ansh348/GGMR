r"""Single-shot helper: do exactly one HardProblemGenerator.generate_one(max_attempts=1).

Reads job spec from --job-json (JSON string), generates ONE problem (or None),
prints the result dict as JSON to stdout. Used by `scripts/generate_hard_eval_set.py`
to enforce a hard wall-clock timeout per attempt: the parent runs this with
subprocess.run(..., timeout=N) and kills the process if it exceeds N seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job-json", required=True, help="JSON-encoded job spec")
    args = p.parse_args()
    job = json.loads(args.job_json)

    import ggmr.rules.core  # noqa: F401  (registers forward rules)
    from ggmr.problems.hard_generator import HardProblemGenerator, RECIPES_BY_NAME
    from ggmr.problems.hard_yaml_emit import hard_problem_to_dict

    recipe = RECIPES_BY_NAME[job["recipe"]]
    try:
        gen = HardProblemGenerator(
            recipe=recipe,
            depth=job["depth"],
            seed=job["seed"],
            min_astar_nodes=job["min_astar_nodes"],
            max_bfs_nodes=job["max_bfs_nodes"],
            max_bfs_depth=job["max_bfs_depth"],
            astar_max_nodes=job["astar_max_nodes"],
            astar_max_depth=job["astar_max_depth"],
            pre_bfs_complexity_max=job["pre_bfs_complexity_max"],
        )
        problem = gen.generate_one(max_attempts=job["max_attempts"])
    except Exception as e:
        print(json.dumps({
            "recipe": job["recipe"],
            "seed": job["seed"],
            "accepted": False,
            "error": f"{type(e).__name__}: {e}",
        }))
        return 0
    if problem is None:
        print(json.dumps({
            "recipe": job["recipe"],
            "seed": job["seed"],
            "accepted": False,
        }))
        return 0
    problem_id = f"hard_{job['recipe']}_{job['seed']:05d}"
    print(json.dumps({
        "recipe": job["recipe"],
        "seed": job["seed"],
        "accepted": True,
        "record": hard_problem_to_dict(problem, problem_id, job["recipe"]),
    }, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
