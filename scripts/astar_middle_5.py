r"""Quick A* vs BFS comparison on the 5 productive-middle problems identified
from the in-progress coverage run. Regenerates each problem with the same seed
(replaying the seed assignment from validate_coverage.run_coverage), runs both
BFS and A*, prints a comparison table.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ggmr.expr.tree import canonical_repr
from ggmr.heuristics.composite import WeightedSumCompositeHeuristic
from ggmr.problems.generator import ReverseGenerator
from ggmr.rules.core import arithmetic, algebra, rational, quadratic, polynomial, exponent  # noqa: F401
from ggmr.search.astar import astar
from ggmr.search.bfs import bfs

# Grid: matches the running validate_coverage invocation
BASE_SEED = 0
DEPTHS = [5, 10, 15]
TEMPLATES = ["linear", "quadratic", "rational", "polynomial", "mixed"]
PROBLEMS_PER_BUCKET = 10


def compute_seed(depth: int, template: str, problem_idx: int) -> int:
    seed = BASE_SEED
    for d in DEPTHS:
        for t in TEMPLATES:
            for i in range(PROBLEMS_PER_BUCKET):
                if d == depth and t == template and i == problem_idx:
                    return seed
                seed += 100
    raise ValueError(f"not in grid: depth={depth}, template={template}, idx={problem_idx}")


PROBLEMS = [
    ("gen_polynomial_10_003", 10, "polynomial", 3),
    ("gen_rational_10_002",   10, "rational",   2),
    ("gen_quadratic_15_003",  15, "quadratic",  3),
    ("gen_mixed_10_005",      10, "mixed",      5),
    ("gen_quadratic_10_006",  10, "quadratic",  6),
]


def run_one(pid: str, depth: int, template: str, idx: int, astar_budget: int = 1000) -> dict:
    seed = compute_seed(depth, template, idx)
    gen = ReverseGenerator(seed=seed, depth=depth, template=template, max_nodes=5000)
    problem = gen.generate_one(max_attempts=3)
    if problem is None:
        return {"id": pid, "error": "generation returned None"}
    target = problem.target
    tlhs = canonical_repr(target.lhs)
    trhs = canonical_repr(target.rhs)

    def is_target(s):
        return (canonical_repr(s.lhs) == tlhs and canonical_repr(s.rhs) == trhs) or s.is_canonical_target()

    bfs_r = bfs(problem.initial, is_target, max_nodes=5000, max_depth=30, check_soundness=False)
    h = WeightedSumCompositeHeuristic()
    # Use a tighter budget for A* — if the heuristic is misleading and A* thrashes
    # near the 5000-node limit, the per-problem wall clock balloons. 1000 is enough
    # to demonstrate compression for problems where BFS solves in <300 nodes.
    astar_r = astar(problem.initial, is_target, heuristic=h, max_nodes=astar_budget,
                    max_depth=30, check_soundness=False)
    return {
        "id": pid,
        "seed": seed,
        "bfs_found": bfs_r.found,
        "bfs_nodes": bfs_r.stats.nodes_expanded,
        "bfs_steps": bfs_r.num_steps if bfs_r.found else None,
        "astar_found": astar_r.found,
        "astar_nodes": astar_r.stats.nodes_expanded,
        "astar_steps": astar_r.num_steps if astar_r.found else None,
        "ratio_bfs_over_astar": (
            bfs_r.stats.nodes_expanded / astar_r.stats.nodes_expanded
            if astar_r.stats.nodes_expanded > 0 else None
        ),
    }


def main() -> int:
    import time
    results = []
    print(f"{'Problem':<28s} {'BFS nodes':>10s} {'A* nodes':>10s} {'Reduction':>10s} "
          f"{'BFS steps':>10s} {'A* steps':>10s} {'wall (s)':>10s}", flush=True)
    print("-" * 92, flush=True)
    for pid, depth, template, idx in PROBLEMS:
        t0 = time.perf_counter()
        r = run_one(pid, depth, template, idx)
        elapsed = time.perf_counter() - t0
        results.append(r)
        if "error" in r:
            print(f"{r['id']:<28s} ERR: {r['error']}", flush=True)
            continue
        ratio = r["ratio_bfs_over_astar"]
        ratio_s = f"{ratio:.2f}x" if ratio is not None else "-"
        astar_marker = "*" if not r["astar_found"] else ""
        print(
            f"{r['id']:<28s} {r['bfs_nodes']:>10d} {r['astar_nodes']:>9d}{astar_marker:1s} "
            f"{ratio_s:>10s} {str(r['bfs_steps']):>10s} {str(r['astar_steps']):>10s} "
            f"{elapsed:>10.1f}",
            flush=True,
        )
    print("(* = A* hit budget without finding target)", flush=True)

    # Aggregate
    ratios = [r["ratio_bfs_over_astar"] for r in results if r.get("ratio_bfs_over_astar") is not None]
    if ratios:
        mean_ratio = sum(ratios) / len(ratios)
        print()
        print(f"Mean BFS/A* reduction: {mean_ratio:.2f}x  (geometric mean: "
              f"{(__import__('math').prod(ratios) ** (1 / len(ratios))):.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
