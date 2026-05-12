"""Random parameter samplers for the 7 motif families.

Each sampler returns a kwargs dict for its motif builder, or None if a
quick local check rules out the combination. The motif builder itself
raises ValueError on remaining degenerate cases; sample_motif_instance
wraps the build + verify_instance loop with bounded retry.
"""
from __future__ import annotations

import random
from typing import Optional

import sympy as sp

from ggmr.problems.motif_templates import (
    MotifInstance,
    motif_l1,
    motif_l3,
    motif_p3,
    motif_p4,
    motif_r1,
    motif_r2,
    motif_v1_ex1,
    verify_instance,
)


def _nonzero_pair(rng: random.Random, lo: int, hi: int) -> Optional[tuple[int, int]]:
    """Sample (a, b) from [lo, hi] with a != 0, b != 0, a != b."""
    candidates = [n for n in range(lo, hi + 1) if n != 0]
    if len(candidates) < 2:
        return None
    a = rng.choice(candidates)
    b_candidates = [n for n in candidates if n != a]
    if not b_candidates:
        return None
    b = rng.choice(b_candidates)
    return (a, b)


def _nonzero_int(rng: random.Random, lo: int, hi: int, exclude: set[int] = frozenset()) -> Optional[int]:
    candidates = [n for n in range(lo, hi + 1) if n != 0 and n not in exclude]
    if not candidates:
        return None
    return rng.choice(candidates)


def sample_v1_ex1(rng: random.Random) -> Optional[dict]:
    linear_coef = rng.choice([1, 2, 3, 4, 5, 6])
    lhs_const = rng.randint(-12, 12)
    rhs_const = rng.randint(-12, 12)
    if (rhs_const - lhs_const) % linear_coef != 0:
        return None
    twin = _nonzero_pair(rng, -7, 7)
    if twin is None:
        return None
    rational_root = _nonzero_int(rng, -6, 6)
    if rational_root is None:
        return None
    target_val = (rhs_const - lhs_const) // linear_coef
    if rational_root == target_val:
        return None
    return {
        "linear_coef": linear_coef,
        "lhs_const": lhs_const,
        "rhs_const": rhs_const,
        "twin_a": twin[0],
        "twin_b": twin[1],
        "rational_root": rational_root,
    }


def sample_l1(rng: random.Random) -> Optional[dict]:
    linear_coef = rng.choice([1, 2, 3, 4, 5, 6])
    lhs_const = rng.randint(-12, 12)
    rhs_const = rng.randint(-12, 12)
    if (rhs_const - lhs_const) % linear_coef != 0:
        return None
    twin1 = _nonzero_pair(rng, -7, 7)
    twin2 = _nonzero_pair(rng, -7, 7)
    if twin1 is None or twin2 is None:
        return None
    return {
        "linear_coef": linear_coef,
        "lhs_const": lhs_const,
        "rhs_const": rhs_const,
        "twin1": twin1,
        "twin2": twin2,
    }


def sample_l3(rng: random.Random) -> Optional[dict]:
    scalar = rng.choice([2, 3, 4, 5])
    inner_coef = rng.choice([1, 2, 3, 4, 5])
    inner_const = rng.randint(-12, 12)
    rhs_const = rng.randint(-12, 12)
    denom = scalar * inner_coef
    if (rhs_const - scalar * inner_const) % denom != 0:
        return None
    inner_twin = _nonzero_pair(rng, -6, 6)
    outer_twin = _nonzero_pair(rng, -6, 6)
    if inner_twin is None or outer_twin is None:
        return None
    return {
        "scalar": scalar,
        "inner_coef": inner_coef,
        "inner_const": inner_const,
        "rhs_const": rhs_const,
        "inner_twin": inner_twin,
        "outer_twin": outer_twin,
    }


def sample_p3(rng: random.Random) -> Optional[dict]:
    root_pool = list(range(-5, 6))
    if len(root_pool) < 3:
        return None
    roots = tuple(rng.sample(root_pool, 3))
    irreducible_p = rng.randint(-4, 4)
    irreducible_q = rng.randint(1, 8)
    if irreducible_p * irreducible_p - 4 * irreducible_q >= 0:
        return None
    decoy = _nonzero_pair(rng, -6, 6)
    if decoy is None:
        return None
    return {
        "roots": roots,
        "irreducible_p": irreducible_p,
        "irreducible_q": irreducible_q,
        "linear_decoy_pair": decoy,
    }


def sample_p4(rng: random.Random) -> Optional[dict]:
    root_pool = list(range(-5, 6))
    if len(root_pool) < 3:
        return None
    target_roots = tuple(rng.sample(root_pool, 3))
    forbidden = set(target_roots) | {0}
    denom1 = _nonzero_int(rng, -6, 6, exclude=forbidden)
    if denom1 is None:
        return None
    denom2 = _nonzero_int(rng, -6, 6, exclude=forbidden | {denom1})
    if denom2 is None:
        return None
    scalar = rng.choice([1, -1, 2, -2, 3, -3])
    twin = _nonzero_pair(rng, -6, 6)
    if twin is None:
        return None
    return {
        "target_roots": target_roots,
        "denom1": denom1,
        "denom2": denom2,
        "scalar": scalar,
        "twin": twin,
    }


def sample_r1(rng: random.Random) -> Optional[dict]:
    lhs_linear = rng.randint(1, 8)
    rhs_linear = rng.randint(1, 8)
    if lhs_linear == rhs_linear:
        return None
    lhs_const = rng.randint(-12, 12)
    rhs_const = rng.randint(-12, 12)
    if (rhs_const - lhs_const) % (lhs_linear - rhs_linear) == 0:
        # Integer target — degenerate as L1
        return None
    twin1 = _nonzero_pair(rng, -7, 7)
    twin2 = _nonzero_pair(rng, -7, 7)
    if twin1 is None or twin2 is None:
        return None
    return {
        "lhs_linear": lhs_linear,
        "rhs_linear": rhs_linear,
        "lhs_const": lhs_const,
        "rhs_const": rhs_const,
        "twin1": twin1,
        "twin2": twin2,
    }


def sample_r2(rng: random.Random) -> Optional[dict]:
    scalar = rng.choice([2, 3, 4, 5])
    inner_coef = rng.choice([1, 2, 3, 4, 5])
    inner_const = rng.randint(-12, 12)
    rhs_const = rng.randint(-12, 12)
    denom = scalar * inner_coef
    if (rhs_const - scalar * inner_const) % denom == 0:
        # Integer target — degenerate as L3
        return None
    inner_twin = _nonzero_pair(rng, -6, 6)
    outer_twin = _nonzero_pair(rng, -6, 6)
    if inner_twin is None or outer_twin is None:
        return None
    return {
        "scalar": scalar,
        "inner_coef": inner_coef,
        "inner_const": inner_const,
        "rhs_const": rhs_const,
        "inner_twin": inner_twin,
        "outer_twin": outer_twin,
    }


SAMPLERS = {
    "v1_ex1": sample_v1_ex1,
    "L1": sample_l1,
    "L3": sample_l3,
    "P3": sample_p3,
    "P4": sample_p4,
    "R1": sample_r1,
    "R2": sample_r2,
}

MOTIF_BUILDERS = {
    "v1_ex1": motif_v1_ex1,
    "L1": motif_l1,
    "L3": motif_l3,
    "P3": motif_p3,
    "P4": motif_p4,
    "R1": motif_r1,
    "R2": motif_r2,
}

FAMILIES = tuple(SAMPLERS.keys())


def sample_motif_instance(
    family: str, rng: random.Random, max_attempts: int = 10
) -> Optional[MotifInstance]:
    """Sample params, build instance, verify; return None if max_attempts exhausted."""
    sampler = SAMPLERS[family]
    builder = MOTIF_BUILDERS[family]
    var = sp.Symbol("x")
    for _ in range(max_attempts):
        kwargs = sampler(rng)
        if kwargs is None:
            continue
        try:
            instance = builder(var=var, **kwargs)
        except ValueError:
            continue
        except Exception:
            continue
        try:
            ok, _reason = verify_instance(instance)
        except Exception:
            continue
        if ok:
            return instance
    return None
