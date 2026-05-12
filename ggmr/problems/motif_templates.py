"""Motif templates for hard-equation problems (Phase 2).

Direct parameterized construction of equations exhibiting the seven motif
families validated on 2026-05-12. Bypasses the reverse-generator entirely.

Seven motif families:
  v1_ex1: linear target + rational twin + polynomial twin       (A*=216)
  L1:     linear target + 2 polynomial twins                    (A*=810)
  L3:     linear target + distributed scalar gate               (A*=907)
  P3:     cubic target via irreducible-quadratic disguise       (A*=542)
  P4:     cubic target with two-denominator cross-ratio         (A*=257)
  R1:     rational target via polynomial-difference hiding      (A*=1678)
  R2:     rational target with distributed scalar gate          (A*=1678)

Excluded from this set (validated as bad):
  L2, R3: pathological — constant-num rationals with cross-denoms.
  Q1-Q4, P1: trivial — FACTOR_POLYNOMIAL one-shots quadratic targets.

Each template:
  * uses `evaluate=False` throughout to preserve structural disguise.
  * raises `ValueError` on degenerate parameter choices.
  * returns a `MotifInstance` with both the initial and target `EqState`.

After construction, `verify_instance` checks algebraic correctness,
op-count budget, non-degeneracy, and excluded/target disjointness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import sympy as sp
from sympy import Add, Mul, Pow, Symbol, Integer, Rational, Expr

from ggmr.state import EqState
from ggmr.expr.tree import canonical_repr, op_count


# ---------------------------------------------------------------------------
# Construction helpers (evaluate=False everywhere)
# ---------------------------------------------------------------------------

def A(*args):
    return Add(*args, evaluate=False)


def M(*args):
    return Mul(*args, evaluate=False)


def P(b, e):
    return Pow(b, e, evaluate=False)


def I(n):
    return Integer(n)


# ---------------------------------------------------------------------------
# MotifInstance
# ---------------------------------------------------------------------------

@dataclass
class MotifInstance:
    """One generated problem with both initial state and target."""

    eq_state: EqState
    target_eq_state: EqState
    category: str          # "linear" | "polynomial" | "rational"
    motif_family: str      # "v1_ex1" | "L1" | "L3" | "P3" | "P4" | "R1" | "R2"
    params: dict = field(default_factory=dict)

    def to_record(self, problem_id: str, recipe: str = "motif_template_v2") -> dict:
        """Build a dict matching `hard_problem_to_dict` schema (validate_hard_set.py-compatible)."""
        initial = self.eq_state
        target = self.target_eq_state
        return {
            "id": problem_id,
            "category": self.category,
            "recipe": recipe,
            "difficulty": "hard",
            "variable": initial.var.name,
            "source": "ggmr-motif-template-v2",
            "seed": 0,
            "depth": 0,
            "astar_nodes_expanded": 0,
            "bfs_nodes_expanded": 0,
            "applied_inverses": [],
            "initial_srepr_lhs": sp.srepr(initial.lhs),
            "initial_srepr_rhs": sp.srepr(initial.rhs),
            "excluded_srepr": sorted(sp.srepr(e) for e in initial.excluded),
            "initial": {
                "lhs": str(initial.lhs),
                "rhs": str(initial.rhs),
            },
            "canonical_target": {
                "lhs": str(target.lhs),
                "rhs": str(target.rhs),
            },
        }


# ---------------------------------------------------------------------------
# Twin builders (semantic-twin decoys)
# ---------------------------------------------------------------------------

def _polynomial_twin(a: int, b: int, var: Symbol):
    """Factored form (x+a)(x+b), evaluate=False."""
    if a == 0 or b == 0:
        raise ValueError(f"polynomial twin has zero factor: ({a},{b})")
    if a == b:
        raise ValueError(f"polynomial twin is degenerate square: ({a},{b})")
    return M(A(var, I(a)), A(var, I(b)))


def _expanded_twin(a: int, b: int, var: Symbol):
    """Expanded form of (x+a)(x+b) = x² + (a+b)x + ab, evaluate=False."""
    if a == 0 or b == 0:
        raise ValueError(f"polynomial twin has zero factor: ({a},{b})")
    if a == b:
        raise ValueError(f"polynomial twin is degenerate square: ({a},{b})")
    s = a + b
    p = a * b
    if s == 0:
        return A(P(var, I(2)), I(p))
    return A(P(var, I(2)), M(I(s), var), I(p))


def _rational_twin_lhs(c: int, var: Symbol):
    """(x² - c²) / (x - c), evaluate=False — LHS form of rational twin."""
    if c == 0:
        raise ValueError("rational_root cannot be 0")
    return M(A(P(var, I(2)), I(-(c * c))), P(A(var, I(-c)), I(-1)))


def _rational_twin_rhs(c: int, var: Symbol):
    """(x + c)(x - c) / (x - c), evaluate=False — RHS form of rational twin."""
    if c == 0:
        raise ValueError("rational_root cannot be 0")
    return M(M(A(var, I(c)), A(var, I(-c))), P(A(var, I(-c)), I(-1)))


def _poly_from_coeffs(a: int, b: int, c: int, var: Symbol):
    """Build a·x² + b·x + c with evaluate=False, dropping zero terms."""
    terms = []
    if a != 0:
        terms.append(M(I(a), P(var, I(2))))
    if b != 0:
        terms.append(M(I(b), var))
    if c != 0:
        terms.append(I(c))
    if not terms:
        return I(0)
    if len(terms) == 1:
        return terms[0]
    return A(*terms)


# ---------------------------------------------------------------------------
# Template functions
# ---------------------------------------------------------------------------

def motif_v1_ex1(*, var: Symbol, linear_coef: int, lhs_const: int, rhs_const: int,
                 twin_a: int, twin_b: int, rational_root: int) -> MotifInstance:
    """v1 Ex1: linear target + rational twin + polynomial twin (validated A*=216).

      LHS = linear_coef·x + lhs_const + (x+twin_a)(x+twin_b) + (x² - c²)/(x - c)
      RHS = rhs_const + (x² + (a+b)x + ab) + (x + c)(x - c)/(x - c)
      where c = rational_root.

      target: x = (rhs_const - lhs_const) / linear_coef  (must be integer)
      excluded: {c}
    """
    if linear_coef == 0:
        raise ValueError("linear_coef cannot be 0")
    if rational_root == 0:
        raise ValueError("rational_root cannot be 0")
    if (rhs_const - lhs_const) % linear_coef != 0:
        raise ValueError(f"non-integer target: ({rhs_const}-{lhs_const})/{linear_coef}")
    target_val = (rhs_const - lhs_const) // linear_coef
    if rational_root == target_val:
        raise ValueError(f"excluded value {rational_root} equals target {target_val}")

    lhs = A(
        M(I(linear_coef), var),
        I(lhs_const),
        _polynomial_twin(twin_a, twin_b, var),
        _rational_twin_lhs(rational_root, var),
    )
    rhs = A(
        I(rhs_const),
        _expanded_twin(twin_a, twin_b, var),
        _rational_twin_rhs(rational_root, var),
    )
    eq_state = EqState(lhs=lhs, rhs=rhs, var=var, excluded=frozenset({I(rational_root)}))
    target_eq_state = EqState(lhs=var, rhs=I(target_val), var=var)
    return MotifInstance(
        eq_state=eq_state,
        target_eq_state=target_eq_state,
        category="linear",
        motif_family="v1_ex1",
        params={
            "linear_coef": linear_coef, "lhs_const": lhs_const, "rhs_const": rhs_const,
            "twin_a": twin_a, "twin_b": twin_b, "rational_root": rational_root,
            "target_val": target_val,
        },
    )


def motif_l1(*, var: Symbol, linear_coef: int, lhs_const: int, rhs_const: int,
             twin1: Tuple[int, int], twin2: Tuple[int, int]) -> MotifInstance:
    """L1: linear target + 2 polynomial twins (validated A*=810).

      LHS = linear_coef·x + lhs_const + (x+a₁)(x+b₁) + (x+a₂)(x+b₂)
      RHS = rhs_const + (expanded₁) + (expanded₂)

      target: x = (rhs_const - lhs_const) / linear_coef
    """
    if linear_coef == 0:
        raise ValueError("linear_coef cannot be 0")
    if (rhs_const - lhs_const) % linear_coef != 0:
        raise ValueError(f"non-integer target: ({rhs_const}-{lhs_const})/{linear_coef}")
    target_val = (rhs_const - lhs_const) // linear_coef
    a1, b1 = twin1
    a2, b2 = twin2

    lhs = A(
        M(I(linear_coef), var),
        I(lhs_const),
        _polynomial_twin(a1, b1, var),
        _polynomial_twin(a2, b2, var),
    )
    rhs = A(
        I(rhs_const),
        _expanded_twin(a1, b1, var),
        _expanded_twin(a2, b2, var),
    )
    return MotifInstance(
        eq_state=EqState(lhs=lhs, rhs=rhs, var=var),
        target_eq_state=EqState(lhs=var, rhs=I(target_val), var=var),
        category="linear",
        motif_family="L1",
        params={
            "linear_coef": linear_coef, "lhs_const": lhs_const, "rhs_const": rhs_const,
            "twin1": twin1, "twin2": twin2, "target_val": target_val,
        },
    )


def motif_l3(*, var: Symbol, scalar: int, inner_coef: int, inner_const: int,
             rhs_const: int, inner_twin: Tuple[int, int],
             outer_twin: Tuple[int, int]) -> MotifInstance:
    """L3: linear target + distributed scalar gate + polynomial twin (validated A*=907).

      LHS = scalar·(inner_coef·x + inner_const + (x+a)(x+b)) + (x+c)(x+d)
      RHS = rhs_const + scalar·(expanded_inner) + (expanded_outer)

      target: x = (rhs_const - scalar·inner_const) / (scalar·inner_coef)
    """
    if scalar == 0 or inner_coef == 0:
        raise ValueError("scalar and inner_coef must be nonzero")
    denom = scalar * inner_coef
    numer = rhs_const - scalar * inner_const
    if numer % denom != 0:
        raise ValueError(f"non-integer target: {numer}/{denom}")
    target_val = numer // denom
    a, b = inner_twin
    c, d = outer_twin

    lhs = A(
        M(I(scalar),
          A(M(I(inner_coef), var), I(inner_const), _polynomial_twin(a, b, var))),
        _polynomial_twin(c, d, var),
    )
    rhs = A(
        I(rhs_const),
        M(I(scalar), _expanded_twin(a, b, var)),
        _expanded_twin(c, d, var),
    )
    return MotifInstance(
        eq_state=EqState(lhs=lhs, rhs=rhs, var=var),
        target_eq_state=EqState(lhs=var, rhs=I(target_val), var=var),
        category="linear",
        motif_family="L3",
        params={
            "scalar": scalar, "inner_coef": inner_coef, "inner_const": inner_const,
            "rhs_const": rhs_const, "inner_twin": inner_twin, "outer_twin": outer_twin,
            "target_val": target_val,
        },
    )


def motif_p3(*, var: Symbol, roots: Tuple[int, int, int],
             irreducible_p: int, irreducible_q: int,
             linear_decoy_pair: Tuple[int, int]) -> MotifInstance:
    """P3: cubic target via irreducible-quadratic disguise (validated A*=542).

      LHS = (x + r₁)(x² + p·x + q) + (x+a)(x+b)
      RHS = (computed_linear)·x + (computed_const) + (expanded twin)

      where (α, β, γ) = `roots` are the cubic target roots and r₁, computed_linear,
      computed_const are derived to satisfy the cubic identity. Requires the
      quadratic factor to be irreducible (discriminant < 0).

      target: (x-α)(x-β)(x-γ) = 0
    """
    disc = irreducible_p * irreducible_p - 4 * irreducible_q
    if disc >= 0:
        raise ValueError(
            f"quadratic x²+{irreducible_p}·x+{irreducible_q} not irreducible (disc={disc})"
        )
    alpha, beta, gamma = roots
    if len({alpha, beta, gamma}) != 3:
        raise ValueError(f"roots must be distinct: {roots}")
    s = alpha + beta + gamma
    t = alpha * beta + beta * gamma + gamma * alpha
    u = alpha * beta * gamma
    # LHS-RHS coefficients of x³,x²,x¹,x⁰ should match (x-α)(x-β)(x-γ) = x³ - sx² + tx - u
    r1 = -s - irreducible_p
    linear_decoy = irreducible_q + r1 * irreducible_p - t
    const_term = r1 * irreducible_q + u

    ta, tb = linear_decoy_pair
    # Build (x² + p·x + q) with evaluate=False
    irreducible_factor = A(P(var, I(2)), M(I(irreducible_p), var), I(irreducible_q))
    lhs = A(
        M(A(var, I(r1)), irreducible_factor),
        _polynomial_twin(ta, tb, var),
    )
    rhs_terms = []
    if linear_decoy != 0:
        rhs_terms.append(M(I(linear_decoy), var))
    if const_term != 0:
        rhs_terms.append(I(const_term))
    rhs_terms.append(_expanded_twin(ta, tb, var))
    rhs = A(*rhs_terms) if len(rhs_terms) > 1 else rhs_terms[0]

    target_lhs = M(A(var, I(-alpha)), A(var, I(-beta)), A(var, I(-gamma)))
    target_rhs = I(0)
    return MotifInstance(
        eq_state=EqState(lhs=lhs, rhs=rhs, var=var),
        target_eq_state=EqState(lhs=target_lhs, rhs=target_rhs, var=var),
        category="polynomial",
        motif_family="P3",
        params={
            "roots": roots, "irreducible_p": irreducible_p, "irreducible_q": irreducible_q,
            "linear_decoy_pair": linear_decoy_pair,
            "r1_computed": r1, "linear_decoy_computed": linear_decoy,
            "const_computed": const_term,
        },
    )


def _solve_p4_coeffs(*, var: Symbol, target_roots, denom1: int, denom2: int,
                     scalar: int, free_d: int, free_e: int):
    """Solve the 4-equation linear system for P4 N₁, N₂ coefficients.

    Returns (a_v, b_v, c_v, f_v) as ints, or raises ValueError on degenerate.
    """
    r1, r2, r3 = target_roots
    a_sym, b_sym, c_sym, f_sym = sp.symbols("p4_a p4_b p4_c p4_f")
    expr = sp.expand(
        (a_sym * var ** 2 + b_sym * var + c_sym) * (var - denom2)
        - (Integer(free_d) * var ** 2 + Integer(free_e) * var + f_sym) * (var - denom1)
        - Integer(scalar) * (var - r1) * (var - r2) * (var - r3)
    )
    coeffs = sp.Poly(expr, var).all_coeffs()
    if len(coeffs) != 4:
        raise ValueError(f"P4 polynomial has unexpected degree, coeffs={coeffs}")
    sol_list = sp.solve(coeffs, [a_sym, b_sym, c_sym, f_sym], dict=True)
    if not sol_list:
        raise ValueError(f"P4 system has no solution")
    sol = sol_list[0]
    raw = (sol[a_sym], sol[b_sym], sol[c_sym], sol[f_sym])
    for name, val in zip(("a", "b", "c", "f"), raw):
        if not val.is_integer:
            raise ValueError(f"P4 non-integer coefficient {name}={val}")
    a_v, b_v, c_v, f_v = (int(v) for v in raw)
    if a_v == 0:
        raise ValueError("P4 N₁ has zero x² coefficient (degenerate)")
    return a_v, b_v, c_v, f_v


# Search range for (free_d, free_e) auto-tuning. Includes negative free_d so
# the search can find integer solutions across a wider divisibility space.
_P4_FREE_D_CANDIDATES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5)
_P4_FREE_E_CANDIDATES = (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6)


def motif_p4(*, var: Symbol, target_roots: Tuple[int, int, int],
             denom1: int, denom2: int, scalar: int,
             twin: Tuple[int, int],
             free_d: int | None = None, free_e: int | None = None) -> MotifInstance:
    """P4: cubic target with two-denominator cross-ratio (validated A*=257).

      LHS = N₁(x)/(x - d₁) + (x+a)(x+b)
      RHS = N₂(x)/(x - d₂) + (expanded twin)

      N₁ = a_v·x² + b_v·x + c_v, N₂ = free_d·x² + free_e·x + f_v
      with (a_v, b_v, c_v, f_v) solved so:
        N₁(x-d₂) - N₂(x-d₁) = scalar·(x-r₁)(x-r₂)(x-r₃)

      If `free_d` / `free_e` are None, searches a small candidate grid for the
      first pair that yields all-integer N₁ coefficients.

      Requires denominators distinct and not target roots.
      target: (x-r₁)(x-r₂)(x-r₃) = 0; excluded: {d₁, d₂}.
    """
    r1, r2, r3 = target_roots
    if len({r1, r2, r3}) != 3:
        raise ValueError(f"target_roots must be distinct: {target_roots}")
    if denom1 in target_roots or denom2 in target_roots:
        raise ValueError(f"denominator coincides with target root: ({denom1},{denom2}) vs {target_roots}")
    if denom1 == denom2:
        raise ValueError(f"denominators must be distinct: {denom1}")
    if scalar == 0:
        raise ValueError("scalar must be nonzero")

    # Resolve free_d, free_e — explicit-then-search
    resolved: tuple[int, int, int, int, int, int] | None = None
    last_err: str | None = None
    explicit = free_d is not None and free_e is not None
    if explicit:
        try:
            coeffs = _solve_p4_coeffs(var=var, target_roots=target_roots,
                                      denom1=denom1, denom2=denom2,
                                      scalar=scalar, free_d=free_d, free_e=free_e)
            resolved = (*coeffs, free_d, free_e)
        except ValueError as e:
            last_err = str(e)
            # Fall through to search
    if resolved is None:
        for fd in _P4_FREE_D_CANDIDATES:
            for fe in _P4_FREE_E_CANDIDATES:
                try:
                    coeffs = _solve_p4_coeffs(var=var, target_roots=target_roots,
                                              denom1=denom1, denom2=denom2,
                                              scalar=scalar, free_d=fd, free_e=fe)
                except ValueError as e:
                    last_err = str(e)
                    continue
                resolved = (*coeffs, fd, fe)
                break
            if resolved is not None:
                break
    if resolved is None:
        raise ValueError(
            f"P4: no valid (free_d, free_e) for target={target_roots}, "
            f"denoms=({denom1},{denom2}), scalar={scalar}. Last error: {last_err}"
        )
    a_v, b_v, c_v, f_v, free_d_used, free_e_used = resolved

    N1 = _poly_from_coeffs(a_v, b_v, c_v, var)
    N2 = _poly_from_coeffs(free_d_used, free_e_used, f_v, var)
    if N2 == I(0):
        raise ValueError("P4 N₂ is zero polynomial; degenerate")

    ta, tb = twin
    lhs = A(
        M(N1, P(A(var, I(-denom1)), I(-1))),
        _polynomial_twin(ta, tb, var),
    )
    rhs = A(
        M(N2, P(A(var, I(-denom2)), I(-1))),
        _expanded_twin(ta, tb, var),
    )

    target_lhs = M(A(var, I(-r1)), A(var, I(-r2)), A(var, I(-r3)))
    target_rhs = I(0)
    excluded = frozenset({I(denom1), I(denom2)})
    return MotifInstance(
        eq_state=EqState(lhs=lhs, rhs=rhs, var=var, excluded=excluded),
        target_eq_state=EqState(lhs=target_lhs, rhs=target_rhs, var=var),
        category="polynomial",
        motif_family="P4",
        params={
            "target_roots": target_roots, "denom1": denom1, "denom2": denom2,
            "scalar": scalar, "twin": twin,
            "free_d_requested": free_d, "free_e_requested": free_e,
            "free_d_used": free_d_used, "free_e_used": free_e_used,
            "N1_coeffs": (a_v, b_v, c_v), "N2_coeffs": (free_d_used, free_e_used, f_v),
        },
    )


def motif_r1(*, var: Symbol, lhs_linear: int, rhs_linear: int,
             lhs_const: int, rhs_const: int,
             twin1: Tuple[int, int], twin2: Tuple[int, int]) -> MotifInstance:
    """R1: rational target via polynomial-difference hiding (validated A*=1678).

      LHS = lhs_linear·x + lhs_const + (x+a₁)(x+b₁) + (x+a₂)(x+b₂)
      RHS = rhs_linear·x + rhs_const + (expanded₁) + (expanded₂)

      target: x = (rhs_const - lhs_const) / (lhs_linear - rhs_linear)
              — must be non-integer rational
    """
    diff_lin = lhs_linear - rhs_linear
    if diff_lin == 0:
        raise ValueError("lhs_linear == rhs_linear (target undefined)")
    target = Rational(rhs_const - lhs_const, diff_lin)
    if target.is_integer:
        raise ValueError(f"R1 target {target} is integer (degenerate as L1)")
    a1, b1 = twin1
    a2, b2 = twin2

    lhs = A(
        M(I(lhs_linear), var),
        I(lhs_const),
        _polynomial_twin(a1, b1, var),
        _polynomial_twin(a2, b2, var),
    )
    rhs = A(
        M(I(rhs_linear), var),
        I(rhs_const),
        _expanded_twin(a1, b1, var),
        _expanded_twin(a2, b2, var),
    )
    return MotifInstance(
        eq_state=EqState(lhs=lhs, rhs=rhs, var=var),
        target_eq_state=EqState(lhs=var, rhs=target, var=var),
        category="rational",
        motif_family="R1",
        params={
            "lhs_linear": lhs_linear, "rhs_linear": rhs_linear,
            "lhs_const": lhs_const, "rhs_const": rhs_const,
            "twin1": twin1, "twin2": twin2,
            "target": str(target),
        },
    )


def motif_r2(*, var: Symbol, scalar: int, inner_coef: int, inner_const: int,
             rhs_const: int, inner_twin: Tuple[int, int],
             outer_twin: Tuple[int, int]) -> MotifInstance:
    """R2: rational target with distributed scalar gate (validated A*=1678).

      LHS = scalar·(inner_coef·x + inner_const + (x+a)(x+b)) + (x+c)(x+d)
      RHS = rhs_const + scalar·(expanded_inner) + (expanded_outer)

      target: x = (rhs_const - scalar·inner_const) / (scalar·inner_coef)
              — must be non-integer rational
    """
    if scalar == 0 or inner_coef == 0:
        raise ValueError("scalar and inner_coef must be nonzero")
    denom = scalar * inner_coef
    numer = rhs_const - scalar * inner_const
    target = Rational(numer, denom)
    if target.is_integer:
        raise ValueError(f"R2 target {target} is integer (degenerate as L3)")
    a, b = inner_twin
    c, d = outer_twin

    lhs = A(
        M(I(scalar),
          A(M(I(inner_coef), var), I(inner_const), _polynomial_twin(a, b, var))),
        _polynomial_twin(c, d, var),
    )
    rhs = A(
        I(rhs_const),
        M(I(scalar), _expanded_twin(a, b, var)),
        _expanded_twin(c, d, var),
    )
    return MotifInstance(
        eq_state=EqState(lhs=lhs, rhs=rhs, var=var),
        target_eq_state=EqState(lhs=var, rhs=target, var=var),
        category="rational",
        motif_family="R2",
        params={
            "scalar": scalar, "inner_coef": inner_coef, "inner_const": inner_const,
            "rhs_const": rhs_const, "inner_twin": inner_twin, "outer_twin": outer_twin,
            "target": str(target),
        },
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

OP_COUNT_BUDGET = 60  # matches pre_bfs_complexity_max in hard_generator.py


def verify_instance(inst: MotifInstance) -> Tuple[bool, str]:
    """Algebraic + structural sanity check for a constructed `MotifInstance`.

    Returns `(ok, reason)`. Caller (driver script + tests) skips on `False`.

    Checks:
      1. op-count(lhs) + op-count(rhs) ≤ OP_COUNT_BUDGET.
      2. canonical_repr(lhs) ≠ canonical_repr(rhs) (non-trivial initial state).
      3. excluded values not in target solution set.
      4. solution set of (lhs - rhs) (excluding excluded) == target solution set.
    """
    eq = inst.eq_state
    tgt = inst.target_eq_state
    var = eq.var

    # 1. Op-count budget
    total_ops = op_count(eq.lhs) + op_count(eq.rhs)
    if total_ops > OP_COUNT_BUDGET:
        return False, f"op_count={total_ops} > {OP_COUNT_BUDGET}"

    # 2. Non-degeneracy
    if canonical_repr(eq.lhs) == canonical_repr(eq.rhs):
        return False, "trivial: canonical_repr(lhs) == canonical_repr(rhs)"

    # 3. Expected target solutions (from target EqState)
    try:
        expected = tgt.solution_set()
    except Exception as e:
        return False, f"target solution_set failed: {e}"
    if not expected:
        return False, f"target has no solutions: {tgt}"

    # 4. Excluded ∩ target = ∅
    excluded_simplified = {sp.simplify(e) for e in eq.excluded}
    for ex in excluded_simplified:
        if ex in expected:
            return False, f"excluded value {ex} ∈ target solutions {expected}"

    # 5. Actual solutions match
    try:
        actual = eq.solution_set()
    except Exception as e:
        return False, f"initial solution_set failed: {e}"
    if actual != expected:
        return False, f"solutions {set(actual)} != expected {set(expected)}"

    return True, "ok"
