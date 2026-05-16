"""Guarded rewrite rules for the GGMR solver.

Registration with `default_registry` happens at import time. Importing this
package triggers all rule registrations.

Algebra (49 rules) + Trigonometry (39 forward + 2 oracle) = 90 rules.
The 2 trig oracle shortcuts (TRIG_SIMPLIFY, TRIG_SOLVE) are gated by
`training_safe=False` and excluded from training-mode enumeration.
"""

from . import (  # noqa: F401
    algebra,
    arithmetic,
    exponent,
    polynomial,
    polynomial_advanced,
    quadratic,
    rational,
    trigonometry,
)

__all__: list[str] = []
