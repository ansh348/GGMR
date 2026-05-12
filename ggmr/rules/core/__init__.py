"""15 core guarded rewrite rules covering Phase 0 problem categories.

Registration with `default_registry` happens at import time. Importing this
package triggers all rule registrations.
"""

from . import arithmetic, algebra, rational, quadratic, polynomial, exponent  # noqa: F401

__all__: list[str] = []
