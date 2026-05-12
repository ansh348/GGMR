"""Walk utilities. Currently a thin re-export shim for tree primitives —
kept as a module so future Phase 1b walks (visitor patterns, e-graph lifting)
have a natural home.
"""

from __future__ import annotations

from .tree import iter_subtrees, replace_at_path

__all__ = ["iter_subtrees", "replace_at_path"]
