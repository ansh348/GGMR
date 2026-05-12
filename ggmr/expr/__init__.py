"""Expression tree primitives: canonical hashing, traversal, serialization."""

from .tree import (
    canonical_repr,
    normalize,
    tree_depth,
    op_count,
    leaf_count,
    iter_subtrees,
    replace_at_path,
)
from .serialize import to_prefix_notation, from_prefix_notation, parse_equation

__all__ = [
    "canonical_repr",
    "normalize",
    "tree_depth",
    "op_count",
    "leaf_count",
    "iter_subtrees",
    "replace_at_path",
    "to_prefix_notation",
    "from_prefix_notation",
    "parse_equation",
]
