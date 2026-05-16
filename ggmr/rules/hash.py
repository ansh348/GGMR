"""Rule set hash for buffer / checkpoint metadata.

The Marcus replay-buffer invariant is that a buffer recorded against one
rule set must never be silently loaded against a different one (the
legal-mask shape and the policy logit positions would mismatch).

`rule_set_hash()` returns a SHA-256 hex digest of the sorted rule names
in `default_registry`. Algebra-only (49 rules) and algebra+trig (92 rules)
hash to different values. Buffer save embeds the hash; buffer load checks it.
"""

from __future__ import annotations

import hashlib

from .registry import Registry, default_registry


def rule_set_hash(registry: Registry | None = None) -> str:
    """SHA-256 hex digest of `'|'.join(sorted(registry.names()))`.

    Sorted (not insertion-ordered) so that the hash depends only on the
    rule set itself, not on the (alphabetically-irrelevant) registration
    order. Two registries with the same rule names hash identically.
    """
    if registry is None:
        registry = default_registry
    payload = "|".join(sorted(registry.names())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
