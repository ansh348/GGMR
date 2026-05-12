"""Search statistics dataclass — common across BFS/A*/beam (Phase 1b adds the others)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SearchStats:
    nodes_expanded: int = 0
    nodes_generated: int = 0
    dedup_hits: int = 0
    guard_rejections: int = 0
    max_depth_reached: int = 0
    time_ms: float = 0.0
    rule_application_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> dict:
        return {
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "dedup_hits": self.dedup_hits,
            "guard_rejections": self.guard_rejections,
            "max_depth_reached": self.max_depth_reached,
            "time_ms": self.time_ms,
            "rule_application_counts": dict(sorted(self.rule_application_counts.items())),
        }
