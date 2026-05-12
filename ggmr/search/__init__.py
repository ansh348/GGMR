"""Search engines. Phase 1a: BFS with deduplication. Phase 1b adds A*/beam."""

from .bfs import bfs, SearchResult
from .stats import SearchStats

__all__ = ["bfs", "SearchResult", "SearchStats"]
