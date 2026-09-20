"""In-memory LRU/TTL cache for application performance optimization."""

from __future__ import annotations

import time
from typing import Any


class SimpleTTLCache:
    """Thread-safe LRU cache with time-to-live (TTL)' expiration."""

    def __init__(self, maxsize: int = 500, ttl_seconds: float = 3600.0) -> None:
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        """Retrieve value from cache if not expired."""
        if key not in self._cache:
            return None
        timestamp, value = self._cache[key]
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store value in cache, evicting oldest item if capacity is reached."""
        if len(self._cache) >= self.maxsize:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[key] = (time.time(), value)

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()


response_cache = SimpleTTLCache(maxsize=500, ttl_seconds=3600.0)
embedding_cache: dict[str, list[float]] = {}
