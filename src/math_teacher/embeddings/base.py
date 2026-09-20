"""Embedder provider Protocol — abstracts embedding model calls."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers.

    Concrete implementations: OllamaEmbedder (local nomic-embed-text).
    """

    async def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a float vector."""
        ...

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Returns a list of float vectors."""
        ...
