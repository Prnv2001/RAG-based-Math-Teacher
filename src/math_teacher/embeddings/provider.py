"""Ollama embedding provider — uses nomic-embed-text via local Ollama server."""

from __future__ import annotations

import ollama

from math_teacher.config.settings import settings
from math_teacher.domain.errors import EmbeddingError


class OllamaEmbedder:
    """Concrete embedding provider backed by Ollama (nomic-embed-text, 768-dim)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or settings.embedding_model
        self._client = ollama.AsyncClient(host=base_url or settings.ollama_base_url)

    async def embed(self, text: str) -> list[float]:
        """Embed a single string and return a float vector."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")
        try:
            response = await self._client.embeddings(model=self.model, prompt=text)
            return response.embedding
        except Exception as exc:
            raise EmbeddingError(f"Ollama embed failed: {exc}") from exc

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Calls embed() sequentially (Ollama has no batch API)."""
        results: list[list[float]] = []
        for text in texts:
            vec = await self.embed(text)
            results.append(vec)
        return results
