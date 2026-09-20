"""Sentence-Transformers embedding provider — replaces OllamaEmbedder.

Uses ``all-mpnet-base-v2`` (768-dim) which matches the existing pgvector
column dimension, so no DB migration is required.

The model (~420 MB) is downloaded once on first use and cached locally.
Inference is CPU-only and takes ~50–150 ms per query on a laptop — far
faster than waiting for local Ollama.

Groq does not offer an embeddings API, so this local approach is the
recommended replacement when switching to Groq for LLM calls.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from math_teacher.domain.errors import EmbeddingError

logger = logging.getLogger(__name__)

# Model name — 768-dim output matches existing pgvector schema.
_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the SentenceTransformer model (cached after first call)."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from exc
    logger.info("Loading SentenceTransformer model '%s' (first-time download may take a moment)…", _MODEL_NAME)
    return SentenceTransformer(_MODEL_NAME)


class GroqEmbedder:
    """Local sentence-transformers embedder — drop-in for OllamaEmbedder.

    Public interface is identical to OllamaEmbedder:
        • embed(text: str) -> list[float]
        • embed_many(texts: list[str]) -> list[list[float]]

    The model runs on CPU. Typical latency: ~50–150 ms per query.
    """

    def __init__(self) -> None:
        # Model is loaded lazily on first embed() call — not at import time.
        pass

    async def embed(self, text: str) -> list[float]:
        """Embed a single string and return a 768-dim float vector."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")

        from math_teacher.utils.cache import embedding_cache

        cache_key = text.strip()
        if cache_key in embedding_cache:
            return embedding_cache[cache_key]

        try:
            # SentenceTransformer.encode() is synchronous; run in thread pool
            # so we don't block the FastAPI event loop.
            loop = asyncio.get_event_loop()
            vector = await loop.run_in_executor(
                None,
                lambda: _get_model().encode(text, normalize_embeddings=True).tolist(),
            )
            if len(embedding_cache) < 5000:
                embedding_cache[cache_key] = vector
            return vector
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"GroqEmbedder embed failed: {exc}") from exc

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings (batched internally for efficiency)."""
        if not texts:
            return []
        try:
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(
                None,
                lambda: _get_model()
                .encode(texts, normalize_embeddings=True, batch_size=32)
                .tolist(),
            )
            return vectors
        except Exception as exc:
            raise EmbeddingError(f"GroqEmbedder embed_many failed: {exc}") from exc
