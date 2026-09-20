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
_HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2"


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the SentenceTransformer model (cached after first call)."""
    try:
        import torch  # type: ignore
        torch.set_num_threads(1)
    except Exception:
        pass
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
    """Lightweight embedder via Hugging Face Inference API with local fallback.

    Public interface is identical to OllamaEmbedder:
        • embed(text: str) -> list[float]
        • embed_many(texts: list[str]) -> list[list[float]]

    Uses HF Inference API first (0 MB RAM on cloud container).
    Falls back to local SentenceTransformer if network API is unreachable.
    """

    def __init__(self) -> None:
        pass

    async def _embed_hf_api(self, text: str) -> list[float] | None:
        """Call Hugging Face Free Inference API for 768-dim embeddings."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.post(
                    _HF_API_URL,
                    json={"inputs": text, "options": {"wait_for_model": True}},
                )
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list):
                        if data and isinstance(data[0], list):
                            return [float(x) for x in data[0]]
                        return [float(x) for x in data]
        except Exception as exc:
            logger.warning("HF API embedding call skipped: %s", exc)
        return None

    async def embed(self, text: str) -> list[float]:
        """Embed a single string and return a 768-dim float vector."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")

        from math_teacher.utils.cache import embedding_cache

        cache_key = text.strip()
        if cache_key in embedding_cache:
            return embedding_cache[cache_key]

        # 1. Try Hugging Face Cloud Inference API (0 MB RAM overhead on Render)
        vector = await self._embed_hf_api(cache_key)

        # 2. Local SentenceTransformer fallback if API is unreachable
        if not vector:
            try:
                loop = asyncio.get_event_loop()
                vector = await loop.run_in_executor(
                    None,
                    lambda: _get_model().encode(text, normalize_embeddings=True).tolist(),
                )
            except Exception as exc:
                raise EmbeddingError(f"GroqEmbedder embed failed: {exc}") from exc

        if len(embedding_cache) < 5000:
            embedding_cache[cache_key] = vector
        return vector

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings (batched internally for efficiency)."""
        if not texts:
            return []
        try:
            tasks = [self.embed(t) for t in texts]
            return await asyncio.gather(*tasks)
        except Exception as exc:
            raise EmbeddingError(f"GroqEmbedder embed_many failed: {exc}") from exc
