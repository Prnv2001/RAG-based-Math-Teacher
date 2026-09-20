"""Grounding checker — embedding-based similarity check (no LLM call).

V2: Replaced the LLM-based grounding evaluator with cosine similarity
between answer and context embeddings using the local
sentence-transformers model (already loaded for retrieval).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np

from math_teacher.config.settings import settings
from math_teacher.domain.models import GroundingResult

if TYPE_CHECKING:
    from math_teacher.embeddings.groq_provider import GroqEmbedder

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0.0:
        return 0.0
    return float(dot / norm)


class GroundingChecker:
    """Embedding-based factual grounding evaluator.

    Computes cosine similarity between the generated answer and the
    curriculum context. Both are embedded with the same local
    sentence-transformers model used for retrieval (CPU, ~50–150 ms).

    For better accuracy, the answer is split into sentences and each
    sentence is scored against the full context. The final score is the
    mean of per-sentence similarities.
    """

    def __init__(self, embedder: "GroqEmbedder") -> None:
        self._embedder = embedder

    async def check(self, answer: str, context: str) -> GroundingResult:
        """Run grounding check.

        Args:
            answer: The LLM-generated answer text.
            context: The curriculum context used to generate the answer.
        Returns:
            GroundingResult with grounded bool, score, and claim lists.
        """
        if not answer.strip():
            return GroundingResult(
                grounded=False,
                score=0.0,
                supported_claims=[],
                unsupported_claims=["Empty answer"],
            )

        if not context.strip():
            return GroundingResult(
                grounded=True,
                score=0.50,
                supported_claims=[answer[:100]],
                unsupported_claims=[],
            )

        try:
            # Split answer into sentences for granular scoring
            sentences = _split_sentences(answer)
            if not sentences:
                sentences = [answer]

            # Split context into paragraphs for accurate per-chunk similarity
            context_blocks = [p.strip() for p in context.split("\n\n") if len(p.strip()) > 30]
            if not context_blocks:
                context_blocks = [context[:2000]]

            # Embed context blocks once
            block_embs = await self._embedder.embed_many(context_blocks)
            sentence_embs = await self._embedder.embed_many(sentences)

            supported: list[str] = []
            unsupported: list[str] = []
            scores: list[float] = []

            for sent, sent_emb in zip(sentences, sentence_embs):
                # Max similarity against any individual context block
                max_sim = max((_cosine_similarity(sent_emb, b_emb) for b_emb in block_embs), default=0.0)
                scores.append(max_sim)
                if max_sim >= settings.grounding_threshold:
                    supported.append(sent.strip())
                else:
                    unsupported.append(sent.strip())

            # Overall score = mean of top per-sentence similarities
            overall_score = float(np.mean(scores)) if scores else 0.0
            grounded = overall_score >= settings.grounding_threshold or len(supported) > 0 or (len(context_blocks) > 0 and overall_score >= 0.20)

            return GroundingResult(
                grounded=grounded,
                score=min(max(overall_score, 0.0), 1.0),
                supported_claims=supported,
                unsupported_claims=unsupported,
            )

        except Exception as exc:
            logger.warning("Grounding check failed: %s", exc)
            return GroundingResult(
                grounded=False,
                score=0.0,
                supported_claims=[],
                unsupported_claims=[f"Grounding check failed — {exc}"],
            )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics."""
    import re

    # Split on period/exclamation/question followed by space or end
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    # Filter out very short fragments (< 10 chars) and numbered step labels
    return [p for p in parts if len(p.strip()) >= 10]
