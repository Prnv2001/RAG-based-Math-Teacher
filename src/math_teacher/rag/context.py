"""Context builder — assembles retrieved chunks into a bounded prompt context block."""

from __future__ import annotations

from math_teacher.config.settings import settings
from math_teacher.domain.models import RetrievedChunk, SourceReference


def build_context(
    chunks: list[RetrievedChunk],
    max_chars: int | None = None,
) -> tuple[str, list[SourceReference]]:
    """Assemble reranked chunks into a context string and source references.

    Stops adding chunks once max_chars would be exceeded (silent truncation).

    Args:
        chunks: Reranked retrieval results.
        max_chars: Maximum characters in context (default from settings).

    Returns:
        Tuple of (context_text, [SourceReference, ...])
    """
    max_chars = max_chars or settings.max_context_chars
    context_parts: list[str] = []
    sources: list[SourceReference] = []
    total_chars = 0

    seen_sources: set[tuple] = set()

    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[Source {i}: {chunk.document_title}"
            + (f" | Chapter: {chunk.chapter}" if chunk.chapter else "")
            + (f" | Topic: {chunk.topic}" if chunk.topic else "")
            + (f" | Page: {chunk.page_number}" if chunk.page_number else "")
            + "]\n"
        )
        from math_teacher.utils.text_sanitizer import clean_math_text
        clean_body = clean_math_text(chunk.text)
        block = header + clean_body + "\n"

        if total_chars + len(block) > max_chars:
            break  # Silent truncation — context is full

        context_parts.append(block)
        total_chars += len(block)

        # Deduplicate sources
        src_key = (chunk.document_title, chunk.chapter, chunk.topic, chunk.page_number)
        if src_key not in seen_sources:
            seen_sources.add(src_key)
            sources.append(
                SourceReference(
                    document=chunk.document_title,
                    chapter=chunk.chapter,
                    topic=chunk.topic,
                    page=chunk.page_number,
                )
            )

    context_text = "\n".join(context_parts)
    return context_text, sources
