"""Semantic chunker — splits parsed sections into retrieval-sized chunks.

Respects document structure (chapter → section → concept → definition/formula → example).
The current implementation splits on paragraph boundaries up to max_chars.
"""

from __future__ import annotations

from dataclasses import dataclass

from math_teacher.config.settings import settings
from math_teacher.domain.enums import ContentType
from math_teacher.ingestion.parser import ParsedSection


@dataclass
class TextChunk:
    """A semantic chunk ready for embedding and storage."""

    text: str
    chapter: str
    topic: str | None
    content_type: ContentType
    page_number: int
    chunk_index: int  # position within the document


def _split_on_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split text on double-newlines (paragraph boundaries).

    If a single paragraph exceeds max_chars, split on sentence boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Para itself larger than max_chars — split on sentence boundaries
            if len(para) > max_chars:
                sentences = para.replace(". ", ".\n").splitlines()
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) + 1 <= max_chars:
                        sub = (sub + " " + sent).strip()
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = sent
                if sub:
                    current = sub
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def chunk_sections(
    sections: list[ParsedSection],
    max_chars: int | None = None,
) -> list[TextChunk]:
    """Convert parsed sections into text chunks suitable for embedding.

    Args:
        sections: Parsed textbook sections.
        max_chars: Max characters per chunk (defaults to settings.max_chunk_chars).
    Returns:
        List of TextChunk objects with metadata preserved.
    """
    max_chars = max_chars or settings.max_chunk_chars
    chunks: list[TextChunk] = []
    global_index = 0

    for section in sections:
        raw_chunks = _split_on_paragraphs(section.text, max_chars)
        for raw in raw_chunks:
            if not raw.strip():
                continue
            chunks.append(
                TextChunk(
                    text=raw.strip(),
                    chapter=section.chapter,
                    topic=section.topic,
                    content_type=section.content_type,
                    page_number=section.page_number,
                    chunk_index=global_index,
                )
            )
            global_index += 1

    return chunks
