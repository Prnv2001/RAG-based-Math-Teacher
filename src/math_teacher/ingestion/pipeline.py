"""Full ingestion pipeline — orchestrates load → parse → chunk → embed → store."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from math_teacher.config.settings import settings
from math_teacher.domain.errors import IngestionError
from math_teacher.embeddings.base import Embedder
from math_teacher.ingestion.chunker import chunk_sections
from math_teacher.ingestion.loader import load_pdf
from math_teacher.ingestion.parser import parse_pages
from math_teacher.storage.models import Document, DocumentChunk


def _checksum(path: Path) -> str:
    """SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


async def ingest_pdf(
    *,
    pdf_path: str | Path,
    title: str,
    class_level: int,
    board: str = "CBSE",
    subject: str = "Mathematics",
    license_info: str | None = None,
    embedder: Embedder,
    session: AsyncSession,
) -> dict:
    """Full ingestion pipeline for a single textbook PDF.

    Steps:
        1. Load PDF pages via pypdf.
        2. Parse structural sections (chapter/topic/content_type detection).
        3. Chunk sections into retrieval-sized pieces.
        4. Generate embeddings for each chunk.
        5. Persist Document + DocumentChunk rows to PostgreSQL.

    Args:
        pdf_path: Path to the textbook PDF.
        title: Human-readable document title.
        class_level: Class level (9 or 10).
        board: Curriculum board (default CBSE).
        subject: Subject (default Mathematics).
        license_info: Optional licensing information.
        embedder: Embedder instance for generating vectors.
        session: Active async SQLAlchemy session.

    Returns:
        Summary dict with document_id, chunk_count, page_count.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise IngestionError(f"PDF not found: {pdf_path}")

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    pages = load_pdf(pdf_path)
    page_count = len(pages)

    # ------------------------------------------------------------------
    # 2. Parse structure
    # ------------------------------------------------------------------
    sections = parse_pages(pages, class_level=class_level)

    # ------------------------------------------------------------------
    # 3. Chunk
    # ------------------------------------------------------------------
    chunks = chunk_sections(sections)
    if not chunks:
        raise IngestionError("No chunks produced from document — check PDF content.")

    # ------------------------------------------------------------------
    # 4. Create Document row
    # ------------------------------------------------------------------
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title=title,
        source_type="pdf",
        source_uri=str(pdf_path.resolve()),
        license_info=license_info,
        checksum=_checksum(pdf_path),
    )
    session.add(doc)
    await session.flush()  # get the ID into DB before FK refs

    # ------------------------------------------------------------------
    # 5. Embed + store chunks
    # ------------------------------------------------------------------
    texts = [c.text for c in chunks]
    embeddings = await embedder.embed_many(texts)

    chunk_rows: list[DocumentChunk] = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        row = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            class_level=class_level,
            chapter=chunk.chapter,
            topic=chunk.topic,
            concept=None,  # future: extracted by LLM concept tagger
            content_type=chunk.content_type.value,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            embedding=embedding,
            metadata_={
                "board": board,
                "subject": subject,
                "class_level": class_level,
            },
        )
        chunk_rows.append(row)

    session.add_all(chunk_rows)
    await session.commit()

    return {
        "document_id": str(doc_id),
        "title": title,
        "class_level": class_level,
        "page_count": page_count,
        "section_count": len(sections),
        "chunk_count": len(chunk_rows),
    }
