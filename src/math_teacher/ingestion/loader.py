"""PDF loader — extracts text per page using pypdf."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from math_teacher.domain.errors import IngestionError


@dataclass
class RawPage:
    """Raw text extracted from a single PDF page."""

    page_number: int  # 1-indexed
    text: str


def load_pdf(path: str | Path) -> list[RawPage]:
    """Load a PDF file and return a list of RawPage objects (one per page).

    Designed for pre-normalized textbook PDFs (text layer present).
    Args:
        path: Absolute path to the PDF file.
    Returns:
        List of RawPage objects sorted by page number.
    Raises:
        IngestionError: If the file cannot be read or is empty.
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise IngestionError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise IngestionError(f"Failed to open PDF: {exc}") from exc

    pages: list[RawPage] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        # Skip completely empty pages
        if text.strip():
            pages.append(RawPage(page_number=i, text=text))

    if not pages:
        raise IngestionError(f"No extractable text found in PDF: {pdf_path}")

    return pages
