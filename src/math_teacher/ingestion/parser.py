"""Document structure parser — detects chapter/section/content boundaries in CBSE textbooks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from math_teacher.domain.enums import ContentType
from math_teacher.ingestion.loader import RawPage


# ---------------------------------------------------------------------------
# Patterns (tuned for CBSE Class 9 & 10 normalized PDFs)
# ---------------------------------------------------------------------------

# Chapter headings: "Chapter 1", "CHAPTER 1", "1. Real Numbers", etc.
_CHAPTER_RE = re.compile(
    r"^(?:chapter\s+\d+|(?:chapter\s+)?(\d+)[.\s]+[A-Z][A-Z\s]{3,})",
    re.IGNORECASE | re.MULTILINE,
)

# Section / topic headings: "1.1", "2.3 Polynomials", standalone CAPS lines
_SECTION_RE = re.compile(
    r"^(\d+\.\d+(?:\.\d+)?)\s+(.+)$",
    re.MULTILINE,
)

# Definition blocks
_DEFINITION_RE = re.compile(r"\b(definition|define)\b", re.IGNORECASE)

# Theorem / lemma
_THEOREM_RE = re.compile(r"\b(theorem|lemma|corollary|proposition)\b", re.IGNORECASE)

# Formula hints
_FORMULA_RE = re.compile(r"(formula|identity|equation|law)\b", re.IGNORECASE)

# Example blocks
_EXAMPLE_RE = re.compile(r"^\s*example\s*\d*[:\.]", re.IGNORECASE | re.MULTILINE)

# Exercise blocks
_EXERCISE_RE = re.compile(r"^\s*exercise\s*\d*[:\.]?", re.IGNORECASE | re.MULTILINE)

# Solution blocks
_SOLUTION_RE = re.compile(r"^\s*solution\s*[:\.]?", re.IGNORECASE | re.MULTILINE)


@dataclass
class ParsedSection:
    """A logical section parsed from the textbook."""

    chapter: str
    topic: str | None
    content_type: ContentType
    text: str
    page_number: int
    raw_heading: str = ""


def detect_content_type(text: str) -> ContentType:
    """Classify a block of text into a ContentType."""
    if _EXERCISE_RE.search(text):
        return ContentType.EXERCISE
    if _SOLUTION_RE.search(text):
        return ContentType.SOLUTION
    if _EXAMPLE_RE.search(text):
        return ContentType.EXAMPLE
    if _DEFINITION_RE.search(text):
        return ContentType.DEFINITION
    if _THEOREM_RE.search(text):
        return ContentType.THEOREM
    if _FORMULA_RE.search(text):
        return ContentType.FORMULA
    return ContentType.THEORY


def parse_pages(pages: list[RawPage], class_level: int) -> list[ParsedSection]:
    """Parse raw pages into structured sections.

    Strategy:
    1. Walk pages in order.
    2. Track current chapter and topic via regex.
    3. Detect content_type per page block.
    4. Group contiguous same-chapter/topic text into sections.
    """
    sections: list[ParsedSection] = []
    current_chapter = f"Class {class_level} Mathematics"
    current_topic: str | None = None
    buffer: list[str] = []
    buffer_page = 1
    buffer_type = ContentType.THEORY

    def flush_buffer() -> None:
        nonlocal buffer, buffer_type
        text = "\n".join(buffer).strip()
        if text:
            sections.append(
                ParsedSection(
                    chapter=current_chapter,
                    topic=current_topic,
                    content_type=buffer_type,
                    text=text,
                    page_number=buffer_page,
                    raw_heading="",
                )
            )
        buffer = []

    for raw_page in pages:
        lines = raw_page.text.splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Detect chapter boundary
            chapter_match = _CHAPTER_RE.match(stripped)
            if chapter_match:
                flush_buffer()
                current_chapter = stripped[:120]
                current_topic = None
                buffer_page = raw_page.page_number
                buffer_type = ContentType.THEORY
                continue

            # Detect section / topic boundary
            section_match = _SECTION_RE.match(stripped)
            if section_match:
                flush_buffer()
                current_topic = stripped[:120]
                buffer_page = raw_page.page_number
                buffer_type = ContentType.THEORY
                continue

            # Accumulate text into buffer
            buffer.append(stripped)

        # Update content type based on accumulated buffer
        if buffer:
            detected = detect_content_type("\n".join(buffer[-10:]))
            if detected != ContentType.THEORY:
                buffer_type = detected

    flush_buffer()
    return sections
