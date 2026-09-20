"""Unit tests for deterministic Geometry SVG Builder Engine."""

from __future__ import annotations

from math_teacher.geometry.svg_engine import (
    build_intersecting_circles_svg,
    build_perpendicular_chords_svg,
    enforce_deterministic_svg_diagrams,
)
from math_teacher.utils.text_sanitizer import clean_math_text


def test_build_intersecting_circles_svg():
    svg = build_intersecting_circles_svg()
    assert "<svg" in svg
    assert "204" in svg  # Intersection B and C x-coordinate
    assert "102" in svg  # Top intersection B
    assert "158" in svg  # Bottom intersection C
    assert 'x1="70" y1="130" x2="280" y2="86"' in svg  # Line ABD through B


def test_build_perpendicular_chords_svg():
    svg = build_perpendicular_chords_svg()
    assert "<svg" in svg
    assert 'cx="160" cy="80"' in svg  # Off-center intersection P


def test_enforce_deterministic_svg_diagrams_interception():
    raw_llm_output = (
        "Question 6: Two circles intersect at points B and C. Through B, line segments ABD and PBQ are drawn.\n"
        "```xml\n<svg width='100'><circle cx='10'/></svg>\n```\n"
        "Prove that ∠ACP = ∠QCD."
    )
    sanitized = clean_math_text(raw_llm_output)
    # The hallucinated LLM <svg> block MUST be intercepted and replaced with exact NCERT SVG!
    assert "204" in sanitized
    assert "102" in sanitized
    assert "158" in sanitized
    assert "<circle cx=\"10\"/>" not in sanitized
