"""Unit tests for the clean_math_text sanitizer module."""

from __future__ import annotations

import pytest

from math_teacher.utils.text_sanitizer import clean_math_text


class TestCleanMathText:
    """Tests for clean_math_text()."""

    def test_converts_frac_and_powers(self):
        """\frac{3x^4}{x} should convert to (3x⁴/x)."""
        raw = r"\frac{3x^4}{x} = 3x^3"
        cleaned = clean_math_text(raw)
        assert "(3x⁴/x) = 3x³" in cleaned
        assert r"\frac" not in cleaned

    def test_negative_fractions(self):
        """Negative fractions -\frac{1}{2} and -\frac12 must keep negative sign."""
        assert clean_math_text(r"-\frac{1}{2}") == "-(1/2)"
        assert clean_math_text(r"-\dfrac12") == "-(1/2)"

    def test_positive_frac_with_hyphen_in_sentence(self):
        """\frac12 in a sentence with minus sign must NOT get false negative sign."""
        raw = r"Solve 2x - 1 = 0 to get \frac12"
        cleaned = clean_math_text(raw)
        assert "-(1/2)" not in cleaned
        assert "(1/2)" in cleaned

    def test_complex_polynomial_fraction(self):
        """Complex fraction \frac{3x^4 - 4x^3}{x-1} should format cleanly."""
        raw = r"\frac{3x^4 - 4x^3 - 3x - 1}{x - 1}"
        cleaned = clean_math_text(raw)
        assert "((3x⁴ - 4x³ - 3x - 1)/(x - 1))" in cleaned or "(3x⁴ - 4x³ - 3x - 1)/(x - 1)" in cleaned
        assert r"\frac" not in cleaned

    def test_radicals(self):
        """\sqrt{b^2 - 4ac} and \sqrt[3]{x} should convert to unicode roots."""
        assert clean_math_text(r"\sqrt{b^2 - 4ac}") == "√(b² - 4ac)"
        assert clean_math_text(r"\sqrt[3]{x}") == "∛(x)"

    def test_greek_letters_and_operators(self):
        """Greek letters, operators, and relations should convert to unicode."""
        raw = r"\alpha \cdot \beta + \pi \le \sqrt{x} \pm \frac{1}{2} \neq 0"
        cleaned = clean_math_text(raw)
        assert "α · β + π ≤ √(x) ± (1/2) ≠ 0" in cleaned

    def test_subscripts_and_superscripts(self):
        """Subscripts and superscripts should convert cleanly."""
        raw = r"x_1^2 + x_2^2 = x_0^{12}"
        cleaned = clean_math_text(raw)
        assert "x₁² + x₂² = x₀¹²" in cleaned

    def test_strips_dollar_signs(self):
        """$x$ and $2x+1=0$ should strip dollar delimiters."""
        raw = "Solve for $x$: $2x + 1 = 0$"
        cleaned = clean_math_text(raw)
        assert "$" not in cleaned
        assert "Solve for x: 2x + 1 = 0" in cleaned

    def test_cleans_alignment_spaces_and_environments(self):
        """&=, \; and \begin{align} should be cleaned up."""
        raw = r"\begin{align} 2x+1=0 \; &= -1 \\ x &= -\frac{1}{2} \end{align}"
        cleaned = clean_math_text(raw)
        assert r"\;" not in cleaned
        assert "&=" not in cleaned
        assert r"\begin" not in cleaned
        assert "2x+1=0 = -1" in cleaned
        assert "x = -(1/2)" in cleaned

    def test_geometry_formatting(self):
        """\cong, angle ABC, triangle ABC, {rm cm}, and 60⁽circ) should format cleanly."""
        raw = r"AB = DE = 5 {rm cm}, angle ABC = angle DEF = 60⁽circ), triangle ABC cong triangle DEF"
        cleaned = clean_math_text(raw)
        assert "AB = DE = 5 cm" in cleaned
        assert "∠ABC = ∠DEF = 60°" in cleaned
        assert "△ABC ≅ △DEF" in cleaned

    def test_perpendicular_bisector_cleanup(self):
        """\perp endicular, ⊥ endicular, and \perp should be cleanly formatted."""
        raw1 = r"Show that the line PQ is the \perp endicular bisector of AB."
        raw2 = "Show that the line PQ is the ⊥ endicular bisector of AB."
        raw3 = r"Line PQ \perp AB."
        raw4 = r"Line PQ is \perpendicular to AB."
        assert clean_math_text(raw1) == "Show that the line PQ is the perpendicular bisector of AB."
        assert clean_math_text(raw2) == "Show that the line PQ is the perpendicular bisector of AB."
        assert clean_math_text(raw3) == "Line PQ ⊥ AB."
        assert clean_math_text(raw4) == "Line PQ is perpendicular to AB."

    def test_circle_word_not_corrupted(self):
        """The word 'circle', 'circumference', or 'circular' must NOT be converted to '°le'."""
        raw = r"AB is a chord of a circle with center O. The angle is 40\circ."
        cleaned = clean_math_text(raw)
        assert "a circle with center O" in cleaned
        assert "°le" not in cleaned
        assert "40°" in cleaned

    def test_malformed_fraction_sanitization(self):
        """Malformed fractions like {22{7} or {22/7} must sanitize to (22/7)."""
        raw = "Take π = {22{7}"
        cleaned = clean_math_text(raw)
        assert "22/7" in cleaned
        assert "{22{7}" not in cleaned

    def test_fractional_words_not_corrupted(self):
        """English words containing 'frac' (fractional, fraction, refraction) must not be corrupted by stripping 'frac'."""
        raw = "A polynomial cannot have fractional powers or a variable fraction. All terms must have whole number exponents."
        cleaned = clean_math_text(raw)
        assert cleaned == raw





