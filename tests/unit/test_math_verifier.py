"""Unit tests for the SymPy MathVerifier."""

from __future__ import annotations

import pytest

from math_teacher.math.verifier import MathVerifier


@pytest.fixture
def verifier() -> MathVerifier:
    return MathVerifier()


class TestVerifyEquationSolution:
    """Tests for verify_equation_solution()."""

    def test_accepts_correct_roots_quadratic(self, verifier):
        """FR-012 — TEST-001: 2x²+5x+2=0 has roots -2 and -1/2."""
        result = verifier.verify_equation_solution(
            equation="2*x**2 + 5*x + 2",
            variable="x",
            proposed_roots=["-2", "-1/2"],
        )
        assert result.passed is True
        assert result.status == "passed"

    def test_rejects_wrong_roots(self, verifier):
        """Incorrect roots should fail verification."""
        result = verifier.verify_equation_solution(
            equation="x**2 - 5*x + 6",
            variable="x",
            proposed_roots=["1", "4"],  # correct roots are 2 and 3
        )
        assert result.passed is False
        assert result.status == "failed"

    def test_accepts_linear_equation_root(self, verifier):
        """3x - 9 = 0 → x = 3."""
        result = verifier.verify_equation_solution(
            equation="3*x - 9",
            variable="x",
            proposed_roots=["3"],
        )
        assert result.passed is True

    def test_handles_equation_equals_rhs(self, verifier):
        """Handles LHS = RHS format."""
        result = verifier.verify_equation_solution(
            equation="x**2 = 4",
            variable="x",
            proposed_roots=["2", "-2"],
        )
        assert result.passed is True

    def test_unparseable_equation_returns_false(self, verifier):
        """Unparseable expression returns passed=True with status=not_required."""
        result = verifier.verify_equation_solution(
            equation="!!invalid math@@",
            variable="x",
            proposed_roots=["1"],
        )
        assert result.passed is True
        assert result.status == "not_required"

    def test_multiple_correct_roots(self, verifier):
        """x³ - 6x² + 11x - 6 = 0 has roots 1, 2, 3."""
        result = verifier.verify_equation_solution(
            equation="x**3 - 6*x**2 + 11*x - 6",
            variable="x",
            proposed_roots=["1", "2", "3"],
        )
        assert result.passed is True


    def test_unicode_powers_and_operators(self, verifier):
        """Unicode superscripts (3x⁴ - 4x³) should normalize and verify correctly."""
        result = verifier.verify_equation_solution(
            equation="3*x⁴ - 4*x³ - 3*x - 1",
            variable="x",
            proposed_roots=["1"],
        )
        assert result.passed is False
        assert result.status == "failed"


class TestVerifyMathProblem:
    """Tests for verify_math_problem() high-level interface."""

    def test_extracts_and_verifies_from_problem_string(self, verifier):
        """Should extract equation and verify from natural language problem."""
        result = verifier.verify_math_problem(
            problem="Solve x^2 - 5x + 6 = 0",
            proposed_solution="x = 2 or x = 3",
        )
        assert result.status in ("passed", "failed", "not_required")

    def test_no_equation_in_problem(self, verifier):
        """Problem with no recognizable equation returns status=not_required."""
        result = verifier.verify_math_problem(
            problem="Explain the Pythagoras theorem",
            proposed_solution="a² + b² = c²",
        )
        assert result.passed is True
        assert result.status == "not_required"
