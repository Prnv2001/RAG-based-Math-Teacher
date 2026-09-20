"""Guardrail tests — 10 scenarios from spec Section 24."""

from __future__ import annotations

import pytest

from math_teacher.domain.enums import GuardAction
from math_teacher.guardrails.input import CurriculumGuardrail
from math_teacher.guardrails.output import OutputGuardrail
from math_teacher.domain.models import GroundingResult, MathVerificationResult


# ---------------------------------------------------------------------------
# Curriculum Guardrail Tests
# ---------------------------------------------------------------------------

class TestCurriculumGuardrail:
    @pytest.fixture
    def guardrail(self) -> CurriculumGuardrail:
        return CurriculumGuardrail()

    def test_class_10_allowed(self, guardrail):
        """TEST-005: Class 10 should pass curriculum guardrail."""
        result = guardrail.check("What is the discriminant?", class_level=10)
        assert result.passed is True
        assert result.action == GuardAction.ALLOW

    def test_class_9_allowed(self, guardrail):
        """Class 9 is within supported scope."""
        result = guardrail.check("Explain polynomials", class_level=9)
        assert result.passed is True

    def test_class_8_blocked(self, guardrail):
        """TEST-005: Class 8 is out of scope → BLOCK."""
        result = guardrail.check("Explain fractions", class_level=8)
        assert result.passed is False
        assert result.action == GuardAction.BLOCK

    def test_class_11_blocked(self, guardrail):
        """Class 11 is out of scope → BLOCK."""
        result = guardrail.check("Explain calculus", class_level=11)
        assert result.passed is False
        assert result.action == GuardAction.BLOCK

    def test_no_class_level_allowed(self, guardrail):
        """No class level → ALLOW (Query Analyzer will infer)."""
        result = guardrail.check("What is a polynomial?", class_level=None)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Output Guardrail Tests
# ---------------------------------------------------------------------------

class TestOutputGuardrail:
    @pytest.fixture
    def guardrail(self) -> OutputGuardrail:
        return OutputGuardrail(max_retries=2)

    def _good_grounding(self) -> GroundingResult:
        return GroundingResult(grounded=True, score=0.92, supported_claims=["claim1"])

    def _bad_grounding(self) -> GroundingResult:
        return GroundingResult(grounded=False, score=0.40, unsupported_claims=["bad claim"])

    def _passed_math(self) -> MathVerificationResult:
        return MathVerificationResult(passed=True, reason="Verified", status="passed")

    def _failed_math(self) -> MathVerificationResult:
        return MathVerificationResult(passed=False, reason="Root mismatch", status="failed")

    def test_good_answer_allowed(self, guardrail):
        """Good grounding → ALLOW."""
        result = guardrail.check(grounding=self._good_grounding())
        assert result.passed is True
        assert result.action == GuardAction.ALLOW

    def test_ungrounded_triggers_retry(self, guardrail):
        """TEST-004: Ungrounded answer → RETRY (first attempt)."""
        result = guardrail.check(grounding=self._bad_grounding())
        assert result.passed is False
        assert result.action == GuardAction.RETRY
        assert guardrail.retry_count == 1

    def test_retry_budget_exhausted_blocks(self, guardrail):
        """TEST-004: After max retries → BLOCK with safe fallback."""
        guardrail.retry_count = 2  # Already exhausted
        result = guardrail.check(grounding=self._bad_grounding())
        assert result.passed is False
        assert result.action == GuardAction.BLOCK

    def test_failed_math_triggers_retry(self, guardrail):
        """Failed math verification → RETRY."""
        result = guardrail.check(
            grounding=self._good_grounding(),
            math_result=self._failed_math(),
            require_math=True,
        )
        assert result.action == GuardAction.RETRY

    def test_math_not_required_ignored(self, guardrail):
        """Math failure ignored when require_math=False (tutor path)."""
        result = guardrail.check(
            grounding=self._good_grounding(),
            math_result=self._failed_math(),
            require_math=False,
        )
        assert result.action == GuardAction.ALLOW

    def test_safe_fallback_message(self):
        """Safe fallback is a non-empty string."""
        msg = OutputGuardrail.safe_fallback_answer()
        assert isinstance(msg, str)
        assert len(msg) > 20
