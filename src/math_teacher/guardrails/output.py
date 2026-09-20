"""Output guardrail — enforces grounding + math verification with bounded retry."""

from __future__ import annotations

from math_teacher.config.settings import settings
from math_teacher.domain.enums import GuardAction
from math_teacher.domain.models import GroundingResult, GuardrailResult, MathVerificationResult


class OutputGuardrail:
    """Final check before returning a response to the client.

    Evaluates grounding score and math verification.
    Manages bounded retry state (max_agent_retries).
    """

    def __init__(self, max_retries: int | None = None) -> None:
        self.max_retries = max_retries or settings.max_agent_retries
        self.retry_count = 0

    def reset(self) -> None:
        """Reset retry counter for a new request."""
        self.retry_count = 0

    def check(
        self,
        grounding: GroundingResult,
        math_result: MathVerificationResult | None = None,
        require_math: bool = False,
    ) -> GuardrailResult:
        """Evaluate output quality and decide ALLOW / RETRY / BLOCK.

        Args:
            grounding: Result from GroundingChecker.
            math_result: Result from MathVerifier (None if not required).
            require_math: Whether math verification is required for this response.

        Returns:
            GuardrailResult with action ALLOW, RETRY, or BLOCK.
        """
        failures = []

        # Check grounding
        if not grounding.grounded:
            failures.append(
                f"Grounding failed (score={grounding.score:.2f}, "
                f"threshold={settings.grounding_threshold})"
            )

        # Check math verification (only when required and explicitly failed)
        if require_math and math_result and not math_result.passed:
            if math_result.status == "failed":
                failures.append(f"Math verification failed: {math_result.reason}")

        if not failures:
            self.retry_count = 0
            return GuardrailResult(
                passed=True,
                action=GuardAction.ALLOW,
                reason="Grounding and math verification passed.",
                score=grounding.score,
            )

        # Something failed — decide retry or block
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            return GuardrailResult(
                passed=False,
                action=GuardAction.RETRY,
                reason=f"Output quality check failed (attempt {self.retry_count}/{self.max_retries}): "
                + "; ".join(failures),
                score=grounding.score,
            )

        # Retry budget exhausted — block with safe fallback
        return GuardrailResult(
            passed=False,
            action=GuardAction.BLOCK,
            reason=(
                f"Output quality check failed after {self.max_retries} retries. "
                "Returning safe fallback to avoid fabricating an answer. "
                + "; ".join(failures)
            ),
            score=grounding.score,
        )

    @staticmethod
    def safe_fallback_answer() -> str:
        """Return a safe fallback message when all retries are exhausted."""
        return (
            "I was unable to generate a sufficiently grounded answer for your question "
            "using the available curriculum material. Please try rephrasing your question, "
            "or ask your teacher for clarification on this topic."
        )
