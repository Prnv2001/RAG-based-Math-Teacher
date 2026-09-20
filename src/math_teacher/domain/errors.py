"""Custom exception types for the AI Maths Teacher system."""

from __future__ import annotations


class MathTeacherError(Exception):
    """Base exception for all application errors."""


class ConfigurationError(MathTeacherError):
    """Raised when required configuration is missing or invalid."""


class DatabaseError(MathTeacherError):
    """Raised when a database operation fails."""


class IngestionError(MathTeacherError):
    """Raised when textbook ingestion fails."""


class RetrievalError(MathTeacherError):
    """Raised when retrieval fails."""


class LLMError(MathTeacherError):
    """Raised when an LLM call fails (timeout, bad response, etc.)."""


class EmbeddingError(MathTeacherError):
    """Raised when an embedding call fails."""


class GuardrailBlockedError(MathTeacherError):
    """Raised when input is blocked by a guardrail."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Blocked by guardrail: {reason}")


class GroundingFailedError(MathTeacherError):
    """Raised when answer fails grounding check after all retries are exhausted."""


class MathVerificationError(MathTeacherError):
    """Raised when math verification fails after retries are exhausted."""


class RetryBudgetExhaustedError(MathTeacherError):
    """Raised when the agent retry budget (max_agent_retries) is exhausted."""


class UnsupportedIntentError(MathTeacherError):
    """Raised when a V1-unsupported intent is requested."""

    def __init__(self, intent: str) -> None:
        self.intent = intent
        super().__init__(f"Intent '{intent}' is not implemented in V1.")


class AdminAuthError(MathTeacherError):
    """Raised when an admin endpoint is called without a valid admin API key."""
