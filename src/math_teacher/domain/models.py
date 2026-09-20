"""Core domain Pydantic models (runtime / transient schemas)."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from math_teacher.domain.enums import (
    ContentType,
    Difficulty,
    GuardAction,
    Intent,
    SolverMode,
)


# ---------------------------------------------------------------------------
# Query Understanding
# ---------------------------------------------------------------------------


class QueryContext(BaseModel):
    """Structured representation of a student's question extracted by the Query Analyzer."""

    original_query: str
    class_level: int | None = Field(None, ge=1, le=12)
    subject: str = "Mathematics"
    chapter: str | None = None
    chapters: list[str] = Field(default_factory=list)
    topic: str | None = None
    concept: str | None = None
    intent: Intent | None = None
    difficulty: Difficulty | None = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A single document chunk returned from retrieval."""

    chunk_id: UUID
    document_id: UUID
    document_title: str
    class_level: int
    chapter: str | None = None
    topic: str | None = None
    concept: str | None = None
    content_type: ContentType | None = None
    page_number: int | None = None
    text: str
    score: float  # retrieval / rerank score


class SourceReference(BaseModel):
    """Source citation included in the API response."""

    document: str
    chapter: str | None = None
    topic: str | None = None
    page: int | None = None


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class GuardrailResult(BaseModel):
    """Result from any guardrail check."""

    passed: bool
    action: GuardAction
    reason: str
    score: float | None = None


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


class GroundingResult(BaseModel):
    """Result from the grounding checker."""

    grounded: bool
    score: float = Field(ge=0.0, le=1.0)
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Math Verification
# ---------------------------------------------------------------------------


class MathVerificationResult(BaseModel):
    """Result from the deterministic SymPy math verifier."""

    passed: bool
    reason: str
    status: str = "not_required"  # "passed" | "failed" | "not_required" | "unparseable"


# ---------------------------------------------------------------------------
# Agent / Router
# ---------------------------------------------------------------------------


class RouterResult(BaseModel):
    """Output from the Router agent."""

    intent: Intent
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: Any) -> float:
        if v is None:
            return 0.5
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.5


class SolverOutput(BaseModel):
    """Structured output from the Solver agent."""

    approach: str = ""
    steps: list[str] = Field(default_factory=list)
    final_answer: str = ""
    equations_to_verify: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("final_answer", "approach", mode="before")
    @classmethod
    def _coerce_to_str(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            items = [f"**{k}**: {val}" for k, val in v.items()]
            return "\n".join(items)
        if isinstance(v, list):
            return "\n".join(str(x) for x in v)
        return str(v)

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce_steps(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, dict):
                    result.append(json.dumps(item))
                else:
                    result.append(str(item))
            return result
        return [str(v)]

    @field_validator("equations_to_verify", mode="before")
    @classmethod
    def _coerce_eqs(cls, v: Any) -> list[dict[str, Any]]:
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        if isinstance(v, list):
            res = []
            for item in v:
                if isinstance(item, dict):
                    res.append(item)
            return res
        return []


# ---------------------------------------------------------------------------
# API Request / Response schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single turn in the chat history."""

    role: str = Field(..., description="Role of the speaker: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class AskRequest(BaseModel):
    """Request body for POST /api/v1/ask."""

    question: str = Field(..., min_length=1)
    class_level: int | None = Field(None, ge=1, le=12)
    chapter: str | None = None
    topic: str | None = None
    mode: SolverMode = SolverMode.FULL_SOLUTION
    chat_history: list[ChatMessage] = Field(default_factory=list)


class AskResponse(BaseModel):
    """Response body for POST /api/v1/ask."""

    request_id: str
    intent: str
    answer: str
    sources: list[SourceReference]
    grounding: dict[str, Any]
    math_verification: dict[str, Any]
    step_logs: list[str] = Field(default_factory=list)
    trace_id: str | None = None


class RetrieveRequest(BaseModel):
    """Request body for POST /api/v1/retrieve."""

    query: str = Field(..., min_length=1)
    class_level: int | None = Field(None, ge=1, le=12)
    chapter: str | None = None
    top_k: int | None = None


class VerifyMathRequest(BaseModel):
    """Request body for POST /api/v1/verify-math."""

    problem: str
    proposed_solution: str


class IngestRequest(BaseModel):
    """Request body for POST /api/v1/ingest (file upload handled separately)."""

    title: str
    class_level: int = Field(..., ge=1, le=12)
    board: str = "CBSE"
    subject: str = "Mathematics"
    license_info: str | None = None


class HealthResponse(BaseModel):
    """Response body for GET /api/v1/health."""

    status: str
    database: str
    llm: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error_code: str
    message: str
    request_id: str | None = None
