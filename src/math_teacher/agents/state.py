"""LangGraph AgentState — shared state passed between graph nodes."""

from __future__ import annotations

from typing import Any, TypedDict

from math_teacher.domain.models import (
    GroundingResult,
    GuardrailResult,
    MathVerificationResult,
    QueryContext,
    RetrievedChunk,
    SourceReference,
)


class AgentState(TypedDict, total=False):
    """Shared state for the LangGraph orchestration graph."""

    # ---- Input ----
    question: str
    class_level: int | None
    chapter: str | None
    topic: str | None
    mode: str  # HINT / GUIDED / FULL_SOLUTION
    chat_history: list[dict[str, str]]

    # ---- Processing ----
    query_context: QueryContext
    intent: str
    intent_confidence: float
    retrieved_chunks: list[RetrievedChunk]
    context_text: str
    sources: list[SourceReference]

    # ---- Output ----
    answer: str
    solver_output: dict[str, Any]

    # ---- Verification ----
    grounding: GroundingResult
    math_result: MathVerificationResult
    require_math: bool

    # ---- Guardrails ----
    input_guardrail: GuardrailResult
    output_guardrail: GuardrailResult
    retry_count: int

    # ---- Trace ----
    request_id: str
    trace_id: str
    step_logs: list[str]
    error: str | None
