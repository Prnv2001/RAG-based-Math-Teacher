"""Main /ask endpoint — the primary teacher interface."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from math_teacher.agents.state import AgentState
from math_teacher.domain.enums import GuardAction, Intent
from math_teacher.domain.models import AskRequest, AskResponse, ErrorResponse
from math_teacher.embeddings.groq_provider import GroqEmbedder   # swap: was OllamaEmbedder
from math_teacher.llm.groq_provider import GroqLLM               # swap: was OllamaLLM
from math_teacher.orchestration.graph import build_graph
from math_teacher.storage.db import get_db

router = APIRouter()


@dataclass
class Deps:
    """Dependency container passed to all graph nodes."""

    llm: GroqLLM
    embedder: GroqEmbedder
    session: AsyncSession


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["Ask"],
    summary="Ask the AI Maths Teacher a question",
)
async def ask(
    body: AskRequest,
    session: AsyncSession = Depends(get_db),
) -> AskResponse:
    """POST /api/v1/ask

    Main teacher endpoint. Accepts a student question with optional curriculum
    context and returns a grounded answer with source citations.

    Pipeline:
        Input Guardrail → Query Analyzer → Router → Hybrid Retrieval →
        Context Builder → Tutor/Solver Agent → Grounding Check → Output Guardrail
    """
    request_id = str(uuid.uuid4())
    trace_id = str(uuid.uuid4())

    # Dynamic pipeline execution (caching bypassed to ensure fresh agent routing and live responses)

    # Build dependency container
    deps = Deps(
        llm=GroqLLM(),
        embedder=GroqEmbedder(),
        session=session,
    )

    # Build and run the LangGraph pipeline
    graph = build_graph(deps)

    initial_state: AgentState = {
        "question": body.question,
        "class_level": body.class_level,
        "chapter": body.chapter,
        "topic": body.topic,
        "mode": body.mode.value,
        "chat_history": [msg.model_dump() for msg in body.chat_history],
        "retry_count": 0,
        "request_id": request_id,
        "trace_id": trace_id,
        "error": None,
    }

    try:
        final_state: AgentState = await graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    # Handle blocked input
    input_guard = final_state.get("input_guardrail")
    if input_guard and not input_guard.passed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=input_guard.reason,
        )

    # Handle NOT_IMPLEMENTED intents
    intent_str = final_state.get("intent", "EXPLAIN_CONCEPT")
    if intent_str in (Intent.NOT_IMPLEMENTED.value, Intent.OUT_OF_SCOPE.value):
        return AskResponse(
            request_id=request_id,
            intent=intent_str,
            answer=(
                "This type of request is not yet available in V1 of the AI Maths Teacher. "
                "Please ask a conceptual or problem-solving question about Class 9 or Class 10 Mathematics."
            ),
            sources=[],
            grounding={"passed": False, "score": 0.0},
            math_verification={"passed": False, "status": "not_required"},
            trace_id=trace_id,
        )

    # Build response
    grounding = final_state.get("grounding")
    math_result = final_state.get("math_result")

    response = AskResponse(
        request_id=request_id,
        intent=intent_str,
        answer=final_state.get("answer", ""),
        sources=final_state.get("sources", []),
        grounding={
            "passed": grounding.grounded if grounding else False,
            "score": grounding.score if grounding else 0.0,
        },
        math_verification={
            "passed": math_result.passed if math_result else True,
            "status": math_result.status if math_result else "not_required",
        },
        step_logs=final_state.get("step_logs", []),
        trace_id=trace_id,
    )

    return response
