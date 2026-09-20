"""Retrieval, curriculum, and math verification endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from math_teacher.domain.models import RetrieveRequest, VerifyMathRequest
from math_teacher.embeddings.groq_provider import GroqEmbedder
from math_teacher.math.verifier import MathVerifier
from math_teacher.retrieval.service import analyze_query, retrieve
from math_teacher.storage.db import get_db

router = APIRouter()


@router.post("/retrieve", tags=["Retrieval"], summary="Debug retrieval — returns chunks without generation")
async def retrieve_chunks(
    body: RetrieveRequest,
    session: AsyncSession = Depends(get_db),
) -> dict:
    """POST /api/v1/retrieve

    Debug/evaluation endpoint. Returns retrieved and reranked chunks
    without generating an answer. Useful for evaluating retrieval quality.
    """
    embedder = GroqEmbedder()

    query_context = analyze_query(
        query=body.query,
        class_level=body.class_level,
        chapter=body.chapter,
    )

    chunks = await retrieve(
        query=body.query,
        query_context=query_context,
        embedder=embedder,
        session=session,
        top_k=body.top_k,
    )

    return {
        "query": body.query,
        "query_context": query_context.model_dump(),
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": str(c.chunk_id),
                "document": c.document_title,
                "chapter": c.chapter,
                "topic": c.topic,
                "page": c.page_number,
                "content_type": c.content_type,
                "score": c.score,
                "text_preview": c.text[:300],
            }
            for c in chunks
        ],
    }


@router.get(
    "/curriculum/classes/{class_level}/chapters",
    tags=["Curriculum"],
    summary="Return available chapters for a class level",
)
async def get_chapters(
    class_level: int,
    session: AsyncSession = Depends(get_db),
) -> dict:
    """GET /api/v1/curriculum/classes/{class_level}/chapters

    Returns the available curriculum scope (distinct chapters) for a given class level.
    """
    from sqlalchemy import select, distinct, text

    result = await session.execute(
        text(
            "SELECT DISTINCT chapter FROM document_chunks "
            "WHERE class_level = :cl AND chapter IS NOT NULL "
            "ORDER BY chapter"
        ),
        {"cl": class_level},
    )
    chapters = [row[0] for row in result.fetchall()]

    return {
        "class_level": class_level,
        "subject": "Mathematics",
        "chapter_count": len(chapters),
        "chapters": chapters,
    }


@router.post("/verify-math", tags=["Math"], summary="Deterministic math verification")
async def verify_math(body: VerifyMathRequest) -> dict:
    """POST /api/v1/verify-math

    Runs deterministic SymPy-based mathematical verification for supported
    problem formats. Does not require database access.
    """
    verifier = MathVerifier()
    result = verifier.verify_math_problem(
        problem=body.problem,
        proposed_solution=body.proposed_solution,
    )
    return {
        "passed": result.passed,
        "status": result.status,
        "reason": result.reason,
    }
