"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from math_teacher.domain.models import HealthResponse
from math_teacher.llm.groq_provider import check_groq_health
from math_teacher.storage.db import check_db_health

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """GET /api/v1/health — service and dependency health status."""
    db_ok = await check_db_health()
    llm_ok = await check_groq_health()

    return HealthResponse(
        status="ok" if (db_ok and llm_ok) else "degraded",
        database="ok" if db_ok else "unavailable",
        llm="ok" if llm_ok else "unavailable",
        version="1.0.0",
    )
