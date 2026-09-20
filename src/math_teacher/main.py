"""FastAPI application factory — the sole public API boundary."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from math_teacher.api.routes_ask import router as ask_router
from math_teacher.api.routes_health import router as health_router
from math_teacher.api.routes_ingestion import router as ingestion_router
from math_teacher.api.routes_retrieval import router as retrieval_router
from math_teacher.domain.errors import (
    AdminAuthError,
    GuardrailBlockedError,
    MathTeacherError,
)
from math_teacher.storage.db import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Startup: verify DB connection
    yield
    # Shutdown: close DB pool
    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Maths Teacher",
        description=(
            "Curriculum-grounded RAG backend for Class 9–10 Mathematics. "
            "Powered by DeepSeek-R1 via Ollama + pgvector hybrid retrieval."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS (permissive for local dev — tighten for production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    api_prefix = "/api/v1"
    app.include_router(health_router, prefix=api_prefix)
    app.include_router(ask_router, prefix=api_prefix)
    app.include_router(ingestion_router, prefix=api_prefix)
    app.include_router(retrieval_router, prefix=api_prefix)

    # Serve chat UI and embeddable widget from /static
    _project_root = Path(__file__).resolve().parents[2]
    app.mount("/static", StaticFiles(directory=str(_project_root)), name="static")

    # ------------------------------------------------------------------
    # Global exception handlers
    # ------------------------------------------------------------------
    @app.exception_handler(GuardrailBlockedError)
    async def guardrail_handler(request: Request, exc: GuardrailBlockedError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error_code": "GUARDRAIL_BLOCKED", "message": exc.reason},
        )

    @app.exception_handler(AdminAuthError)
    async def auth_handler(request: Request, exc: AdminAuthError):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error_code": "UNAUTHORIZED", "message": str(exc)},
        )

    @app.exception_handler(MathTeacherError)
    async def app_error_handler(request: Request, exc: MathTeacherError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error_code": "SERVICE_ERROR", "message": str(exc)},
        )

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "AI Maths Teacher API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()
