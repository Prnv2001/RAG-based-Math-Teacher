"""Ingestion endpoint — admin-only textbook PDF ingestion."""

from __future__ import annotations

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, status
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from math_teacher.config.settings import settings
from math_teacher.domain.errors import IngestionError
from math_teacher.embeddings.groq_provider import GroqEmbedder
from math_teacher.ingestion.pipeline import ingest_pdf
from math_teacher.storage.db import get_db

import shutil
import tempfile
from pathlib import Path

router = APIRouter()


def _verify_admin_key(x_admin_key: str | None = Header(default=None)) -> None:
    """Dependency: verify admin API key from X-Admin-Key header."""
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key header.",
        )


@router.post(
    "/ingest",
    tags=["Ingestion"],
    dependencies=[Depends(_verify_admin_key)],
    summary="Ingest an authorized textbook PDF (admin only)",
)
async def ingest_textbook(
    file: UploadFile,
    title: str,
    class_level: int,
    board: str = "CBSE",
    subject: str = "Mathematics",
    license_info: str | None = None,
    session: AsyncSession = Depends(get_db),
) -> dict:
    """POST /api/v1/ingest

    Admin/development endpoint. Ingests an authorized textbook PDF through the
    full pipeline: parse → structure → chunk → tag → embed → store.

    Protected by X-Admin-Key header.
    """
    if class_level not in (9, 10):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="class_level must be 9 or 10 for V1.",
        )

    # Save upload to a temp file
    suffix = Path(file.filename or "upload.pdf").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        embedder = GroqEmbedder()
        result = await ingest_pdf(
            pdf_path=tmp_path,
            title=title,
            class_level=class_level,
            board=board,
            subject=subject,
            license_info=license_info,
            embedder=embedder,
            session=session,
        )
        return {"status": "success", **result}
    except IngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )
    finally:
        tmp_path.unlink(missing_ok=True)
