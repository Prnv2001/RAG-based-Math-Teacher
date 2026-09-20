"""SQLAlchemy ORM models — system of record for all domains."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from math_teacher.config.settings import settings
from math_teacher.storage.db import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


class CurriculumNode(Base):
    """Hierarchical curriculum node: Board → Class → Subject → Chapter → Topic → Concept."""

    __tablename__ = "curriculum_nodes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("curriculum_nodes.id"), nullable=True
    )
    board: Mapped[str | None] = mapped_column(String(100))
    class_level: Mapped[int] = mapped_column(Integer, nullable=False)
    subject: Mapped[str] = mapped_column(String(100), nullable=False)
    node_type: Mapped[str] = mapped_column(String(40), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    code: Mapped[str | None] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Self-referential relationship
    children: Mapped[list[CurriculumNode]] = relationship("CurriculumNode", back_populates="parent")
    parent: Mapped[CurriculumNode | None] = relationship(
        "CurriculumNode", back_populates="children", remote_side="CurriculumNode.id"
    )

    __table_args__ = (
        Index("idx_curriculum_scope", "board", "class_level", "subject"),
    )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class Document(Base):
    """An ingested source document (e.g., a textbook PDF)."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    license_info: Mapped[str | None] = mapped_column(Text)
    checksum: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list[DocumentChunk]] = relationship("DocumentChunk", back_populates="document")


# ---------------------------------------------------------------------------
# Document Chunks (with embeddings)
# ---------------------------------------------------------------------------


class DocumentChunk(Base):
    """A semantic chunk of a document, with curriculum metadata and embedding vector."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )
    curriculum_node_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("curriculum_nodes.id"), nullable=True
    )
    class_level: Mapped[int] = mapped_column(Integer, nullable=False)
    chapter: Mapped[str | None] = mapped_column(Text)
    topic: Mapped[str | None] = mapped_column(Text)
    concept: Mapped[str | None] = mapped_column(Text)
    content_type: Mapped[str | None] = mapped_column(String(40))
    page_number: Mapped[int | None] = mapped_column(Integer)
    chunk_index: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Vector dimension matches settings.embedding_dimensions (768 for nomic-embed-text)
    embedding: Mapped[Any] = mapped_column(Vector(settings.embedding_dimensions), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_scope", "class_level", "chapter", "topic", "content_type"),
        # HNSW index created via Alembic migration (requires raw DDL)
    )


# ---------------------------------------------------------------------------
# Questions (future phase — table created now for schema completeness)
# ---------------------------------------------------------------------------


class Question(Base):
    """A verified curriculum-aligned question in the question bank."""

    __tablename__ = "questions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    class_level: Mapped[int] = mapped_column(Integer, nullable=False)
    chapter: Mapped[str] = mapped_column(Text, nullable=False)
    topic: Mapped[str | None] = mapped_column(Text)
    question_type: Mapped[str] = mapped_column(String(40), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(20), nullable=False)
    marks: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    solution: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text)
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True
    )
    source_page: Mapped[int | None] = mapped_column(Integer)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_questions_scope", "class_level", "chapter", "topic", "difficulty"),
    )


# ---------------------------------------------------------------------------
# Students (future phase)
# ---------------------------------------------------------------------------


class Student(Base):
    """A student user."""

    __tablename__ = "students"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    class_level: Mapped[int] = mapped_column(Integer, nullable=False)
    board: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    mastery_records: Mapped[list[StudentMastery]] = relationship(
        "StudentMastery", back_populates="student"
    )


# ---------------------------------------------------------------------------
# Student Mastery (future phase)
# ---------------------------------------------------------------------------


class StudentMastery(Base):
    """Per-concept mastery score for a student."""

    __tablename__ = "student_mastery"

    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"), primary_key=True
    )
    curriculum_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("curriculum_nodes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    mastery: Mapped[float] = mapped_column(Numeric(5, 4), default=0.0)
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    last_attempted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    student: Mapped[Student] = relationship("Student", back_populates="mastery_records")


# ---------------------------------------------------------------------------
# Student Attempts (future phase)
# ---------------------------------------------------------------------------


class StudentAttempt(Base):
    """A student's attempt at a question."""

    __tablename__ = "student_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE")
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("questions.id", ondelete="CASCADE")
    )
    answer: Mapped[str | None] = mapped_column(Text)
    correct: Mapped[bool | None] = mapped_column(Boolean)
    score: Mapped[float | None] = mapped_column(Numeric(6, 2))
    time_seconds: Mapped[int | None] = mapped_column(Integer)
    feedback: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Misconceptions (future phase)
# ---------------------------------------------------------------------------


class Misconception(Base):
    """A detected student misconception."""

    __tablename__ = "misconceptions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE")
    )
    curriculum_node_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("curriculum_nodes.id"), nullable=True
    )
    label: Mapped[str | None] = mapped_column(String(255))
    evidence: Mapped[str | None] = mapped_column(Text)
    severity: Mapped[str | None] = mapped_column(String(50))
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
