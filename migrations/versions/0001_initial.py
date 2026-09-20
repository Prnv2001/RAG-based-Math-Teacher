"""Initial migration: create all V1 tables with pgvector HNSW index."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from math_teacher.config.settings import settings

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # curriculum_nodes
    op.create_table(
        "curriculum_nodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("curriculum_nodes.id"), nullable=True),
        sa.Column("board", sa.String(100)),
        sa.Column("class_level", sa.Integer, nullable=False),
        sa.Column("subject", sa.String(100), nullable=False),
        sa.Column("node_type", sa.String(40), nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("code", sa.String(100)),
        sa.Column("description", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_curriculum_scope", "curriculum_nodes", ["board", "class_level", "subject"])

    # documents
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("source_type", sa.String(50), nullable=False),
        sa.Column("source_uri", sa.Text),
        sa.Column("license_info", sa.Text),
        sa.Column("checksum", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # document_chunks
    dims = settings.embedding_dimensions
    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("curriculum_node_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("curriculum_nodes.id"), nullable=True),
        sa.Column("class_level", sa.Integer, nullable=False),
        sa.Column("chapter", sa.Text),
        sa.Column("topic", sa.Text),
        sa.Column("concept", sa.Text),
        sa.Column("content_type", sa.String(40)),
        sa.Column("page_number", sa.Integer),
        sa.Column("chunk_index", sa.Integer),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(dims)),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_chunks_scope", "document_chunks", ["class_level", "chapter", "topic", "content_type"])
    # HNSW vector index for cosine similarity
    op.execute(
        f"CREATE INDEX idx_chunks_embedding ON document_chunks "
        f"USING hnsw (embedding vector_cosine_ops);"
    )

    # questions
    op.create_table(
        "questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("class_level", sa.Integer, nullable=False),
        sa.Column("chapter", sa.Text, nullable=False),
        sa.Column("topic", sa.Text),
        sa.Column("question_type", sa.String(40), nullable=False),
        sa.Column("difficulty", sa.String(20), nullable=False),
        sa.Column("marks", sa.Integer, nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column("solution", sa.Text, nullable=False),
        sa.Column("explanation", sa.Text),
        sa.Column("source_document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=True),
        sa.Column("source_page", sa.Integer),
        sa.Column("verified", sa.Boolean, server_default="false"),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_questions_scope", "questions", ["class_level", "chapter", "topic", "difficulty"])

    # students
    op.create_table(
        "students",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("class_level", sa.Integer, nullable=False),
        sa.Column("board", sa.String(100)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # student_mastery
    op.create_table(
        "student_mastery",
        sa.Column("student_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("students.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("curriculum_node_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("curriculum_nodes.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("mastery", sa.Numeric(5, 4), server_default="0.0"),
        sa.Column("confidence", sa.Numeric(5, 4), server_default="0.0"),
        sa.Column("attempts", sa.Integer, server_default="0"),
        sa.Column("correct", sa.Integer, server_default="0"),
        sa.Column("last_attempted_at", sa.DateTime(timezone=True)),
    )

    # student_attempts
    op.create_table(
        "student_attempts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("students.id", ondelete="CASCADE")),
        sa.Column("question_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("questions.id", ondelete="CASCADE")),
        sa.Column("answer", sa.Text),
        sa.Column("correct", sa.Boolean),
        sa.Column("score", sa.Numeric(6, 2)),
        sa.Column("time_seconds", sa.Integer),
        sa.Column("feedback", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # misconceptions
    op.create_table(
        "misconceptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("students.id", ondelete="CASCADE")),
        sa.Column("curriculum_node_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("curriculum_nodes.id"), nullable=True),
        sa.Column("label", sa.String(255)),
        sa.Column("evidence", sa.Text),
        sa.Column("severity", sa.String(50)),
        sa.Column("resolved", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("misconceptions")
    op.drop_table("student_attempts")
    op.drop_table("student_mastery")
    op.drop_table("students")
    op.drop_table("questions")
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding;")
    op.drop_index("idx_chunks_scope", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_index("idx_curriculum_scope", table_name="curriculum_nodes")
    op.drop_table("curriculum_nodes")
    op.execute("DROP EXTENSION IF EXISTS vector;")
