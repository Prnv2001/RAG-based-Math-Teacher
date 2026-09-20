"""Domain enumerations used across the entire system."""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    """Classified intent of a student query."""

    EXPLAIN_CONCEPT = "EXPLAIN_CONCEPT"
    SOLVE_PROBLEM = "SOLVE_PROBLEM"
    CHECK_ANSWER = "CHECK_ANSWER"
    GIVE_HINT = "GIVE_HINT"
    # Future phases — V1 returns NOT_IMPLEMENTED for these
    PRACTICE = "PRACTICE"
    START_EXAM = "START_EXAM"
    START_VIVA = "START_VIVA"
    REVISE_TOPIC = "REVISE_TOPIC"
    ANALYZE_PERFORMANCE = "ANALYZE_PERFORMANCE"
    # Meta
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class Difficulty(str, Enum):
    """Problem/concept difficulty level."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ContentType(str, Enum):
    """Type of content within a document chunk."""

    THEORY = "theory"
    DEFINITION = "definition"
    FORMULA = "formula"
    THEOREM = "theorem"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    SOLUTION = "solution"
    SUMMARY = "summary"


class GuardAction(str, Enum):
    """Action taken by a guardrail."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    RETRY = "RETRY"
    REFORMULATE = "REFORMULATE"
    ESCALATE = "ESCALATE"


class SolverMode(str, Enum):
    """Mode for tutor / solver responses."""

    INTERACTIVE = "INTERACTIVE"
    HINT = "HINT"
    GUIDED = "GUIDED"
    FULL_SOLUTION = "FULL_SOLUTION"



class NodeType(str, Enum):
    """Curriculum hierarchy node type."""

    BOARD = "board"
    CLASS = "class"
    SUBJECT = "subject"
    CHAPTER = "chapter"
    TOPIC = "topic"
    CONCEPT = "concept"


class DocumentState(str, Enum):
    """Lifecycle state of an ingested document."""

    INGESTED = "INGESTED"
    PARSED = "PARSED"
    INDEXED = "INDEXED"
    ACTIVE = "ACTIVE"


class AnswerState(str, Enum):
    """Lifecycle state of a generated answer."""

    GENERATED = "GENERATED"
    VERIFYING = "VERIFYING"
    ACCEPTED = "ACCEPTED"
    RETRY = "RETRY"
    REJECTED = "REJECTED"
