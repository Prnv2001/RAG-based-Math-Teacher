"""Conditional routing functions for the LangGraph graph."""

from __future__ import annotations

from langgraph.graph import END

from math_teacher.agents.state import AgentState
from math_teacher.domain.enums import GuardAction, Intent


def route_after_input_guardrail(state: AgentState) -> str:
    """Route after input guardrail: ALLOW → query_analyzer, else → END."""
    result = state.get("input_guardrail")
    if result and result.passed:
        return "query_analyzer"
    return END


def route_after_router(state: AgentState) -> str:
    """Route after intent classification."""
    intent_str = state.get("intent", "OUT_OF_SCOPE")
    try:
        intent = Intent(intent_str)
    except ValueError:
        intent = Intent.OUT_OF_SCOPE

    if intent in (Intent.START_EXAM, Intent.PRACTICE):
        return "retrieve_for_exam"
    if intent in (
        Intent.EXPLAIN_CONCEPT,
        Intent.GIVE_HINT,
        Intent.CHECK_ANSWER,
        Intent.REVISE_TOPIC,
        Intent.START_VIVA,
    ):
        return "retrieve_for_tutor"
    if intent == Intent.SOLVE_PROBLEM:
        return "retrieve_for_solver"
    # NOT_IMPLEMENTED, OUT_OF_SCOPE → END (set a polite message)
    return END


def route_after_output_guardrail(state: AgentState) -> str:
    """Route after output guardrail: ALLOW → END, RETRY → tutor/solver, BLOCK → END."""
    result = state.get("output_guardrail")
    if not result or result.action == GuardAction.ALLOW:
        return END
    if result.action == GuardAction.RETRY:
        intent_str = state.get("intent", "EXPLAIN_CONCEPT")
        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.EXPLAIN_CONCEPT
        if intent == Intent.SOLVE_PROBLEM:
            return "solver"
        if intent in (Intent.START_EXAM, Intent.PRACTICE):
            return "exam"
        return "tutor"
    # BLOCK
    return END
