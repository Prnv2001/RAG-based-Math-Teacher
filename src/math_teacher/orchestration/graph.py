"""LangGraph orchestration graph — wires all nodes with conditional routing."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from math_teacher.agents.state import AgentState
from math_teacher.orchestration.nodes import (
    exam_node,
    grounding_check_node,
    input_guardrail_node,
    output_guardrail_node,
    query_analyzer_node,
    retrieve_node,
    router_node,
    solver_node,
    tutor_node,
)
from math_teacher.orchestration.routing import (
    route_after_input_guardrail,
    route_after_output_guardrail,
    route_after_router,
)


def build_graph(deps) -> StateGraph:
    """Build and compile the LangGraph orchestration graph.

    Args:
        deps: Dependency container with llm, embedder, session, etc.
    Returns:
        Compiled StateGraph ready to invoke.
    """
    graph = StateGraph(AgentState)

    # ------------------------------------------------------------------
    # Async wrappers — Python has no async lambdas, so we define them here
    # ------------------------------------------------------------------
    async def _input_guardrail(s): return await input_guardrail_node(s, deps)
    async def _query_analyzer(s): return await query_analyzer_node(s, deps)
    async def _router(s): return await router_node(s, deps)
    async def _retrieve(s): return await retrieve_node(s, deps)
    async def _tutor(s): return await tutor_node(s, deps)
    async def _exam(s): return await exam_node(s, deps)
    async def _solver(s): return await solver_node(s, deps)
    async def _grounding_check(s): return await grounding_check_node(s, deps)
    async def _output_guardrail(s): return await output_guardrail_node(s, deps)

    # ------------------------------------------------------------------
    # Register nodes
    # ------------------------------------------------------------------
    graph.add_node("input_guardrail", _input_guardrail)
    graph.add_node("query_analyzer", _query_analyzer)
    graph.add_node("router", _router)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("tutor", _tutor)
    graph.add_node("exam", _exam)
    graph.add_node("solver", _solver)
    graph.add_node("grounding_check", _grounding_check)
    graph.add_node("output_guardrail", _output_guardrail)

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    graph.set_entry_point("input_guardrail")

    # Input guardrail → (ALLOW → query_analyzer) | (BLOCK → END)
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"query_analyzer": "query_analyzer", END: END},
    )

    # Query analyzer → router (always)
    graph.add_edge("query_analyzer", "router")

    # Router → (tutor path | exam path | solver path | END for unsupported)
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve_for_tutor": "retrieve",
            "retrieve_for_exam": "retrieve",
            "retrieve_for_solver": "retrieve",
            END: END,
        },
    )

    # Retrieve → tutor, exam, or solver based on intent in state
    def _route_retrieve(s):
        intent = s.get("intent")
        if intent in ("START_EXAM", "PRACTICE"):
            return "exam"
        if intent in ("EXPLAIN_CONCEPT", "GIVE_HINT", "CHECK_ANSWER", "REVISE_TOPIC", "START_VIVA"):
            return "tutor"
        return "solver"

    graph.add_conditional_edges(
        "retrieve",
        _route_retrieve,
        {"tutor": "tutor", "exam": "exam", "solver": "solver"},
    )

    # Tutor → grounding_check
    graph.add_edge("tutor", "grounding_check")

    # Exam → grounding_check
    graph.add_edge("exam", "grounding_check")

    # Solver → grounding_check
    graph.add_edge("solver", "grounding_check")

    # Grounding → output_guardrail
    graph.add_edge("grounding_check", "output_guardrail")

    # Output guardrail → (ALLOW → END) | (RETRY → tutor/exam/solver) | (BLOCK → END)
    graph.add_conditional_edges(
        "output_guardrail",
        route_after_output_guardrail,
        {
            END: END,
            "tutor": "tutor",
            "exam": "exam",
            "solver": "solver",
        },
    )

    return graph.compile()
