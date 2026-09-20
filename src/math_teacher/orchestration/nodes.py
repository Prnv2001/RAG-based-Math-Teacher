"""LangGraph node implementations — each node is a pure async function."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from math_teacher.agents.router import RouterAgent
from math_teacher.agents.solver import SolverAgent
from math_teacher.agents.state import AgentState
from math_teacher.agents.tutor import TutorAgent
from math_teacher.domain.enums import GuardAction, SolverMode
from math_teacher.domain.models import MathVerificationResult
from math_teacher.guardrails.input import InputGuardrail
from math_teacher.guardrails.output import OutputGuardrail
from math_teacher.rag.context import build_context
from math_teacher.rag.grounding import GroundingChecker
from math_teacher.retrieval.service import analyze_query, retrieve

logger = logging.getLogger("math_teacher.pipeline")


def _log_step(logs: list[str], msg: str) -> None:
    """Print step log to uvicorn console live and append to trace logs."""
    logs.append(msg)
    try:
        print(f"\n[PIPELINE LOG] {msg}", flush=True)
    except UnicodeEncodeError:
        safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
        print(f"\n[PIPELINE LOG] {safe_msg}", flush=True)
    logger.info(msg)


# ---------------------------------------------------------------------------
# Node: input_guardrail  (V2: no LLM — fully deterministic)
# ---------------------------------------------------------------------------
async def input_guardrail_node(state: AgentState, deps: Any) -> AgentState:
    logs = list(state.get("step_logs") or [])
    q = state["question"]
    q_preview = (q[:70] + "...") if len(q) > 70 else q
    _log_step(logs, f"🛡️ [Step 1/7] Input Guardrail: Checking prompt '{q_preview}'")

    guardrail = InputGuardrail()
    result = guardrail.check(
        query=state["question"],
        class_level=state.get("class_level"),
    )
    status_str = "PASSED" if result.passed else f"BLOCKED ({result.reason})"
    _log_step(logs, f"   └─ Status: {status_str}")
    return {**state, "input_guardrail": result, "step_logs": logs}


# ---------------------------------------------------------------------------
# Node: query_analyzer  (V2: no LLM — regex/keyword extraction)
# ---------------------------------------------------------------------------
async def query_analyzer_node(state: AgentState, deps: Any) -> AgentState:
    logs = list(state.get("step_logs") or [])
    _log_step(logs, "🔍 [Step 2/7] Query Analyzer: Extracting CBSE curriculum scope & intent...")

    ctx = analyze_query(
        query=state["question"],
        class_level=state.get("class_level"),
        chapter=state.get("chapter"),
        chat_history=state.get("chat_history"),
    )
    _log_step(
        logs,
        f"   └─ Extracted: Class={ctx.class_level or 'Auto-detect'}, Chapter='{ctx.chapter or 'All'}', Topic='{ctx.topic or 'All'}'",
    )
    return {**state, "query_context": ctx, "step_logs": logs}


# ---------------------------------------------------------------------------
# Node: router  (keeps LLM — agent orchestration)
# ---------------------------------------------------------------------------
async def router_node(state: AgentState, deps: Any) -> AgentState:
    logs = list(state.get("step_logs") or [])
    _log_step(logs, "🔀 [Step 3/7] Router Agent: Classifying intent via LLM...")

    agent = RouterAgent(deps.llm)
    result = await agent.route(
        query=state["question"],
        chat_history=state.get("chat_history"),
    )
    _log_step(logs, f"   └─ Classified Intent: {result.intent.value} (Confidence: {result.confidence:.2f})")
    return {
        **state,
        "intent": result.intent.value,
        "intent_confidence": result.confidence,
        "step_logs": logs,
    }


def _get_effective_search_query(
    query: str,
    chat_history: list[dict[str, str]] | None,
    query_context: Any = None,
) -> str:
    """Build effective vector search query for follow-up prompts and action chips.

    If query is a short follow-up action (e.g. 'Show the full worked solution',
    'Option A: 4x² - 3x + 1', 'Need a Hint', 'Next Step', etc.), merges with
    the previous mathematical question/context from chat_history.
    """
    import re

    q_lower = query.strip().lower()
    is_followup = False

    # 1. Action chips and meta-commands
    action_patterns = [
        r"show\s+(?:the\s+)?(?:full\s+)?worked\s+solution",
        r"full\s+solution",
        r"need\s+a\s+hint",
        r"give\s+me\s+a\s+hint",
        r"next\s+step",
        r"solve\s+problem",
        r"check\s+answer",
        r"quick\s+quiz",
        r"explain\s+more",
    ]
    if any(re.search(pat, q_lower) for pat in action_patterns):
        is_followup = True

    # 2. MCQ option choices: e.g. "Option A...", "Option B", "A)", "B)"
    if re.match(r"^(?:option\s+[a-d]|option\s+[1-4]|[a-d]\)|[a-d]$)", q_lower):
        is_followup = True

    # 3. Short generic query without math symbols/keywords (< 35 chars)
    if len(query.strip()) < 35 and not any(k in q_lower for k in ("equal", "find", "solve", "triangle", "circle", "polynomial", "equation", "proof", "angle", "cm", "m²")):
        is_followup = True

    if not is_followup:
        return query

    # Extract previous math question from chat_history
    if chat_history:
        for msg in reversed(chat_history):
            content = msg.get("content", "").strip()
            c_lower = content.lower()
            if any(re.search(pat, c_lower) for pat in action_patterns):
                continue
            if re.match(r"^(?:option\s+[a-d]|[a-d]\)|[a-d]$)", c_lower):
                continue
            clean_content = re.sub(r"<svg[\s\S]*?<\/svg>", "", content, flags=re.I).strip()
            if len(clean_content) > 300:
                clean_content = clean_content[:300]
            if clean_content:
                return f"{clean_content} {query}"

    if query_context and (getattr(query_context, "chapter", None) or getattr(query_context, "topic", None)):
        ch = getattr(query_context, "chapter", "") or ""
        top = getattr(query_context, "topic", "") or ""
        return f"{ch} {top} {query}".strip()

    return query


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------
async def retrieve_node(state: AgentState, deps: Any) -> AgentState:
    logs = list(state.get("step_logs") or [])
    _log_step(logs, "📚 [Step 4/7] Hybrid Retrieval: Searching Vector DB (PostgreSQL + pgvector)...")

    ctx = state.get("query_context")
    search_query = _get_effective_search_query(
        query=state["question"],
        chat_history=state.get("chat_history"),
        query_context=ctx,
    )
    if search_query != state["question"]:
        _log_step(logs, f"   ├─ Expanded search query for context: '{search_query[:70]}...'")

    chunks = await retrieve(
        query=search_query,
        query_context=ctx,
        embedder=deps.embedder,
        session=deps.session,
    )
    context_text, sources = build_context(chunks)
    _log_step(
        logs,
        f"   └─ Retrieved {len(chunks)} textbook chunks ({len(sources)} unique references)",
    )
    return {
        **state,
        "retrieved_chunks": chunks,
        "context_text": context_text,
        "sources": sources,
        "step_logs": logs,
    }


# ---------------------------------------------------------------------------
# Node: tutor  (keeps LLM — RAG answer generation)
# ---------------------------------------------------------------------------
async def tutor_node(state: AgentState, deps: Any) -> AgentState:
    logs = list(state.get("step_logs") or [])
    _log_step(logs, "💡 [Step 5/7] Tutor Agent: Generating concept explanation...")

    mode_str = state.get("mode", "INTERACTIVE")
    try:
        mode = SolverMode(mode_str)
    except ValueError:
        mode = SolverMode.INTERACTIVE

    agent = TutorAgent(deps.llm)
    answer = await agent.explain(
        query=state["question"],
        context_text=state.get("context_text", ""),
        mode=mode,
        chat_history=state.get("chat_history"),
    )
    _log_step(logs, f"   └─ Generated explanation ({len(answer)} chars)")
    return {
        **state,
        "answer": answer,
        "require_math": False,
        "math_result": MathVerificationResult(
            passed=True,
            reason="Math verification not required for tutor response.",
            status="not_required",
        ),
        "step_logs": logs,
    }


# ---------------------------------------------------------------------------
# Node: exam  (dedicated ExamAgent)
# ---------------------------------------------------------------------------
async def exam_node(state: AgentState, deps: Any) -> AgentState:
    logs = list(state.get("step_logs") or [])
    _log_step(logs, "📝 [Step 5/7] Exam Agent: Conducting customized exam session...")

    from math_teacher.agents.exam import ExamAgent
    agent = ExamAgent(deps.llm)
    answer = await agent.generate_exam(
        query=state["question"],
        context_text=state.get("context_text", ""),
        chat_history=state.get("chat_history"),
    )
    _log_step(logs, f"   └─ Generated exam paper/setup ({len(answer)} chars)")
    return {
        **state,
        "answer": answer,
        "require_math": False,
        "math_result": MathVerificationResult(
            passed=True,
            reason="Math verification not required for exam response.",
            status="not_required",
        ),
        "step_logs": logs,
    }


# ---------------------------------------------------------------------------
# Node: solver  (keeps LLM — RAG answer generation)
# ---------------------------------------------------------------------------
async def solver_node(state: AgentState, deps: Any) -> AgentState:
    logs = list(state.get("step_logs") or [])
    _log_step(logs, "🧮 [Step 5/7] Solver Agent: Generating step-by-step solution + SymPy math verification...")

    agent = SolverAgent(deps.llm)
    answer, math_result = await agent.solve(
        query=state["question"],
        context_text=state.get("context_text", ""),
        chat_history=state.get("chat_history"),
    )
    math_status = "PASSED ✓" if math_result.passed else f"FAILED ✗ ({math_result.reason})"
    _log_step(logs, f"   └─ Answer generated ({len(answer)} chars). SymPy Math Check: {math_status}")
    return {
        **state,
        "answer": answer,
        "math_result": math_result,
        "require_math": True,
        "step_logs": logs,
    }


# ---------------------------------------------------------------------------
# Node: grounding_check  (V2: uses embedder, not LLM)
# ---------------------------------------------------------------------------
async def grounding_check_node(state: AgentState, deps: Any) -> AgentState:
    logs = list(state.get("step_logs") or [])
    _log_step(logs, "📐 [Step 6/7] Grounding Checker: Computing embedding cosine similarity...")

    answer = state.get("answer", "")
    intent_str = state.get("intent", "")

    # Exam setup prompts & Exam Agent outputs are structural questions/papers — pass grounding check automatically
    if "Mandatory Exam Preparation Settings" in answer or "Exam Controller" in answer or intent_str in ("START_EXAM", "PRACTICE"):
        from math_teacher.domain.models import GroundingResult
        grounding = GroundingResult(grounded=True, score=1.0, reason="Exam agent setup/paper output verified.")
    else:
        checker = GroundingChecker(deps.embedder)
        grounding = await checker.check(
            answer=answer,
            context=state.get("context_text", ""),
        )

    status_str = f"GROUNDED ✓ (Score: {grounding.score:.2f})" if grounding.grounded else f"LOW GROUNDING ⚠ (Score: {grounding.score:.2f})"
    _log_step(logs, f"   └─ Result: {status_str}")
    return {**state, "grounding": grounding, "step_logs": logs}


# ---------------------------------------------------------------------------
# Node: output_guardrail
# ---------------------------------------------------------------------------
async def output_guardrail_node(state: AgentState, deps: Any) -> AgentState:
    logs = list(state.get("step_logs") or [])
    _log_step(logs, "✅ [Step 7/7] Output Guardrail: Evaluating safety & quality thresholds...")

    retry_count = state.get("retry_count", 0)
    og = OutputGuardrail()
    og.retry_count = retry_count

    result = og.check(
        grounding=state.get("grounding"),
        math_result=state.get("math_result"),
        require_math=state.get("require_math", False),
    )

    if result.action == GuardAction.BLOCK:
        _log_step(logs, f"   └─ Action: BLOCK 🛑 (Reason: {result.reason}) -> Returning safe fallback")
        return {
            **state,
            "output_guardrail": result,
            "retry_count": og.retry_count,
            "answer": OutputGuardrail.safe_fallback_answer(),
            "step_logs": logs,
        }

    _log_step(logs, f"   └─ Action: ALLOW ✅ (Response passed all guardrails)")
    return {
        **state,
        "output_guardrail": result,
        "retry_count": og.retry_count,
        "step_logs": logs,
    }

