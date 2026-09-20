"""Router agent — classifies student query intent using the LLM."""

from __future__ import annotations

import re

from math_teacher.config.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT
from math_teacher.domain.enums import Intent
from math_teacher.domain.models import RouterResult
from math_teacher.llm.base import LLM

# V1 supported intents — all others return NOT_IMPLEMENTED
V1_SUPPORTED_INTENTS = {
    Intent.EXPLAIN_CONCEPT,
    Intent.SOLVE_PROBLEM,
    Intent.CHECK_ANSWER,
    Intent.GIVE_HINT,
    Intent.PRACTICE,
    Intent.START_EXAM,
    Intent.REVISE_TOPIC,
    Intent.START_VIVA,
}


def _format_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return "None"
    formatted = []
    for msg in history[-3:]:
        role = "Student" if msg.get("role") == "user" else "Teacher"
        content = msg.get("content", "")
        content = re.sub(r"<svg[\s\S]*?<\/svg>", "[SVG Diagram]", content, flags=re.I)
        if len(content) > 80:
            content = content[:80] + "..."
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


class RouterAgent:
    """Classifies query intent and routes to appropriate V1 agent."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def route(
        self, query: str, chat_history: list[dict[str, str]] | None = None
    ) -> RouterResult:
        """Classify query intent with conversation context."""
        q_lower = query.lower()

        # Deterministic Exam keyword override (regex pattern to catch all exam initiation requests)
        exam_pattern = r"\b(?:exam|test\s*paper|question\s*paper|sample\s*paper|quiz\s*paper|conduct\s*(?:an?|a)?\s*exam|create\s*(?:an?|a)?\s*exam|make\s*(?:an?|a)?\s*exam|generate\s*(?:an?|a)?\s*exam|take\s*(?:an?|a)?\s*exam|exam\s*for|test\s*on)\b"
        if re.search(exam_pattern, q_lower):
            return RouterResult(intent=Intent.START_EXAM, confidence=1.0)

        try:
            hist_str = _format_history(chat_history)
            result = await self._llm.generate_json(
                system_prompt=ROUTER_SYSTEM_PROMPT,
                user_prompt=ROUTER_USER_PROMPT.format(history=hist_str, query=query),
                temperature=0.0,
                max_tokens=150,
            )
            intent_str = result.get("intent", "OUT_OF_SCOPE")
            confidence = float(result.get("confidence", 0.5))

            try:
                intent = Intent(intent_str)
            except ValueError:
                intent = Intent.OUT_OF_SCOPE

            # Map V1-unsupported intents to NOT_IMPLEMENTED
            if intent not in V1_SUPPORTED_INTENTS and intent != Intent.OUT_OF_SCOPE:
                intent = Intent.NOT_IMPLEMENTED

            return RouterResult(intent=intent, confidence=confidence)

        except Exception as exc:
            # Default to EXPLAIN_CONCEPT on failure — safe fallback
            return RouterResult(intent=Intent.EXPLAIN_CONCEPT, confidence=0.0)
