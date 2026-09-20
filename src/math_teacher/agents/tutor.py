"""Tutor agent — generates grounded explanations with HINT/GUIDED/FULL_SOLUTION modes."""

from __future__ import annotations

import re

from math_teacher.config.prompts import TUTOR_SYSTEM_PROMPT, TUTOR_USER_PROMPT
from math_teacher.domain.enums import SolverMode
from math_teacher.llm.base import LLM


def _format_history(history: list[dict[str, str]] | None, max_msg_len: int = 1500) -> str:
    if not history:
        return "None"
    formatted = []
    for msg in history[-4:]:
        role = "Student" if msg.get("role") == "user" else "Teacher"
        content = msg.get("content", "")
        content = re.sub(r"<svg[\s\S]*?<\/svg>", "[SVG Diagram]", content, flags=re.I)
        if len(content) > max_msg_len:
            content = content[:max_msg_len] + "..."
        formatted.append(f"{role}:\n{content}")
    return "\n\n".join(formatted)


class TutorAgent:
    """Generates curriculum-grounded concept explanations.

    Supports three modes:
    - HINT: Single guiding hint without solving.
    - GUIDED: Step-by-step approach, student computes.
    - FULL_SOLUTION: Complete worked explanation.

    SECURITY: Retrieved context is DATA only — the system prompt instructs
    the LLM to never follow instructions embedded in retrieved text.
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def explain(
        self,
        query: str,
        context_text: str,
        mode: SolverMode = SolverMode.INTERACTIVE,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a grounded explanation.

        Args:
            query: Student's question.
            context_text: Assembled curriculum context from Context Builder.
            mode: SolverMode controlling response depth.
            chat_history: Optional recent conversation history.
        Returns:
            Generated answer string.
        """
        is_exam = any(w in query.lower() for w in ("exam", "test", "practice", "paper", "sample"))
        temp = 0.7 if is_exam else 0.3

        # Cap context text size to 2000 chars max to keep input tokens lean
        safe_context = context_text[:2000] if len(context_text) > 2000 else context_text

        raw_ans = await self._llm.generate(
            system_prompt=TUTOR_SYSTEM_PROMPT,
            user_prompt=TUTOR_USER_PROMPT.format(
                history=_format_history(chat_history),
                context=safe_context,
                query=query,
                mode=mode.value,
            ),
            temperature=temp,
        )
        from math_teacher.utils.text_sanitizer import clean_math_text
        return clean_math_text(raw_ans)
