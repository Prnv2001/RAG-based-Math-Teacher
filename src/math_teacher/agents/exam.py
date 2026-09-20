"""Exam agent — dedicated agent for interactive exam configuration & custom exam paper generation."""

from __future__ import annotations

import re
from typing import Any

from math_teacher.config.prompts import EXAM_SYSTEM_PROMPT, EXAM_USER_PROMPT
from math_teacher.llm.base import LLM


def _format_history(history: list[dict[str, str]] | None, max_msg_len: int = 2000) -> str:
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


def _extract_exam_params(query: str, chat_history: list[dict[str, str]] | None) -> dict[str, Any]:
    """Parse query and user chat history for total_marks, num_questions, and difficulty."""
    text_lower = query.lower()

    # Check if this query is a new exam command (e.g. "create a exam for chapter 10")
    is_new_exam_cmd = any(k in text_lower for k in ("create", "make", "generate", "take", "conduct")) and any(k in text_lower for k in ("exam", "test", "paper"))

    user_texts = [query]
    # Only pull chat history if user is answering a setup question, NOT when starting a new exam command
    if chat_history and not is_new_exam_cmd:
        for msg in chat_history[-3:]:
            if msg.get("role") == "user":
                user_texts.append(msg.get("content", ""))

    combined_text = " ".join(user_texts)
    comb_lower = combined_text.lower()

    # Total Marks: e.g. "20 marks", "50 marks", "100 marks"
    marks_match = re.search(r"\b(\d{1,3})\s*marks?\b", comb_lower)
    total_marks = marks_match.group(1) + " Marks" if marks_match else None

    # Num Questions: e.g. "3 questions", "5 qns" (do NOT match isolated 'q' to avoid matching 'question paper')
    q_match = re.search(r"\b(\d{1,2})\s*(?:questions?|qns?)\b", comb_lower)
    num_questions = q_match.group(1) + " Questions" if q_match else None

    # Difficulty: e.g. "easy", "medium", "hard", "tough", "board level"
    difficulty = None
    if "hard" in comb_lower or "tough" in comb_lower or "advanced" in comb_lower:
        difficulty = "Hard (Board Standard)"
    elif "easy" in comb_lower or "simple" in comb_lower or "basic" in comb_lower:
        difficulty = "Easy"
    elif "medium" in comb_lower or "moderate" in comb_lower:
        difficulty = "Medium"

    # Default override request: "default", "start default", "quick exam", "begin exam"
    is_default = any(k in text_lower for k in ("default", "start default", "quick exam", "begin exam", "default exam"))

    return {
        "total_marks": total_marks,
        "num_questions": num_questions,
        "difficulty": difficulty,
        "is_default": is_default,
    }


def _is_asking_for_solutions(query: str) -> bool:
    """Check if query requests exam solutions or answer key."""
    q_lower = query.lower()
    sol_keywords = (
        "solution", "solutions", "answer", "answers", "answer key",
        "solution key", "show answer", "show solution", "give answer",
        "give solution", "check my answer", "check answer", "reveal", "key"
    )
    return any(kw in q_lower for kw in sol_keywords)


class ExamAgent:
    """Dedicated Exam Controller Agent.

    Handles interactive exam configuration and custom exam generation.
    - Stage 1: Interactive configuration setup.
    - Stage 2: Question Paper ONLY (hides solutions until requested).
    - Stage 3: Solution & Answer Key ONLY (when student requests solutions).
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def generate_exam(
        self,
        query: str,
        context_text: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Conduct an exam or return interactive configuration prompt / solution key."""
        is_sol_req = _is_asking_for_solutions(query)
        params = _extract_exam_params(query, chat_history)

        total_marks = params["total_marks"]
        num_questions = params["num_questions"]
        difficulty = params["difficulty"]
        is_default = params["is_default"]

        has_all_params = bool(total_marks and num_questions and difficulty)

        # MANDATE: If user hasn't specified ALL 3 parameters, hasn't requested default, and isn't asking for solutions:
        if not has_all_params and not is_default and not is_sol_req:
            return (
                "📝 **Exam Controller — Mandatory Exam Preparation Settings**\n\n"
                "Before I generate your customized question paper, please specify your exam preferences:\n\n"
                "1. **Total Marks:** 20 Marks / 30 Marks / 50 Marks *(Default: 20 Marks)*\n"
                "2. **Number of Questions:** 3 Questions / 5 Questions *(Default: 3 Questions)*\n"
                "3. **Difficulty Level:** Easy / Medium / Hard *(Default: Medium)*\n\n"
                "👉 **Please reply with your settings** (e.g. *\"50 marks, 5 questions, hard\"*) or type **\"start default exam\"** to begin immediately!"
            )

        # Mode selection: QUESTION_PAPER unless user explicitly asks for solutions
        mode = "SOLUTIONS_ONLY" if is_sol_req else "QUESTION_PAPER"

        # Fill defaults for unstated parameters
        final_marks = total_marks or "20 Marks"
        final_q_count = num_questions or "3 Questions"
        final_diff = difficulty or "Medium"

        # Dynamically scale max_tokens for larger question counts or full solution key
        q_count_num = 3
        q_num_match = re.search(r"(\d+)", final_q_count)
        if q_num_match:
            q_count_num = int(q_num_match.group(1))

        max_tok = 3800 if (q_count_num >= 5 or is_sol_req) else 2200

        safe_context = context_text[:1200] if len(context_text) > 1200 else context_text

        raw_ans = await self._llm.generate(
            system_prompt=EXAM_SYSTEM_PROMPT.format(
                total_marks=final_marks,
                num_questions=final_q_count,
                difficulty=final_diff,
            ),
            user_prompt=EXAM_USER_PROMPT.format(
                history=_format_history(chat_history),
                context=safe_context,
                query=query,
                total_marks=final_marks,
                num_questions=final_q_count,
                difficulty=final_diff,
                mode=mode,
            ),
            temperature=0.7,
            max_tokens=max_tok,
        )

        from math_teacher.utils.text_sanitizer import clean_math_text
        return clean_math_text(raw_ans)
