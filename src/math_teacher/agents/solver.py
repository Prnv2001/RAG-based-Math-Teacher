"""Solver agent — generates step-by-step math solutions with SymPy verification."""

from __future__ import annotations

import json

import re

from math_teacher.config.prompts import SOLVER_SYSTEM_PROMPT, SOLVER_USER_PROMPT
from math_teacher.domain.models import MathVerificationResult, SolverOutput
from math_teacher.llm.base import LLM
from math_teacher.math.verifier import MathVerifier


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


class SolverAgent:
    """Solves math problems step-by-step with deterministic verification.

    SECURITY: Retrieved context is DATA only — never followed as instructions.
    """

    def __init__(self, llm: LLM, verifier: MathVerifier | None = None) -> None:
        self._llm = llm
        self._verifier = verifier or MathVerifier()

    async def solve(
        self,
        query: str,
        context_text: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> tuple[str, MathVerificationResult]:
        """Generate a step-by-step solution and verify mathematically.

        Args:
            query: Student's problem statement.
            context_text: Assembled curriculum context.
            chat_history: Optional recent conversation history.
        Returns:
            Tuple of (formatted_answer_string, MathVerificationResult).
        """
        safe_context = context_text[:1500] if len(context_text) > 1500 else context_text

        raw = await self._llm.generate_json(
            system_prompt=SOLVER_SYSTEM_PROMPT,
            user_prompt=SOLVER_USER_PROMPT.format(
                history=_format_history(chat_history),
                context=safe_context,
                query=query,
            ),
            temperature=0.0,
        )

        if isinstance(raw, list):
            raw = {
                "approach": "Practice / Exam Questions",
                "steps": [json.dumps(item) if isinstance(item, dict) else str(item) for item in raw],
                "final_answer": "Complete the questions above.",
            }
        elif not isinstance(raw, dict):
            raw = {"approach": str(raw), "steps": [], "final_answer": str(raw)}

        # Parse solver output
        solver_out = SolverOutput(
            approach=raw.get("approach", ""),
            steps=raw.get("steps", []),
            final_answer=raw.get("final_answer", ""),
            equations_to_verify=raw.get("equations_to_verify", []),
        )

        # Run SymPy verification on any equations provided
        math_result = MathVerificationResult(
            passed=True,
            reason="No equations to verify.",
            status="not_required",
        )

        has_verified = False
        for eq_spec in solver_out.equations_to_verify:
            equation = eq_spec.get("equation", "")
            variable = eq_spec.get("variable", "x")
            roots = eq_spec.get("roots", [])

            if equation and roots:
                result = self._verifier.verify_equation_solution(
                    equation=equation,
                    variable=variable,
                    proposed_roots=roots,
                )
                has_verified = True
                if not result.passed:
                    math_result = result
                    break
                else:
                    math_result = result

        if not has_verified:
            math_result = self._verifier.verify_math_problem(
                problem=query,
                proposed_solution=solver_out.final_answer,
            )

        # Format answer as readable text
        answer_parts = []
        if solver_out.approach:
            answer_parts.append(f"**Approach:** {solver_out.approach}\n")
        if solver_out.steps:
            answer_parts.append("**Steps:**")
            import re
            step_idx = 1
            for step in solver_out.steps:
                step_str = str(step).strip()
                if not step_str or step_str.startswith("final_answer") or step_str.startswith("equations_to_verify"):
                    continue
                clean_step = re.sub(r"^(?:\d+[\.\)]|step\s*\d+[:\.]?)\s*", "", step_str, flags=re.I)
                answer_parts.append(f"{step_idx}. {clean_step}")
                step_idx += 1
        if solver_out.final_answer:
            answer_parts.append(f"\n**Answer:** {solver_out.final_answer}")

        from math_teacher.utils.text_sanitizer import clean_math_text
        answer = clean_math_text("\n".join(answer_parts))
        return answer, math_result
