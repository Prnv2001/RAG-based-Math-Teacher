"""Input guardrail — regex-based injection detection + curriculum scope check.

V2: Replaced the LLM-based injection classifier with a deterministic
regex / keyword approach to save one Groq API call per request.
"""

from __future__ import annotations

import re

from math_teacher.domain.enums import GuardAction
from math_teacher.domain.models import GuardrailResult


# ---------------------------------------------------------------------------
# Injection patterns (compiled once at import time)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Role / persona manipulation
    (re.compile(r"\b(you are now|act as|pretend to be|roleplay as|behave like)\b", re.I),
     "Role manipulation attempt"),
    # Instruction override
    (re.compile(r"\b(ignore (all |your |previous )?instructions|forget (your|all) (rules|instructions))\b", re.I),
     "Instruction override attempt"),
    (re.compile(r"\b(disregard|override|bypass|skip) (the |your )?(system|safety|guard|rules|prompt)\b", re.I),
     "Safety bypass attempt"),
    # Prompt / secret extraction
    (re.compile(r"\b(show|reveal|display|print|output|repeat|tell me) (me )?(the |your )?(system ?prompt|instructions|rules|hidden|secret|internal)\b", re.I),
     "Prompt extraction attempt"),
    (re.compile(r"\b(what are your (instructions|rules|prompts))\b", re.I),
     "Prompt extraction attempt"),
    # DAN / jailbreak
    (re.compile(r"\b(DAN|do anything now|jailbreak|unrestricted mode)\b", re.I),
     "Jailbreak attempt"),
    # Data-as-instructions
    (re.compile(r"\b(execute|run|eval|follow) (the |this )?(code|command|instruction|script)\b", re.I),
     "Code execution attempt"),
    # Delimiter injection
    (re.compile(r"<\|?(system|im_start|im_end|endoftext)\|?>", re.I),
     "Delimiter injection"),
    # Encoding evasion (base64 / hex instructions)
    (re.compile(r"\b(base64|decode this|hex encode|rot13)\b", re.I),
     "Encoding evasion attempt"),
]

# Short / empty message threshold
_MIN_QUERY_LENGTH = 2


class InjectionGuardrail:
    """Regex / keyword prompt-injection classifier.

    Deterministic — no LLM call required.
    """

    def check(self, query: str) -> GuardrailResult:
        """Run injection detection on the raw student query."""
        if not query or len(query.strip()) < _MIN_QUERY_LENGTH:
            return GuardrailResult(
                passed=False,
                action=GuardAction.BLOCK,
                reason="Query too short or empty.",
                score=1.0,
            )

        for pattern, reason in _INJECTION_PATTERNS:
            if pattern.search(query):
                return GuardrailResult(
                    passed=False,
                    action=GuardAction.BLOCK,
                    reason=f"Injection detected: {reason}",
                    score=1.0,
                )

        return GuardrailResult(
            passed=True,
            action=GuardAction.ALLOW,
            reason="No injection detected",
            score=0.0,
        )


class CurriculumGuardrail:
    """Restricts queries to supported class levels (9 and 10)."""

    SUPPORTED_CLASSES = {9, 10}

    def check(self, query: str, class_level: int | None) -> GuardrailResult:
        """Check if the requested class level is within scope."""
        if class_level is None:
            # No class specified — allow and let Query Analyzer infer
            return GuardrailResult(
                passed=True,
                action=GuardAction.ALLOW,
                reason="No class level specified; Query Analyzer will infer.",
            )
        if class_level not in self.SUPPORTED_CLASSES:
            return GuardrailResult(
                passed=False,
                action=GuardAction.BLOCK,
                reason=(
                    f"Class {class_level} is not supported in V1. "
                    f"This system covers Class 9 and Class 10 Mathematics."
                ),
            )
        return GuardrailResult(
            passed=True,
            action=GuardAction.ALLOW,
            reason=f"Class {class_level} is within supported scope.",
        )


class InputGuardrail:
    """Orchestrates all input-level guardrail checks.

    Order: injection detection → curriculum scope.
    Fails fast on the first blocking result.

    V2: No LLM dependency — fully deterministic.
    """

    def __init__(self) -> None:
        self._injection = InjectionGuardrail()
        self._curriculum = CurriculumGuardrail()

    def check(self, query: str, class_level: int | None = None) -> GuardrailResult:
        """Run all input guardrails. Returns first blocking result or ALLOW.

        Note: This is now synchronous (no async needed without LLM calls).
        """
        # 1. Injection check
        injection_result = self._injection.check(query)
        if not injection_result.passed:
            return injection_result

        # 2. Curriculum scope check
        curriculum_result = self._curriculum.check(query, class_level)
        if not curriculum_result.passed:
            return curriculum_result

        return GuardrailResult(
            passed=True,
            action=GuardAction.ALLOW,
            reason="All input guardrails passed.",
        )
