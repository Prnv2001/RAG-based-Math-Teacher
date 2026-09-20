"""Default LLM provider — backed by Groq Cloud API."""

from __future__ import annotations

from math_teacher.llm.groq_provider import GroqLLM

# GroqLLM is the primary LLM provider (no local Ollama required)
OllamaLLM = GroqLLM


async def check_llm_health() -> bool:
    """Return True if Groq API key is set and Groq Cloud is reachable."""
    try:
        llm = GroqLLM()
        res = await llm.generate(
            system_prompt="You are a health check assistant.",
            user_prompt="Reply OK",
            temperature=0.0,
        )
        return bool(res)
    except Exception:
        return False
