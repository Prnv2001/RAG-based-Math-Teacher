"""Ollama LLM provider — uses DeepSeek-R1 via local Ollama server."""

from __future__ import annotations

import json
import re
from typing import Any

import ollama

from math_teacher.config.settings import settings
from math_teacher.domain.errors import LLMError


class OllamaLLM:
    """Concrete LLM provider backed by Ollama (local DeepSeek-R1 or any Ollama model)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or settings.llm_model
        self._client = ollama.AsyncClient(host=base_url or settings.ollama_base_url)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate a plain-text response."""
        try:
            response = await self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": temperature},
            )
            return response.message.content or ""
        except Exception as exc:
            raise LLMError(f"Ollama generate failed: {exc}") from exc

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response from the model.

        DeepSeek-R1 wraps its reasoning in <think>...</think> tags.
        We strip those before attempting JSON parsing.
        """
        # Append explicit JSON instruction so the model always returns JSON
        json_instruction = "\n\nRespond with ONLY valid JSON. No explanation, no markdown, no code fences."
        try:
            raw = await self.generate(
                system_prompt=system_prompt + json_instruction,
                user_prompt=user_prompt,
                temperature=temperature,
            )
            cleaned = self._strip_thinking(raw)
            return self._parse_json(cleaned)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama generate_json failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove DeepSeek-R1 <think>...</think> reasoning blocks."""
        # Remove <think>...</think> blocks (DeepSeek-R1 specific)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Strip markdown code fences if model wraps JSON
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text)
        return text.strip()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from text, trying to find a JSON object if text has noise."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to extract first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise LLMError(f"Could not parse JSON from LLM response: {text[:300]!r}")


async def check_llm_health(model: str | None = None) -> bool:
    """Return True if Ollama is reachable and the model is available."""
    try:
        client = ollama.AsyncClient(host=settings.ollama_base_url)
        models = await client.list()
        target = model or settings.llm_model
        return any(target in m.model for m in models.models)
    except Exception:
        return False
