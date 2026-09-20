"""Groq cloud LLM provider — drop-in replacement for OllamaLLM.

Uses the Groq Python SDK (OpenAI-compatible).
Primary key is tried first; on rate-limit (429) or auth error the fallback
key is used automatically so the pipeline keeps running.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import groq as groq_sdk

from math_teacher.config.settings import settings
from math_teacher.domain.errors import LLMError

logger = logging.getLogger(__name__)


class GroqLLM:
    """Async LLM provider backed by Groq cloud API.

    Identical public interface to OllamaLLM:
        • generate(system_prompt, user_prompt, temperature) -> str
        • generate_json(system_prompt, user_prompt, schema, temperature) -> dict

    Fallback behaviour:
        If the primary key raises a RateLimitError or AuthenticationError,
        the request is automatically retried once with the fallback key.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_key_fallback: str | None = None,
    ) -> None:
        self.model = model or settings.groq_model
        self._primary_key = api_key or settings.groq_api_key
        self._fallback_key = api_key_fallback or settings.groq_api_key_fallback
        self._primary = groq_sdk.AsyncGroq(api_key=self._primary_key)
        self._fallback = (
            groq_sdk.AsyncGroq(api_key=self._fallback_key)
            if self._fallback_key
            else None
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a plain-text response."""
        raw = await self._call_with_fallback(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            json_mode=False,
            max_tokens=max_tokens or 3500,
        )
        return self._strip_thinking(raw)


    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response.

        Appends an explicit JSON instruction and uses Groq's
        ``response_format={"type": "json_object"}`` for reliable output.
        """
        json_instruction = (
            "\n\nRespond with ONLY valid JSON. "
            "No explanation, no markdown, no code fences."
        )
        raw = await self._call_with_fallback(
            system_prompt=system_prompt + json_instruction,
            user_prompt=user_prompt,
            temperature=temperature,
            json_mode=True,
            max_tokens=max_tokens or 2000,
        )
        cleaned = self._strip_thinking(raw)
        return self._parse_json(cleaned)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_with_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        json_mode: bool = False,
        max_tokens: int = 2000,
    ) -> str:
        """Try active client; on rate-limit / auth error, rotate keys & fallback model with adaptive max_tokens."""
        import asyncio

        max_attempts = 4
        last_exception: Exception | None = None

        base_tokens = max_tokens
        fallback_models = [settings.groq_model, "qwen/qwen3.8-27b", "groq/compound-mini"]

        for attempt in range(max_attempts):
            token_cap = max(600, base_tokens - (attempt * 150))
            active_model = fallback_models[attempt % len(fallback_models)]

            try:
                return await self._do_call(
                    self._primary,
                    system_prompt,
                    user_prompt,
                    temperature,
                    json_mode,
                    max_tokens=token_cap,
                    model=active_model,
                )
            except (groq_sdk.RateLimitError, groq_sdk.AuthenticationError, groq_sdk.APIStatusError, groq_sdk.APIError) as exc:
                last_exception = exc
                if self._fallback is not None:
                    # Rotate keys so future calls immediately use working key
                    self._primary, self._fallback = self._fallback, self._primary
                    logger.info(
                        "Groq key/model rotated on %s (attempt %d/%d). Model: %s, token_cap: %d.",
                        type(exc).__name__,
                        attempt + 1,
                        max_attempts,
                        active_model,
                        token_cap,
                    )
                # Wait briefly before retrying
                await asyncio.sleep(1.0 * (attempt + 1))
            except Exception as exc:
                raise LLMError(f"Groq generate failed: {exc}") from exc

        raise LLMError(
            f"Groq API rate limit reached on all configured keys after {max_attempts} attempts. Details: {last_exception}"
        ) from last_exception

    @staticmethod
    async def _do_call(
        client: groq_sdk.AsyncGroq,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        json_mode: bool,
        max_tokens: int = 2000,
        model: str | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model or settings.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove any <think>...</think> blocks (forward-compat with reasoning models)."""
        if not text:
            return ""
        # 1. Strip complete <think>...</think> blocks
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # 2. If <think> was unclosed (truncated), strip <think>... up to end of string ONLY IF there is content outside it
        if "<think>" in cleaned:
            no_think = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
            if no_think:
                cleaned = no_think
            else:
                # Extract whatever text was produced inside think as emergency fallback so output is not empty
                inside_think = re.sub(r"^.*?<think>", "", cleaned, flags=re.DOTALL).strip()
                cleaned = inside_think
        cleaned = re.sub(r"```(?:json|markdown|xml|svg)?\s*", "", cleaned)
        cleaned = re.sub(r"```", "", cleaned)
        return cleaned.strip()


    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from LLM text, with fallback extraction and truncation auto-repair."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Attempt auto-repair on truncated JSON output
        repaired = text.strip()
        if not repaired.startswith("{"):
            start_idx = repaired.find("{")
            if start_idx != -1:
                repaired = repaired[start_idx:]
            else:
                repaired = "{" + repaired

        # Auto-close open quotes if unescaped quote count is odd
        quote_count = len(re.findall(r'(?<!\\)"', repaired))
        if quote_count % 2 != 0:
            repaired += '"'

        open_brackets = repaired.count("[") - repaired.count("]")
        open_braces = repaired.count("{") - repaired.count("}")

        repaired += "]" * max(0, open_brackets)
        repaired += "}" * max(0, open_braces)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        raise LLMError(f"Could not parse JSON from Groq response: {text[:300]!r}")


async def check_groq_health() -> bool:
    """Return True if Groq is reachable with the configured primary key."""
    try:
        client = groq_sdk.AsyncGroq(api_key=settings.groq_api_key)
        models = await client.models.list()
        return any(settings.groq_model in m.id for m in models.data)
    except Exception:
        return False
