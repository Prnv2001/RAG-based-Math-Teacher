"""LLM provider Protocol — abstracts vendor-specific LLM calls."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Protocol for LLM providers.

    Concrete implementations: OllamaLLM (local Ollama/DeepSeek).
    LLM and embedding access MUST go through this interface, never raw SDK calls.
    """

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text response."""
        ...

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response.

        The returned dict is parsed from the LLM's output.
        Raises LLMError if the output is not valid JSON.
        """
        ...
