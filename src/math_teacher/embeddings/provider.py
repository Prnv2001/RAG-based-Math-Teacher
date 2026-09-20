"""Default embedding provider — backed by Sentence-Transformers / GroqEmbedder."""

from __future__ import annotations

from math_teacher.embeddings.groq_provider import GroqEmbedder

# GroqEmbedder is the default provider (no local Ollama required)
OllamaEmbedder = GroqEmbedder

