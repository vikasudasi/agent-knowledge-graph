"""LLM client interfaces for prompt-to-Cypher and summarization."""

from __future__ import annotations


class LLMClient:
    """Abstraction for provider-backed large language model operations."""

    def __init__(self, api_base: str, api_key: str) -> None:
        self.api_base = api_base
        self.api_key = api_key

    def complete(self, prompt: str) -> str:
        """Generate a text completion for the input prompt."""
        raise NotImplementedError("LLMClient.complete is not yet implemented")
