"""Embedding provider interfaces for semantic retrieval."""

from __future__ import annotations


class EmbeddingProvider:
    """Generate vector embeddings for textual records."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        """Embed text into a fixed-size float vector."""
        raise NotImplementedError("EmbeddingProvider.embed is not yet implemented")
