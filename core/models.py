"""Pydantic data models used across the repository."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """Structured memory item that can be stored in graph and vector index."""

    node_id: str = Field(..., description="Stable node identifier")
    content: str = Field(..., description="Canonical text content")
    tags: list[str] = Field(default_factory=list, description="Tag labels for filtering")


class ModelRegistry:
    """Placeholder registry for model metadata and schemas."""

    def register(self, key: str, model: type[BaseModel]) -> None:
        """Register a model class under a unique key."""
        raise NotImplementedError("ModelRegistry.register is not yet implemented")
