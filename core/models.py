"""Pydantic data models used across the repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

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


@dataclass
class Resource:
    """A node in the knowledge graph."""

    id: str
    type: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    ingested_at: datetime | None = None


@dataclass
class Relationship:
    """A directed edge between two resources."""

    source_id: str
    target_id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineCheckpoint:
    """Track pipeline progress for idempotent runs."""

    pipeline_name: str
    last_processed_id: str = ""
    last_processed_timestamp: datetime | None = None
    total_processed: int = 0
    updated_at: datetime | None = None


@dataclass
class QueryResult:
    """Result container for graph/query operations."""

    nodes: list[Resource] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    scores: list[float] | None = None
    execution_time_ms: float = 0.0


@dataclass
class GraphStats:
    """Statistics about graph contents and index state."""

    node_count: int = 0
    relationship_count: int = 0
    vector_index_ready: bool = False
    last_checkpoints: dict[str, PipelineCheckpoint] = field(default_factory=dict)
    database_size_mb: float = 0.0
