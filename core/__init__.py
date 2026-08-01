"""Core package for agent-knowledge-graph."""

from __future__ import annotations

from core.config import EmbeddingConfig, KGConfig, LLMConfig, Neo4jConfig, load_config
from core.embedding import (
    EmbeddingProvider,
    EmbeddingProviderFactory,
    LocalEmbeddingProvider,
    RemoteEmbeddingProvider,
)
from core.graph import Neo4jClient
from core.llm import LLMClient, LLMProviderFactory, OpenRouterProvider
from core.models import GraphStats, PipelineCheckpoint, QueryResult, Relationship, Resource

__all__ = [
    "KGConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "RemoteEmbeddingProvider",
    "EmbeddingProviderFactory",
    "Neo4jConfig",
    "load_config",
    "Neo4jClient",
    "LLMClient",
    "OpenRouterProvider",
    "LLMProviderFactory",
    "Resource",
    "Relationship",
    "GraphStats",
    "PipelineCheckpoint",
    "QueryResult",
]
