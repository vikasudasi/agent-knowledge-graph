"""MCP server helpers exposing the query layer to MCP clients."""

from __future__ import annotations

from typing import Any

from core.config import load_config
from core.embedding import EmbeddingProviderFactory
from core.graph import Neo4jClient
from core.llm import LLMProviderFactory
from core.query import QueryEngine


def create_mcp_handlers() -> dict[str, Any]:
    """Create MCP tool handlers and lifecycle hooks."""
    cfg = load_config(auto_create=False)
    graph = Neo4jClient(cfg)
    graph.connect()
    embedder = EmbeddingProviderFactory.create(cfg)
    llm = LLMProviderFactory.create(cfg)
    engine = QueryEngine(graph=graph, embedder=embedder, llm=llm)

    def get_graph() -> Neo4jClient:
        return graph

    def close() -> None:
        graph.close()

    async def handle_query(question: str, **_kwargs: Any) -> dict[str, Any]:
        result = engine.nl_query(question)
        return {
            "type": "nl_query",
            "question": question,
            "cypher": result.cypher,
            "results": result.results,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
        }

    async def handle_semantic(query: str, top_k: int = 5, **_kwargs: Any) -> dict[str, Any]:
        result = engine.semantic(query, top_k=top_k)
        return {
            "type": "semantic_search",
            "query": query,
            "results": [
                {
                    "id": node.id,
                    "type": node.type,
                    "label": node.label,
                    "score": result.scores[idx] if result.scores else None,
                }
                for idx, node in enumerate(result.nodes)
            ],
            "execution_time_ms": result.execution_time_ms,
        }

    async def handle_traverse(start_id: str, hops: int = 1, **_kwargs: Any) -> dict[str, Any]:
        result = engine.traverse(start_id, hops=hops)
        return {
            "type": "traverse",
            "start_id": start_id,
            "nodes": [{"id": node.id, "type": node.type, "label": node.label} for node in result.nodes],
            "relationships": [
                {"source": rel.source_id, "target": rel.target_id, "type": rel.type} for rel in result.relationships
            ],
            "execution_time_ms": result.execution_time_ms,
        }

    return {
        "handlers": {
            "kg_query": handle_query,
            "kg_semantic_search": handle_semantic,
            "kg_traverse": handle_traverse,
        },
        "close": close,
        "get_graph": get_graph,
    }
