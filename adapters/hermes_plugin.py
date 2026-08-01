"""Hermes agent plugin — adds kg knowledge commands to Hermes."""

from __future__ import annotations

from typing import Any

try:
    from hermes_plugin_base import HermesPlugin, plugin_tool
except ImportError:
    HermesPlugin = object

    def plugin_tool(**_kwargs):
        def _decorator(fn):
            return fn

        return _decorator


class KnowledgeGraphPlugin(HermesPlugin):
    """Hermes plugin for knowledge graph operations."""

    name = "knowledge-graph"
    description = "Query and manage the agent knowledge graph"

    def __init__(self) -> None:
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from core.config import load_config
            from core.embedding import EmbeddingProviderFactory
            from core.graph import Neo4jClient
            from core.llm import LLMProviderFactory
            from core.query import QueryEngine

            cfg = load_config(auto_create=False)
            graph = Neo4jClient(cfg)
            graph.connect()
            embedder = EmbeddingProviderFactory.create(cfg)
            llm = LLMProviderFactory.create(cfg)
            self._engine = QueryEngine(graph=graph, embedder=embedder, llm=llm)
        return self._engine

    @plugin_tool(name="kg_query", description="Query the knowledge graph using natural language")
    def query(self, question: str) -> dict[str, Any]:
        """Ask a natural language question about the knowledge graph."""
        result = self.engine.nl_query(question)
        return {
            "question": question,
            "cypher": result.cypher,
            "results": result.results,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
        }

    @plugin_tool(name="kg_semantic_search", description="Semantic search across graph resources")
    def semantic_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Find resources semantically similar to the query string."""
        result = self.engine.semantic(query, top_k=top_k)
        return {
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

    @plugin_tool(name="kg_traverse", description="Traverse relationships from a graph node")
    def traverse(self, start_id: str, hops: int = 1) -> dict[str, Any]:
        """Traverse the graph from a starting node."""
        result = self.engine.traverse(start_id, hops=hops)
        return {
            "start_id": start_id,
            "nodes": [{"id": node.id, "type": node.type, "label": node.label} for node in result.nodes],
            "relationships": [
                {"source": rel.source_id, "target": rel.target_id, "type": rel.type}
                for rel in result.relationships
            ],
            "execution_time_ms": result.execution_time_ms,
        }

    @plugin_tool(name="kg_stats", description="Get knowledge graph statistics")
    def stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        from core.config import load_config
        from core.graph import Neo4jClient

        cfg = load_config(auto_create=False)
        graph = Neo4jClient(cfg)
        graph.connect()
        try:
            stats = graph.get_stats()
        finally:
            graph.close()

        return {
            "node_count": stats.node_count,
            "relationship_count": stats.relationship_count,
            "vector_index_ready": stats.vector_index_ready,
            "checkpoints": {
                name: {
                    "last_processed_id": cp.last_processed_id,
                    "total_processed": cp.total_processed,
                }
                for name, cp in stats.last_checkpoints.items()
            },
        }
