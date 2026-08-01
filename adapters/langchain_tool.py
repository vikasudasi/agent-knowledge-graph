"""LangChain tool wrappers for the knowledge graph query layer."""

from __future__ import annotations

import json

try:
    from langchain_core.tools import BaseTool

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    BaseTool = object

from core.config import load_config
from core.embedding import EmbeddingProviderFactory
from core.graph import Neo4jClient
from core.llm import LLMProviderFactory
from core.query import QueryEngine


class _QueryEngineMixin:
    """Lazy-load a singleton query engine."""

    _engine: QueryEngine | None = None

    @classmethod
    def _get_engine(cls) -> QueryEngine:
        if cls._engine is None:
            cfg = load_config(auto_create=False)
            graph = Neo4jClient(cfg)
            graph.connect()
            embedder = EmbeddingProviderFactory.create(cfg)
            llm = LLMProviderFactory.create(cfg)
            cls._engine = QueryEngine(graph=graph, embedder=embedder, llm=llm)
        return cls._engine


if HAS_LANGCHAIN:

    class KGQueryTool(_QueryEngineMixin, BaseTool):
        name: str = "kg_query"
        description: str = "Query the knowledge graph using natural language. Returns results and generated Cypher."

        def _run(self, question: str) -> str:
            result = self._get_engine().nl_query(question)
            return json.dumps(
                {
                    "question": question,
                    "cypher": result.cypher,
                    "results": result.results[:10],
                    "error": result.error,
                },
                default=str,
            )

        async def _arun(self, question: str) -> str:
            return self._run(question)

    class KGSemanticSearchTool(_QueryEngineMixin, BaseTool):
        name: str = "kg_semantic_search"
        description: str = "Semantic search across the knowledge graph. Finds resources by meaning."

        def _run(self, query: str, top_k: int = 5) -> str:
            result = self._get_engine().semantic(query, top_k=top_k)
            return json.dumps(
                {
                    "results": [
                        {
                            "id": node.id,
                            "type": node.type,
                            "label": node.label,
                            "score": result.scores[idx] if result.scores else None,
                        }
                        for idx, node in enumerate(result.nodes)
                    ]
                },
                default=str,
            )

        async def _arun(self, query: str, top_k: int = 5) -> str:
            return self._run(query, top_k=top_k)

    class KGTraverseTool(_QueryEngineMixin, BaseTool):
        name: str = "kg_traverse"
        description: str = "Traverse relationships from a node in the knowledge graph."

        def _run(self, start_id: str, hops: int = 1) -> str:
            result = self._get_engine().traverse(start_id, hops=hops)
            return json.dumps(
                {
                    "nodes": [{"id": node.id, "type": node.type, "label": node.label} for node in result.nodes],
                    "relationships": [
                        {"source": rel.source_id, "target": rel.target_id, "type": rel.type}
                        for rel in result.relationships
                    ],
                },
                default=str,
            )

        async def _arun(self, start_id: str, hops: int = 1) -> str:
            return self._run(start_id, hops=hops)

    AVAILABLE_TOOLS = [KGQueryTool, KGSemanticSearchTool, KGTraverseTool]
else:
    AVAILABLE_TOOLS = []
