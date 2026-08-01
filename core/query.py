"""Query engine — semantic, traversal, hybrid, and NL→Cypher."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.embedding import EmbeddingProvider
from core.graph import Neo4jClient
from core.llm import LLMClient
from core.models import QueryResult as KGQueryResult

NL_CYPHER_SYSTEM_PROMPT = """You are a Neo4j Cypher query generator for a knowledge graph.

SCHEMA:
- Nodes: (:Resource {id, type, label, properties, embedding})
- Relationships: (:Resource)-[:RELATES {type, weight, context}]->(:Resource)
- Resource types: session, person, project, tool, concept, file, task, skill, artifact
- Relationship types: mentions, produces, uses, decides, references, blocks, resolves, assigns

Generate ONLY the Cypher query. No explanations. Return valid Cypher only.

GUIDELINES:
- Use MATCH with labels for filtering: MATCH (r:Resource {type: 'person'})
- For text search: WHERE r.label CONTAINS 'term' OR r.properties CONTAINS 'term'
- For relationships: MATCH (a:Resource {id: $id})-[:RELATES]->(b:Resource)
- Always RETURN distinct results
- Limit results to 20 unless the user asks for more
- Use parameterized queries ($param) for user input"""


@dataclass
class NLQueryResult:
    """Result of a natural language query against the knowledge graph."""

    cypher: str = ""
    explanation: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    execution_time_ms: float = 0.0


class QueryEngine:
    """Main query interface — semantic, traversal, hybrid, NL→Cypher."""

    def __init__(
        self,
        graph: Neo4jClient,
        embedder: EmbeddingProvider | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self._graph = graph
        self._embedder = embedder
        self._llm = llm

    def semantic(self, query: str, top_k: int = 10, type_filter: str | None = None) -> KGQueryResult:
        """Search by semantic meaning — embed query, vector search."""
        if not self._embedder:
            raise RuntimeError("QueryEngine needs embedder for semantic search")

        t0 = time.monotonic()
        query_vec = self._embedder.embed_query(query)
        result = self._graph.vector_search(query_vec, top_k=top_k, type_filter=type_filter)
        result.execution_time_ms = (time.monotonic() - t0) * 1000
        return result

    def traverse(
        self,
        start_id: str,
        hops: int = 1,
        rel_types: list[str] | None = None,
        direction: str = "both",
    ) -> KGQueryResult:
        """Graph traversal from a node."""
        return self._graph.traverse(start_id, hops=hops, rel_types=rel_types, direction=direction)

    def hybrid(
        self,
        query: str,
        cypher_filter: str = "",
        top_k: int = 10,
    ) -> KGQueryResult:
        """Vector search + optional Cypher pre-filter."""
        if not self._embedder:
            raise RuntimeError("QueryEngine needs embedder for hybrid search")

        t0 = time.monotonic()
        query_vec = self._embedder.embed_query(query)
        result = self._graph.hybrid_search(query_vec, cypher_filter=cypher_filter, top_k=top_k)
        result.execution_time_ms = (time.monotonic() - t0) * 1000
        return result

    def nl_query(self, question: str) -> NLQueryResult:
        """Natural language -> Cypher -> execution."""
        if not self._llm:
            return NLQueryResult(error="QueryEngine needs LLM for NL->Cypher")

        t0 = time.monotonic()
        try:
            cypher = self._llm.generate(
                prompt=f"Convert this question to a Cypher query:\n\n{question}",
                system_prompt=NL_CYPHER_SYSTEM_PROMPT,
                model=self._llm.config.query_model,
                temperature=0.1,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            return NLQueryResult(error=f"LLM translation failed: {exc}", execution_time_ms=elapsed)

        cypher = self._strip_fences(cypher)
        if not cypher:
            elapsed = (time.monotonic() - t0) * 1000
            return NLQueryResult(error="LLM returned empty Cypher", execution_time_ms=elapsed)

        try:
            raw_results = self._graph.run_cypher(cypher)
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            return NLQueryResult(
                cypher=cypher,
                error=f"Cypher execution failed: {exc}",
                execution_time_ms=elapsed,
            )

        elapsed = (time.monotonic() - t0) * 1000
        return NLQueryResult(
            cypher=cypher,
            explanation=f"Generated query returned {len(raw_results)} results",
            results=[dict(row) for row in raw_results],
            execution_time_ms=elapsed,
        )

    def explain(self, cypher: str) -> str:
        """Generate a plain-English explanation of a Cypher query."""
        if not self._llm:
            return "LLM not configured"
        try:
            return self._llm.generate(
                prompt=f"Explain this Neo4j Cypher query in plain English:\n\n{cypher}",
                system_prompt="You are a Cypher expert. Provide clear, concise explanations.",
                temperature=0.3,
            )
        except Exception as exc:
            return f"Explanation failed: {exc}"

    @staticmethod
    def _strip_fences(cypher: str) -> str:
        text = cypher.strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        if lines and lines[0].strip().lower() in {"cypher", "sql"}:
            lines = lines[1:]
        return "\n".join(lines).strip()
