"""Tests for the query engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.models import QueryResult, Resource
from core.query import QueryEngine


@pytest.fixture
def engine() -> QueryEngine:
    return QueryEngine(
        graph=MagicMock(),
        embedder=MagicMock(),
        llm=MagicMock(),
    )


class TestSemanticSearch:
    def test_semantic_returns_results(self, engine: QueryEngine) -> None:
        engine._embedder.embed_query.return_value = [0.1] * 384
        engine._graph.vector_search.return_value = QueryResult(
            nodes=[Resource(id="r1", type="session", label="Test")],
            scores=[0.95],
        )
        result = engine.semantic("test query", top_k=5)
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "r1"

    def test_semantic_no_embedder(self, engine: QueryEngine) -> None:
        engine._embedder = None
        with pytest.raises(RuntimeError, match="embedder"):
            engine.semantic("test")

    def test_semantic_with_type_filter(self, engine: QueryEngine) -> None:
        engine._embedder.embed_query.return_value = [0.1] * 384
        engine._graph.vector_search.return_value = QueryResult()
        engine.semantic("test", type_filter="session")
        engine._graph.vector_search.assert_called_with(
            [0.1] * 384,
            top_k=10,
            type_filter="session",
        )


class TestTraverse:
    def test_traverse_delegates(self, engine: QueryEngine) -> None:
        engine._graph.traverse.return_value = QueryResult()
        engine.traverse("node-1", hops=2, direction="outgoing")
        engine._graph.traverse.assert_called_with(
            "node-1",
            hops=2,
            rel_types=None,
            direction="outgoing",
        )


class TestHybridSearch:
    def test_hybrid(self, engine: QueryEngine) -> None:
        engine._embedder.embed_query.return_value = [0.1] * 384
        engine._graph.hybrid_search.return_value = QueryResult()
        result = engine.hybrid("test", cypher_filter="node.type='session'", top_k=5)
        assert result is not None
        engine._graph.hybrid_search.assert_called_once()

    def test_hybrid_no_embedder(self, engine: QueryEngine) -> None:
        engine._embedder = None
        with pytest.raises(RuntimeError):
            engine.hybrid("test")


class TestNLQuery:
    def test_nl_query_generates_and_executes(self, engine: QueryEngine) -> None:
        engine._llm.generate.return_value = "MATCH (r:Resource) RETURN r LIMIT 5"
        engine._graph.run_cypher.return_value = [{"id": "r1", "label": "Test"}]

        result = engine.nl_query("Show me resources")
        assert result.cypher == "MATCH (r:Resource) RETURN r LIMIT 5"
        assert len(result.results) == 1
        assert result.error == ""

    def test_nl_query_handles_markdown_fences(self, engine: QueryEngine) -> None:
        engine._llm.generate.return_value = "```cypher\nMATCH (r:Resource) RETURN r\n```"
        engine._graph.run_cypher.return_value = []

        result = engine.nl_query("Show resources")
        assert "```" not in result.cypher

    def test_nl_query_empty_result(self, engine: QueryEngine) -> None:
        engine._llm.generate.return_value = ""
        result = engine.nl_query("test")
        assert "empty" in result.error

    def test_nl_query_no_llm(self, engine: QueryEngine) -> None:
        engine._llm = None
        result = engine.nl_query("test")
        assert "LLM" in result.error

    def test_cypher_execution_error(self, engine: QueryEngine) -> None:
        engine._llm.generate.return_value = "MATCH (r:Resource) RETURN r"
        engine._graph.run_cypher.side_effect = Exception("Syntax error")

        result = engine.nl_query("test")
        assert "Syntax error" in result.error

    def test_explain(self, engine: QueryEngine) -> None:
        engine._llm.generate.return_value = "This query finds all sessions."
        result = engine.explain("MATCH (s:Resource {type:'session'}) RETURN s")
        assert "sessions" in result


class TestQueryEngineInit:
    def test_init_without_optional(self) -> None:
        engine = QueryEngine(graph=MagicMock())
        assert engine._embedder is None
        assert engine._llm is None
