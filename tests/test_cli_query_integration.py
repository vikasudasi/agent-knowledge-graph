"""Integration-style tests for query CLI commands with mocked engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


@dataclass
class _Node:
    id: str
    type: str
    label: str


@dataclass
class _Relationship:
    source_id: str
    target_id: str
    type: str


@dataclass
class _QueryResult:
    nodes: list[_Node] = field(default_factory=list)
    relationships: list[_Relationship] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    execution_time_ms: float = 12.0


@dataclass
class _NLQueryResult:
    cypher: str = "MATCH (n) RETURN n LIMIT 1"
    results: list[dict] = field(default_factory=lambda: [{"id": "n1", "label": "Node 1"}])
    error: str = ""
    execution_time_ms: float = 7.0


class _FakeEngine:
    def semantic(self, query: str, top_k: int = 10, type_filter: str | None = None):
        _ = (query, top_k, type_filter)
        return _QueryResult(nodes=[_Node(id="r1", type="concept", label="retry policy")], scores=[0.98])

    def traverse(self, start_id: str, hops: int = 1, direction: str = "both"):
        _ = (start_id, hops, direction)
        return _QueryResult(
            nodes=[_Node(id="root-1", type="session", label="Root")],
            relationships=[_Relationship(source_id="root-1", target_id="child-1", type="mentions")],
        )

    def nl_query(self, question: str):
        _ = question
        return _NLQueryResult()


def test_query_semantic_output(monkeypatch) -> None:
    from cli import query as query_mod

    monkeypatch.setattr(query_mod, "_get_engine", lambda: _FakeEngine())

    result = runner.invoke(app, ["query", "semantic", "test", "--top", "1"])
    assert result.exit_code == 0
    assert "Semantic Search" in result.stdout
    assert "retry policy" in result.stdout


def test_query_traverse_output(monkeypatch) -> None:
    from cli import query as query_mod

    monkeypatch.setattr(query_mod, "_get_engine", lambda: _FakeEngine())

    result = runner.invoke(app, ["query", "traverse", "root-1"])
    assert result.exit_code == 0
    assert "Traversal" in result.stdout
    assert "mentions" in result.stdout


def test_query_ask_output(monkeypatch) -> None:
    from cli import query as query_mod

    monkeypatch.setattr(query_mod, "_get_engine", lambda: _FakeEngine())

    result = runner.invoke(app, ["query", "ask", "What do I know?"])
    assert result.exit_code == 0
    assert "Generated Cypher" in result.stdout
    assert "Results:" in result.stdout


def test_query_explain_output(monkeypatch) -> None:
    from cli import query as query_mod

    class _Cfg:
        class llm:
            api_key = "test-key"

    class _ExplainEngine:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def explain(self, cypher: str) -> str:
            return f"Explanation for: {cypher}"

    monkeypatch.setattr(query_mod, "load_config", lambda auto_create=False: _Cfg())
    monkeypatch.setattr(query_mod.LLMProviderFactory, "create", lambda _cfg: object())
    monkeypatch.setattr(query_mod, "Neo4jClient", lambda _cfg: object())
    monkeypatch.setattr(query_mod, "QueryEngine", _ExplainEngine)

    result = runner.invoke(app, ["query", "explain", "MATCH (n) RETURN n"])
    assert result.exit_code == 0
    assert "Cypher Explanation" in result.stdout


def test_query_explain_requires_llm(monkeypatch) -> None:
    from cli import query as query_mod

    class _Cfg:
        class llm:
            api_key = ""

    monkeypatch.setattr(query_mod, "load_config", lambda auto_create=False: _Cfg())

    result = runner.invoke(app, ["query", "explain", "MATCH (n) RETURN n"])
    assert result.exit_code == 1
    assert "LLM not configured" in result.stdout
