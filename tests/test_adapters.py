"""Tests for agent adapters."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from adapters.hermes_plugin import KnowledgeGraphPlugin
from adapters.mcp_server import create_mcp_handlers


@dataclass
class _FakeNode:
    id: str = "n1"
    type: str = "concept"
    label: str = "Node"


@dataclass
class _FakeRelationship:
    source_id: str = "n1"
    target_id: str = "n2"
    type: str = "mentions"


@dataclass
class _FakeQueryResult:
    nodes: list[_FakeNode] = field(default_factory=lambda: [_FakeNode()])
    relationships: list[_FakeRelationship] = field(default_factory=lambda: [_FakeRelationship()])
    scores: list[float] = field(default_factory=lambda: [0.9])
    execution_time_ms: float = 1.0


@dataclass
class _FakeNLQueryResult:
    cypher: str = "RETURN 1"
    results: list[dict] = field(default_factory=lambda: [{"ok": True}])
    error: str = ""
    execution_time_ms: float = 1.0


class _FakeGraph:
    def connect(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_stats(self):
        class _Stats:
            node_count = 10
            relationship_count = 12
            vector_index_ready = True
            last_checkpoints = {}

        return _Stats()


class _FakeEngine:
    def nl_query(self, _question: str) -> _FakeNLQueryResult:
        return _FakeNLQueryResult()

    def semantic(self, _query: str, top_k: int = 5) -> _FakeQueryResult:
        _ = top_k
        return _FakeQueryResult()

    def traverse(self, _start_id: str, hops: int = 1) -> _FakeQueryResult:
        _ = hops
        return _FakeQueryResult()


def test_plugin_name() -> None:
    plugin = KnowledgeGraphPlugin()
    assert plugin.name == "knowledge-graph"
    assert plugin.description


def test_plugin_tools_exist() -> None:
    plugin = KnowledgeGraphPlugin()
    assert hasattr(plugin, "query")
    assert hasattr(plugin, "semantic_search")
    assert hasattr(plugin, "traverse")
    assert hasattr(plugin, "stats")


def test_create_mcp_handlers(monkeypatch) -> None:
    from adapters import mcp_server as mod

    monkeypatch.setattr(mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(mod, "Neo4jClient", lambda _cfg: _FakeGraph())
    monkeypatch.setattr(mod.EmbeddingProviderFactory, "create", lambda _cfg: object())
    monkeypatch.setattr(mod.LLMProviderFactory, "create", lambda _cfg: object())
    monkeypatch.setattr(mod, "QueryEngine", lambda **_kwargs: _FakeEngine())

    handlers = create_mcp_handlers()
    assert "handlers" in handlers
    assert "close" in handlers
    assert "kg_query" in handlers["handlers"]
    assert "kg_semantic_search" in handlers["handlers"]
    assert "kg_traverse" in handlers["handlers"]
    for handler in handlers["handlers"].values():
        assert asyncio.iscoroutinefunction(handler)
    handlers["close"]()


def test_imports_without_langchain() -> None:
    from adapters.langchain_tool import AVAILABLE_TOOLS

    assert isinstance(AVAILABLE_TOOLS, list)
