"""Tests for visualize CLI commands."""

from __future__ import annotations

from dataclasses import dataclass

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


@dataclass
class _Stats:
    node_count: int = 20
    relationship_count: int = 30
    vector_index_ready: bool = True


class _FakeTraverseResult:
    def __init__(self) -> None:
        self.relationships = [type("Rel", (), {"source_id": "root-1", "target_id": "child-1", "type": "mentions"})()]


class _FakeGraph:
    def connect(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_stats(self) -> _Stats:
        return _Stats()

    def traverse(self, node_id: str, hops: int = 1):
        _ = (node_id, hops)
        return _FakeTraverseResult()


def test_visualize_tree_overview(monkeypatch) -> None:
    from cli import visualize as visualize_mod

    monkeypatch.setattr(visualize_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(visualize_mod, "Neo4jClient", lambda _cfg: _FakeGraph())

    result = runner.invoke(app, ["visualize", "tree"])
    assert result.exit_code == 0
    assert "Knowledge Graph Overview" in result.stdout


def test_visualize_tree_with_root(monkeypatch) -> None:
    from cli import visualize as visualize_mod

    monkeypatch.setattr(visualize_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(visualize_mod, "Neo4jClient", lambda _cfg: _FakeGraph())

    result = runner.invoke(app, ["visualize", "tree", "root-1", "--depth", "1"])
    assert result.exit_code == 0
    assert "root-1" in result.stdout


def test_visualize_tree_connection_failure(monkeypatch) -> None:
    from cli import visualize as visualize_mod

    class _BrokenGraph(_FakeGraph):
        def connect(self) -> None:
            raise RuntimeError("cannot connect")

    monkeypatch.setattr(visualize_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(visualize_mod, "Neo4jClient", lambda _cfg: _BrokenGraph())

    result = runner.invoke(app, ["visualize", "tree"])
    assert result.exit_code == 1
    assert "Visualization failed" in result.stdout
