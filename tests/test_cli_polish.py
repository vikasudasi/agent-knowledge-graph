"""Tests for CLI polish commands."""

from __future__ import annotations

from dataclasses import dataclass, field

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


@dataclass
class _FakeCheckpoint:
    last_processed_id: str = "cp-1"
    total_processed: int = 1


@dataclass
class _FakePipelineResult:
    records_processed: int = 1
    resources_created: int = 1
    relationships_created: int = 0
    errors: int = 0
    duration_seconds: float = 0.01
    checkpoint: _FakeCheckpoint | None = field(default_factory=_FakeCheckpoint)


class _FakePipeline:
    name = "fake-pipeline"
    version = "1.0"
    description = "test pipeline"

    def run(self, context, full_rebuild: bool = False):
        _ = context
        _ = full_rebuild
        return _FakePipelineResult()


class _FakeGraph:
    def connect(self) -> None:
        return None

    def close(self) -> None:
        return None

    def health_check(self) -> bool:
        return True

    def get_stats(self):
        class _Stats:
            node_count = 3
            relationship_count = 2
            vector_index_ready = True
            last_checkpoints = {"session-ingest": _FakeCheckpoint()}

        return _Stats()

    def traverse(self, node_id: str, hops: int = 1):
        _ = node_id
        _ = hops

        class _Rel:
            source_id = "root"
            target_id = "child"
            type = "mentions"

        class _Result:
            relationships = [_Rel()]

        return _Result()


class _FakeContext:
    graph = _FakeGraph()


def test_build_list_pipelines(monkeypatch) -> None:
    from cli import build as build_mod

    monkeypatch.setattr(build_mod, "_registered_pipelines", lambda: [_FakePipeline()])
    result = runner.invoke(app, ["build", "list-pipelines"])
    assert result.exit_code == 0
    assert "fake-pipeline" in result.stdout


def test_build_run_unknown_pipeline(monkeypatch) -> None:
    from cli import build as build_mod

    monkeypatch.setattr(build_mod, "_registered_pipelines", lambda: [_FakePipeline()])
    monkeypatch.setattr(build_mod.PipelineRegistry, "get", lambda _name: None)
    result = runner.invoke(app, ["build", "run", "nonexistent"])
    assert result.exit_code == 1
    assert "not found" in result.stdout.lower()


def test_build_run_all(monkeypatch) -> None:
    from cli import build as build_mod

    monkeypatch.setattr(build_mod, "_registered_pipelines", lambda: [_FakePipeline()])
    monkeypatch.setattr(build_mod, "_build_context", lambda *_args, **_kwargs: _FakeContext())
    result = runner.invoke(app, ["build", "run", "all", "--dry-run"])
    assert result.exit_code == 0
    assert "Running 1 pipeline" in result.stdout


def test_status_status(monkeypatch) -> None:
    from cli import status as status_mod

    monkeypatch.setattr(status_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(status_mod, "Neo4jClient", lambda _cfg: _FakeGraph())
    result = runner.invoke(app, ["status", "status"])
    assert result.exit_code == 0
    assert "Graph Statistics" in result.stdout


def test_status_health(monkeypatch) -> None:
    from cli import status as status_mod

    monkeypatch.setattr(status_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(status_mod, "Neo4jClient", lambda _cfg: _FakeGraph())
    result = runner.invoke(app, ["status", "health"])
    assert result.exit_code == 0
    assert "healthy" in result.stdout


def test_visualize_tree(monkeypatch) -> None:
    from cli import visualize as visualize_mod

    monkeypatch.setattr(visualize_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(visualize_mod, "Neo4jClient", lambda _cfg: _FakeGraph())
    result = runner.invoke(app, ["visualize", "tree", "root", "--depth", "1"])
    assert result.exit_code == 0
    assert "root" in result.stdout


def test_watch_once(monkeypatch) -> None:
    from cli import watch as watch_mod

    monkeypatch.setattr(watch_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(watch_mod, "_pipeline_map", lambda: {"fake-pipeline": _FakePipeline()})
    monkeypatch.setattr(watch_mod, "_build_context", lambda _cfg: _FakeContext())
    result = runner.invoke(app, ["watch", "watch", "--once", "--interval", "1"])
    assert result.exit_code == 0
    assert "Watch mode" in result.stdout
