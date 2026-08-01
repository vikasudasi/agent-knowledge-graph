"""Tests for watch mode CLI commands."""

from __future__ import annotations

from dataclasses import dataclass

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


@dataclass
class _PipelineResult:
    records_processed: int = 1
    errors: int = 0


class _FakePipeline:
    name = "fake-pipeline"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def run(self, context):
        _ = context
        if self.fail:
            raise RuntimeError("pipeline failed")
        return _PipelineResult(records_processed=2)


class _FakeGraph:
    def close(self) -> None:
        return None


class _FakeContext:
    graph = _FakeGraph()


def test_watch_once_success(monkeypatch) -> None:
    from cli import watch as watch_mod

    monkeypatch.setattr(watch_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(watch_mod, "_pipeline_map", lambda: {"fake-pipeline": _FakePipeline()})
    monkeypatch.setattr(watch_mod, "_build_context", lambda _cfg: _FakeContext())

    result = runner.invoke(app, ["watch", "watch", "--once", "--interval", "1"])
    assert result.exit_code == 0
    assert "Watch mode" in result.stdout


def test_watch_invalid_pipeline(monkeypatch) -> None:
    from cli import watch as watch_mod

    monkeypatch.setattr(watch_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(watch_mod, "_pipeline_map", lambda: {"known": _FakePipeline()})

    result = runner.invoke(app, ["watch", "watch", "--once", "--pipeline", "unknown"])
    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_watch_pipeline_error_isolated(monkeypatch) -> None:
    from cli import watch as watch_mod

    monkeypatch.setattr(watch_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(watch_mod, "_pipeline_map", lambda: {"bad": _FakePipeline(fail=True)})
    monkeypatch.setattr(watch_mod, "_build_context", lambda _cfg: _FakeContext())

    result = runner.invoke(app, ["watch", "watch", "--once", "--pipeline", "bad"])
    assert result.exit_code == 0
    assert "bad:" in result.stdout
