"""Tests for kg init command behavior."""

from __future__ import annotations

from dataclasses import dataclass

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


@dataclass
class _Stats:
    node_count: int = 4
    relationship_count: int = 2
    vector_index_ready: bool = True


class _FakeClient:
    def __init__(self, _cfg) -> None:
        self.drop_schema_called = False
        self.initialize_schema_called = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)

    def drop_schema(self) -> None:
        self.drop_schema_called = True

    def initialize_schema(self) -> None:
        self.initialize_schema_called = True

    def get_stats(self) -> _Stats:
        return _Stats()


def test_cli_init_invokes_run_init(monkeypatch) -> None:
    calls: list[tuple[bool, bool]] = []

    def _fake_run_init(reset: bool = False, with_docker: bool = False) -> None:
        calls.append((reset, with_docker))

    monkeypatch.setattr("cli.main.run_init", _fake_run_init)

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert calls == [(False, False)]


def test_cli_init_flags(monkeypatch) -> None:
    calls: list[tuple[bool, bool]] = []

    def _fake_run_init(reset: bool = False, with_docker: bool = False) -> None:
        calls.append((reset, with_docker))

    monkeypatch.setattr("cli.main.run_init", _fake_run_init)

    result = runner.invoke(app, ["init", "--reset", "--with-docker"])
    assert result.exit_code == 0
    assert calls == [(True, True)]


def test_run_init_reset_calls_drop_schema(monkeypatch) -> None:
    from cli import init as init_mod

    fake_client = _FakeClient(object())
    monkeypatch.setattr(init_mod, "load_config", lambda auto_create=True: object())
    monkeypatch.setattr(init_mod, "Neo4jClient", lambda _cfg: fake_client)

    init_mod.run_init(reset=True, with_docker=False)

    assert fake_client.drop_schema_called is True
    assert fake_client.initialize_schema_called is True


def test_run_init_with_docker_attempt(monkeypatch) -> None:
    from cli import init as init_mod

    monkeypatch.setattr(init_mod, "_docker_compose_up", lambda: True)
    monkeypatch.setattr(init_mod, "load_config", lambda auto_create=True: object())
    monkeypatch.setattr(init_mod, "Neo4jClient", lambda _cfg: _FakeClient(_cfg))

    init_mod.run_init(reset=False, with_docker=True)


def test_run_init_exits_on_config_error(monkeypatch) -> None:
    from cli import init as init_mod

    def _boom(auto_create=True):
        _ = auto_create
        raise RuntimeError("broken config")

    monkeypatch.setattr(init_mod, "load_config", _boom)

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 1
    assert "Config error" in result.stdout
