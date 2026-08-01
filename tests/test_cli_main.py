"""Tests for top-level CLI behavior and docker subgroup."""

from __future__ import annotations

import subprocess

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "agent-knowledge-graph" in result.stdout.lower()


def test_unknown_command() -> None:
    result = runner.invoke(app, ["does-not-exist"])
    assert result.exit_code != 0


def test_docker_up_success(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        _ = (args, kwargs)
        return subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    result = runner.invoke(app, ["docker", "up"])
    assert result.exit_code == 0
    assert "started" in result.stdout.lower()


def test_docker_up_missing_binary(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        _ = (args, kwargs)
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr("subprocess.run", _fake_run)

    result = runner.invoke(app, ["docker", "up"])
    assert result.exit_code == 1


def test_docker_down_failure(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        _ = (args, kwargs)
        return subprocess.CompletedProcess(args=["docker"], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr("subprocess.run", _fake_run)

    result = runner.invoke(app, ["docker", "down"])
    assert result.exit_code == 1
    assert "failed" in result.stdout.lower()


def test_docker_status(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        _ = (args, kwargs)
        return subprocess.CompletedProcess(
            args=["docker"],
            returncode=0,
            stdout="NAME STATUS PORTS\nneo4j running 7687",
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", _fake_run)

    result = runner.invoke(app, ["docker", "status"])
    assert result.exit_code == 0
    assert "neo4j" in result.stdout.lower()


def test_docker_reset_success(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        _ = (args, kwargs)
        return subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    result = runner.invoke(app, ["docker", "reset"])
    assert result.exit_code == 0
    assert "volumes removed" in result.stdout.lower()
