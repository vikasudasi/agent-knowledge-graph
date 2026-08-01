"""Tests for kg llm command group."""

from __future__ import annotations

from dataclasses import dataclass

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


@dataclass
class _ExtractResult:
    def model_dump_json(self, indent: int = 2) -> str:
        _ = indent
        return '{"name": "Vik", "role": "architect"}'


class _FakeLLM:
    def __init__(self, fail_generate: bool = False, fail_extract: bool = False) -> None:
        self.fail_generate = fail_generate
        self.fail_extract = fail_extract

    def generate(self, *args, **kwargs) -> str:
        _ = (args, kwargs)
        if self.fail_generate:
            raise RuntimeError("no key")
        return "OK"

    def extract_structured(self, *args, **kwargs):
        _ = (args, kwargs)
        if self.fail_extract:
            raise RuntimeError("extract failed")
        return _ExtractResult()


def test_llm_ping_success(monkeypatch) -> None:
    from cli import llm as llm_mod

    monkeypatch.setattr(llm_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(llm_mod.LLMProviderFactory, "create", lambda _cfg: _FakeLLM())

    result = runner.invoke(app, ["llm", "ping"])
    assert result.exit_code == 0
    assert "LLM Ping" in result.stdout


def test_llm_ping_failure(monkeypatch) -> None:
    from cli import llm as llm_mod

    monkeypatch.setattr(llm_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(llm_mod.LLMProviderFactory, "create", lambda _cfg: _FakeLLM(fail_generate=True))

    result = runner.invoke(app, ["llm", "ping"])
    assert result.exit_code == 1
    assert "LLM ping failed" in result.stdout


def test_llm_extract_success(monkeypatch) -> None:
    from cli import llm as llm_mod

    monkeypatch.setattr(llm_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(llm_mod.LLMProviderFactory, "create", lambda _cfg: _FakeLLM())

    result = runner.invoke(app, ["llm", "extract"])
    assert result.exit_code == 0
    assert "Structured Extraction" in result.stdout


def test_llm_extract_failure(monkeypatch) -> None:
    from cli import llm as llm_mod

    monkeypatch.setattr(llm_mod, "load_config", lambda auto_create=False: object())
    monkeypatch.setattr(llm_mod.LLMProviderFactory, "create", lambda _cfg: _FakeLLM(fail_extract=True))

    result = runner.invoke(app, ["llm", "extract"])
    assert result.exit_code == 1
    assert "Extraction failed" in result.stdout
