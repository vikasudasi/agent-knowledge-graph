"""Tests for the config system."""

from __future__ import annotations

import pytest
import yaml

from core.config import (
    KGConfig,
    _default_config_dict,
    _write_default_config,
    load_config,
)


class TestConfigDefaults:
    def test_default_llm_provider(self) -> None:
        cfg = KGConfig()
        assert cfg.llm.provider == "openrouter"
        assert cfg.llm.default_model == "deepseek/deepseek-v4-flash-0731"

    def test_default_embedding(self) -> None:
        cfg = KGConfig()
        assert cfg.embedding.provider == "local"
        assert cfg.embedding.local_model == "all-MiniLM-L6-v2"

    def test_default_neo4j(self) -> None:
        cfg = KGConfig()
        assert cfg.neo4j.uri == "bolt://localhost:7687"


class TestEnvOverrides:
    def test_env_override_llm_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KG_LLM_API_KEY", "sk-test-123")
        cfg = load_config(auto_create=False)
        assert cfg.llm.api_key == "sk-test-123"

    def test_env_override_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KG_VERBOSE", "true")
        cfg = load_config(auto_create=False)
        assert cfg.verbose is True

    def test_env_override_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KG_EMBEDDING_DIMENSION", "768")
        cfg = load_config(auto_create=False)
        assert cfg.embedding.dimension == 768

    def test_env_override_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KG_NEO4J_URI", "bolt://remote:7687")
        cfg = load_config(auto_create=False)
        assert cfg.neo4j.uri == "bolt://remote:7687"


class TestConfigFile:
    def test_write_and_load_default(self, tmp_path) -> None:
        config_dir = tmp_path / ".config" / "agent-knowledge-graph"
        config_path = config_dir / "config.yaml"
        _write_default_config(config_path)
        assert config_path.exists()

        cfg = load_config(config_path=config_path, auto_create=False)
        assert cfg.llm.provider == "openrouter"

    def test_yaml_overrides_default(self, tmp_path) -> None:
        config_path = tmp_path / "test-config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = _default_config_dict()
        data["llm"]["provider"] = "openai"
        data["llm"]["api_key"] = "sk-from-file"
        with config_path.open("w", encoding="utf-8") as file:
            yaml.dump(data, file)

        cfg = load_config(config_path=config_path, auto_create=False)
        assert cfg.llm.provider == "openai"

    def test_env_overrides_file(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars should override file values."""

        config_path = tmp_path / "test-config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = _default_config_dict()
        data["neo4j"]["uri"] = "bolt://file-config:7687"
        with config_path.open("w", encoding="utf-8") as file:
            yaml.dump(data, file)

        monkeypatch.setenv("KG_NEO4J_URI", "bolt://env-override:7687")
        cfg = load_config(config_path=config_path, auto_create=False)
        assert cfg.neo4j.uri == "bolt://env-override:7687"

    def test_validation_error_bad_provider(self, tmp_path) -> None:
        config_path = tmp_path / "bad-config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = _default_config_dict()
        data["llm"]["provider"] = "nonexistent"
        with config_path.open("w", encoding="utf-8") as file:
            yaml.dump(data, file)

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            load_config(config_path=config_path, auto_create=False)

    def test_empty_file_uses_defaults(self, tmp_path) -> None:
        config_path = tmp_path / "empty.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as file:
            yaml.dump({}, file)

        cfg = load_config(config_path=config_path, auto_create=False)
        assert cfg.llm.default_model == "deepseek/deepseek-v4-flash-0731"
