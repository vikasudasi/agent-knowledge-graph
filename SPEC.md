# agent-knowledge-graph — Config System (Task 2)

## What to Build

Implement the XDG-based configuration system for agent-knowledge-graph. This is the core `core/config.py` module that every other layer depends on. It must support YAML config files, environment variable overrides, schema validation, and default generation.

## Files to Modify

- `core/config.py` — full implementation (replace the placeholder)
- `core/models.py` — add Pydantic config models
- `core/__init__.py` — export the public API
- `cli/init.py` — wire up default config generation
- `tests/test_config.py` — comprehensive test suite

## Detailed Implementation

### 1. Config Path Resolution (core/config.py)

```python
"""Configuration management for agent-knowledge-graph."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "openrouter"
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "deepseek/deepseek-v4-flash-0731"
    extraction_model: str = "deepseek/deepseek-v4-flash-0731"
    query_model: str = "deepseek/deepseek-v4-flash-0731"
    max_retries: int = 3
    timeout: int = 60


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""
    provider: str = "local"
    local_model: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    # Remote embedding settings (only used when provider != "local")
    api_key: str = ""
    model: str = "text-embedding-3-small"


class Neo4jConfig(BaseModel):
    """Neo4j connection configuration."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_pool_size: int = 10
    connection_timeout: int = 30


class PipelineConfig(BaseModel):
    """Pipeline-specific configuration."""
    session_ingest_batch_size: int = 10
    session_ingest_max_workers: int = 4
    dry_run: bool = False


class StorageConfig(BaseModel):
    """Storage and data directory configuration."""
    data_dir: str = "~/.local/share/agent-knowledge-graph"


class KGConfig(BaseModel):
    """Root configuration model."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    pipelines: PipelineConfig = Field(default_factory=PipelineConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    verbose: bool = False

    @field_validator("llm")
    @classmethod
    def validate_llm(cls, v: LLMConfig) -> LLMConfig:
        if v.provider not in ("openrouter", "openai", "anthropic", "custom"):
            raise ValueError(f"Unknown LLM provider: {v.provider}")
        return v

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: EmbeddingConfig) -> EmbeddingConfig:
        if v.provider not in ("local", "openrouter", "openai"):
            raise ValueError(f"Unknown embedding provider: {v.provider}")
        if v.dimension not in (384, 768, 1024, 1536, 3072):
            raise ValueError(f"Unsupported embedding dimension: {v.dimension}")
        return v
```

### 2. Config Loader

```python
# In core/config.py, after the models

def _xdg_config_home() -> Path:
    """Return the XDG config directory."""
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))


def _default_config_paths() -> list[Path]:
    """Return config paths in priority order (first found wins)."""
    return [
        _xdg_config_home() / "agent-knowledge-graph" / "config.yaml",
        Path.home() / ".agent-knowledge-graph.yaml",
    ]


ENV_PREFIX = "KG_"
ENV_MAP: dict[str, str] = {
    # kg config key path -> env var
    "llm.api_key": "KG_LLM_API_KEY",
    "llm.base_url": "KG_LLM_BASE_URL",
    "llm.provider": "KG_LLM_PROVIDER",
    "llm.default_model": "KG_LLM_DEFAULT_MODEL",
    "llm.extraction_model": "KG_LLM_EXTRACTION_MODEL",
    "llm.query_model": "KG_LLM_QUERY_MODEL",
    "embedding.provider": "KG_EMBEDDING_PROVIDER",
    "embedding.dimension": "KG_EMBEDDING_DIMENSION",
    "neo4j.uri": "KG_NEO4J_URI",
    "neo4j.user": "KG_NEO4J_USER",
    "neo4j.password": "KG_NEO4J_PASSWORD",
    "neo4j.database": "KG_NEO4J_DATABASE",
    "storage.data_dir": "KG_DATA_DIR",
    "verbose": "KG_VERBOSE",
    "pipelines.dry_run": "KG_DRY_RUN",
}


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Override config values from KG_* environment variables."""
    for key_path, env_var in ENV_MAP.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Navigate nested dict by dotted path
            parts = key_path.split(".")
            target = config
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            # Coerce types: bool, int
            part = parts[-1]
            if value.lower() in ("true", "1", "yes"):
                target[part] = True
            elif value.lower() in ("false", "0", "no"):
                target[part] = False
            elif value.isdigit():
                target[part] = int(value)
            else:
                target[part] = value
    return config


def _default_config_dict() -> dict[str, Any]:
    """Return the default config as a dict."""
    return KGConfig().model_dump()


def _write_default_config(path: Path) -> Path:
    """Write default config to the given path. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(_default_config_dict(), f, default_flow_style=False, sort_keys=False)
    return path


def load_config(
    config_path: Path | None = None,
    auto_create: bool = True,
) -> KGConfig:
    """Load config from file + env overrides.

    Resolution order (first wins):
    1. Explicit config_path argument
    2. XDG config home (~/.config/agent-knowledge-graph/config.yaml)
    3. Home fallback (~/.agent-knowledge-graph.yaml)
    4. KG_* environment variables (always applied on top)
    5. Built-in defaults

    If auto_create is True and no config file exists, writes the default
    to the XDG config path.
    """
    # Determine which config file to load
    config_file: Path | None = None
    
    if config_path is not None:
        if config_path.exists():
            config_file = config_path
    else:
        for path in _default_config_paths():
            if path.exists():
                config_file = path
                break
    
    # Load from file or start with defaults
    if config_file:
        with open(config_file) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}
        if auto_create:
            config_file = _default_config_paths()[0]
            _write_default_config(config_file)
    
    # Apply env overrides
    raw = _apply_env_overrides(raw)
    
    # Validate and return typed config
    return KGConfig(**raw)


__all__ = [
    "KGConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "Neo4jConfig",
    "PipelineConfig",
    "StorageConfig",
    "load_config",
]
```

### 3. Wire into CLI (cli/init.py)

```python
"""KG init command — bootstrap configuration and Neo4j."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from core.config import load_config


console = Console()


def run_init(reset: bool = False) -> None:
    """Initialize config and Neo4j.
    
    If reset is True, overwrite existing config with defaults.
    """
    try:
        cfg = load_config(auto_create=True)
        console.print(Panel.fit(
            f"[bold green]✓[/] Config loaded from {cfg._config_path}\n"
            f"  LLM provider: {cfg.llm.provider}\n"
            f"  Embedding: {cfg.embedding.provider} ({cfg.embedding.local_model})\n"
            f"  Neo4j: {cfg.neo4j.uri}",
            title="agent-knowledge-graph",
        ))
    except Exception as e:
        console.print(f"[bold red]✗[/] Config error: {e}")
        raise typer.Exit(1)
```

Note: `cfg._config_path` isn't on the current model — add a `_config_path: Path | None = None` field to `KGConfig` that gets set by `load_config`.

### 4. Update core/__init__.py

```python
"""Core modules for agent-knowledge-graph."""

from core.config import KGConfig, LLMConfig, EmbeddingConfig, Neo4jConfig, load_config

__all__ = [
    "KGConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "Neo4jConfig",
    "load_config",
]
```

### 5. Update cli/main.py

Wire the init command to call `run_init()` instead of printing a placeholder:

```python
from cli.init import run_init

@app.command()
def init(
    reset: Annotated[bool, typer.Option("--reset", help="Reset Neo4j and re-initialize")] = False,
) -> None:
    """Initialize Neo4j and create schema."""
    run_init(reset=reset)
```

### 6. Test Suite (tests/test_config.py)

Replace the existing placeholder test with comprehensive tests:

```python
"""Tests for the config system."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from core.config import (
    KGConfig,
    LLMConfig,
    load_config,
    _default_config_paths,
    _apply_env_overrides,
    _default_config_dict,
    _write_default_config,
)


class TestConfigDefaults:
    def test_default_llm_provider(self):
        cfg = KGConfig()
        assert cfg.llm.provider == "openrouter"
        assert cfg.llm.default_model == "deepseek/deepseek-v4-flash-0731"
    
    def test_default_embedding(self):
        cfg = KGConfig()
        assert cfg.embedding.provider == "local"
        assert cfg.embedding.local_model == "all-MiniLM-L6-v2"
    
    def test_default_neo4j(self):
        cfg = KGConfig()
        assert cfg.neo4j.uri == "bolt://localhost:7687"


class TestEnvOverrides:
    def test_env_override_llm_key(self, monkeypatch):
        monkeypatch.setenv("KG_LLM_API_KEY", "sk-test-123")
        cfg = load_config(auto_create=False)
        assert cfg.llm.api_key == "sk-test-123"
    
    def test_env_override_bool(self, monkeypatch):
        monkeypatch.setenv("KG_VERBOSE", "true")
        cfg = load_config(auto_create=False)
        assert cfg.verbose is True
    
    def test_env_override_int(self, monkeypatch):
        monkeypatch.setenv("KG_EMBEDDING_DIMENSION", "768")
        cfg = load_config(auto_create=False)
        assert cfg.embedding.dimension == 768
    
    def test_env_override_uri(self, monkeypatch):
        monkeypatch.setenv("KG_NEO4J_URI", "bolt://remote:7687")
        cfg = load_config(auto_create=False)
        assert cfg.neo4j.uri == "bolt://remote:7687"


class TestConfigFile:
    def test_write_and_load_default(self, tmp_path):
        config_dir = tmp_path / ".config" / "agent-knowledge-graph"
        config_path = config_dir / "config.yaml"
        _write_default_config(config_path)
        assert config_path.exists()
        
        cfg = load_config(config_path=config_path, auto_create=False)
        assert cfg.llm.provider == "openrouter"
    
    def test_yaml_overrides_default(self, tmp_path):
        config_path = tmp_path / "test-config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = _default_config_dict()
        data["llm"]["provider"] = "openai"
        data["llm"]["api_key"] = "sk-from-file"
        with open(config_path, "w") as f:
            yaml.dump(data, f)
        
        cfg = load_config(config_path=config_path, auto_create=False)
        assert cfg.llm.provider == "openai"
    
    def test_env_overrides_file(self, tmp_path, monkeypatch):
        """Env vars should override file values."""
        config_path = tmp_path / "test-config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = _default_config_dict()
        data["neo4j"]["uri"] = "bolt://file-config:7687"
        with open(config_path, "w") as f:
            yaml.dump(data, f)
        
        monkeypatch.setenv("KG_NEO4J_URI", "bolt://env-override:7687")
        cfg = load_config(config_path=config_path, auto_create=False)
        assert cfg.neo4j.uri == "bolt://env-override:7687"
    
    def test_validation_error_bad_provider(self, tmp_path):
        config_path = tmp_path / "bad-config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = _default_config_dict()
        data["llm"]["provider"] = "nonexistent"
        with open(config_path, "w") as f:
            yaml.dump(data, f)
        
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            load_config(config_path=config_path, auto_create=False)
    
    def test_empty_file_uses_defaults(self, tmp_path):
        config_path = tmp_path / "empty.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump({}, f)
        
        cfg = load_config(config_path=config_path, auto_create=False)
        assert cfg.llm.default_model == "deepseek/deepseek-v4-flash-0731"
```

## Instructions for Cursor

1. Replace `core/config.py` with the full implementation above (models + loader + env overrides)
2. Update `cli/init.py` to call `load_config(auto_create=True)` with Rich console output
3. Update `cli/main.py` to wire init command to `run_init()`
4. Update `core/__init__.py` to export config classes
5. Replace `tests/test_config.py` with the comprehensive test suite
6. Run `uv run pytest tests/test_config.py -v` and ensure all tests pass
7. Run `uv run python -c "from core.config import load_config; cfg = load_config(auto_create=False); print(cfg.llm.provider)"` to verify import
8. Run `uv run kg init --help` to verify CLI integration