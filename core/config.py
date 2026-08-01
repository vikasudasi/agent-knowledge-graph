"""Configuration management for agent-knowledge-graph."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator


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
    _config_path: Path | None = PrivateAttr(default=None)

    @field_validator("llm")
    @classmethod
    def validate_llm(cls, value: LLMConfig) -> LLMConfig:
        if value.provider not in ("openrouter", "openai", "anthropic", "custom"):
            raise ValueError(f"Unknown LLM provider: {value.provider}")
        return value

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: EmbeddingConfig) -> EmbeddingConfig:
        if value.provider not in ("local", "openrouter", "openai"):
            raise ValueError(f"Unknown embedding provider: {value.provider}")
        if value.dimension not in (384, 768, 1024, 1536, 3072):
            raise ValueError(f"Unsupported embedding dimension: {value.dimension}")
        return value


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
            # Navigate nested dict by dotted path.
            parts = key_path.split(".")
            target = config
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]

            # Coerce types: bool, int.
            leaf = parts[-1]
            if value.lower() in ("true", "1", "yes"):
                target[leaf] = True
            elif value.lower() in ("false", "0", "no"):
                target[leaf] = False
            elif value.isdigit():
                target[leaf] = int(value)
            else:
                target[leaf] = value
    return config


def _default_config_dict() -> dict[str, Any]:
    """Return the default config as a dict."""

    return KGConfig().model_dump()


def _write_default_config(path: Path) -> Path:
    """Write default config to the given path. Creates parent dirs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.dump(_default_config_dict(), file, default_flow_style=False, sort_keys=False)
    return path


def load_config(
    config_path: Path | None = None,
    auto_create: bool = True,
) -> KGConfig:
    """Load config from file + env overrides."""

    # Determine which config file to load.
    config_file: Path | None = None

    if config_path is not None:
        if config_path.exists():
            config_file = config_path
    else:
        for path in _default_config_paths():
            if path.exists():
                config_file = path
                break

    # Load from file or start with defaults.
    if config_file:
        with config_file.open(encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    else:
        raw = {}
        if auto_create:
            config_file = _default_config_paths()[0]
            _write_default_config(config_file)

    # Apply env overrides.
    raw = _apply_env_overrides(raw)

    # Validate and return typed config.
    cfg = KGConfig(**raw)
    cfg._config_path = config_file
    return cfg


__all__ = [
    "KGConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "Neo4jConfig",
    "PipelineConfig",
    "StorageConfig",
    "load_config",
]
