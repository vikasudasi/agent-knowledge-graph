"""Configuration management for agent-knowledge-graph."""

from __future__ import annotations

from pathlib import Path


class ConfigManager:
    """Load and manage application configuration from disk."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path

    def load(self) -> dict[str, str]:
        """Load configuration values from configured paths."""
        raise NotImplementedError("ConfigManager.load is not yet implemented")
