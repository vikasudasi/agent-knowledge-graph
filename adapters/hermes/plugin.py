"""Hermes plugin adapter for knowledge graph memory hooks."""

from __future__ import annotations


class HermesMemoryPlugin:
    """Bridge Hermes lifecycle events into ingest pipelines."""

    def install(self) -> None:
        """Install plugin callbacks into the Hermes runtime."""
        raise NotImplementedError("HermesMemoryPlugin.install is not yet implemented")
