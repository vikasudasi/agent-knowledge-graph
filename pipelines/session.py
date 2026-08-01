"""Session ingestion pipeline for agent event streams."""

from __future__ import annotations

from pipelines.base import Pipeline


class SessionPipeline(Pipeline):
    """Ingest assistant and user sessions into canonical graph records."""

    name = "sessions"

    def run(self, limit: int | None = None) -> int:
        """Run the session ingest pipeline."""
        raise NotImplementedError("SessionPipeline.run is not yet implemented")
