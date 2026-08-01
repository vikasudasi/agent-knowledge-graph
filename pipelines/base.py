"""Base classes and contracts for ingestion pipelines."""

from __future__ import annotations


class Pipeline:
    """Abstract pipeline interface for ETL-like processing."""

    name: str = "base"

    def run(self, limit: int | None = None) -> int:
        """Run the pipeline and return processed item count."""
        raise NotImplementedError("Pipeline.run is not yet implemented")
