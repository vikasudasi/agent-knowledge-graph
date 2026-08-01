# agent-knowledge-graph — Pipeline Framework (Task 6)

## What to Build

The pluggable pipeline framework that defines the standard contract for all data ingestion. Each pipeline runs a four-phase lifecycle: extract → resolve → embed → write. Checkpointing ensures idempotent incremental runs.

## Files to Create

- `pipelines/base.py` — full replacement with `KnowledgePipeline`, `PipelineContext`, `PipelineRegistry`
- `tests/test_pipelines.py` — test suite

## Implementation

### pipelines/base.py

```python
"""Pluggable pipeline framework — the standard contract for all data ingestion."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Generator, Generic, TypeVar

from core.config import KGConfig
from core.embedding import EmbeddingProvider, EmbeddingProviderFactory
from core.graph import Neo4jClient
from core.llm import LLMClient, LLMProviderFactory
from core.models import PipelineCheckpoint, Resource, Relationship

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Source record type


@dataclass
class PipelineContext:
    """Context passed to every pipeline phase."""
    config: KGConfig
    llm: LLMClient
    embedder: EmbeddingProvider
    graph: Neo4jClient
    dry_run: bool = False
    full_rebuild: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    pipeline_name: str
    records_processed: int = 0
    resources_created: int = 0
    relationships_created: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    checkpoint: PipelineCheckpoint | None = None


ProgressCallback = Callable[[int, int, str], None]


class KnowledgePipeline(ABC, Generic[T]):
    """Abstract base for all knowledge graph ingestion pipelines.

    Lifecycle:
        1. extract()   — yield raw records from source
        2. resolve()   — deduplicate/enrich records against existing graph state
        3. embed()     — generate vector embeddings for resolved records
        4. write()     — upsert resources + relationships to Neo4j
    """

    def __init__(self, name: str, description: str = "", version: str = "1.0") -> None:
        self._name = name
        self._description = description
        self._version = version
        self._context: PipelineContext | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def version(self) -> str:
        return self._version

    # ── Public API ──────────────────────────────────────────────────

    def run(
        self,
        context: PipelineContext,
        full_rebuild: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Run the pipeline: extract → resolve → embed → write.

        Args:
            context: Pipeline context with config, clients, etc.
            full_rebuild: If True, ignore checkpoint and process all records.
            progress_callback: Optional callback (current, total, phase).
        """
        self._context = context
        t0 = time.monotonic()
        result = PipelineResult(pipeline_name=self._name)

        # Load checkpoint
        checkpoint: PipelineCheckpoint | None = None
        if not full_rebuild:
            checkpoint = context.graph.get_checkpoint(self._name)
            if checkpoint and checkpoint.last_processed_id:
                logger.info(f"Resuming from checkpoint: {checkpoint.last_processed_id} "
                            f"({checkpoint.total_processed} processed)")

        # Phase 1: Extract
        if progress_callback:
            progress_callback(0, 0, "extracting")
        logger.info(f"Pipeline '{self._name}': starting extract phase")
        records = list(self.extract(context, checkpoint))
        total = len(records)
        if progress_callback:
            progress_callback(0, total, "resolving")
        logger.info(f"Extracted {total} records")

        if total == 0:
            elapsed = time.monotonic() - t0
            result.duration_seconds = elapsed
            result.checkpoint = checkpoint
            logger.info(f"Pipeline '{self._name}': no new records, done in {elapsed:.2f}s")
            return result

        # Phase 2: Resolve
        resolved = []
        for i, record in enumerate(records):
            try:
                resolved_resources = self.resolve(context, record)
                resolved.extend(resolved_resources)
            except Exception as e:
                logger.error(f"Error resolving record {i}: {e}")
                result.errors += 1
            if progress_callback:
                progress_callback(i + 1, total, "resolving")

        if not resolved:
            elapsed = time.monotonic() - t0
            result.duration_seconds = elapsed
            logger.info(f"Pipeline '{self._name}': no resources resolved, done in {elapsed:.2f}s")
            return result

        # Phase 3: Embed
        if progress_callback:
            progress_callback(0, len(resolved), "embedding")
        embedded = []
        for i, resource in enumerate(resolved):
            try:
                if resource.embedding is None:
                    resource.embedding = context.embedder.embed(resource.label + " " + str(resource.properties))
                embedded.append(resource)
            except Exception as e:
                logger.error(f"Error embedding resource {resource.id}: {e}")
                result.errors += 1
            if progress_callback:
                progress_callback(i + 1, len(resolved), "embedding")

        # Phase 4: Write
        if progress_callback:
            progress_callback(0, len(embedded), "writing")
        if not context.dry_run:
            for i, resource in enumerate(embedded):
                try:
                    context.graph.upsert_resource(resource)
                    result.resources_created += 1
                except Exception as e:
                    logger.error(f"Error writing resource {resource.id}: {e}")
                    result.errors += 1
                if progress_callback:
                    progress_callback(i + 1, len(embedded), "writing")

            # Also upsert relationships if the pipeline generates them
            relationships = self.get_relationships(context, records, embedded)
            for rel in relationships:
                try:
                    context.graph.upsert_relationship(rel)
                    result.relationships_created += 1
                except Exception as e:
                    logger.error(f"Error writing relationship {rel.source_id}->{rel.target_id}: {e}")
                    result.errors += 1
        else:
            logger.info(f"Dry-run: would upsert {len(embedded)} resources")
            result.resources_created = len(embedded)

        # Save checkpoint
        result.records_processed = total
        if records and not context.dry_run and not full_rebuild:
            last_id = getattr(records[-1], "id", None) or getattr(records[-1], "session_id", str(total))
            new_checkpoint = PipelineCheckpoint(
                pipeline_name=self._name,
                last_processed_id=str(last_id),
                total_processed=(checkpoint.total_processed if checkpoint else 0) + total,
                updated_at=datetime.now(timezone.utc),
            )
            context.graph.save_checkpoint(new_checkpoint)
            result.checkpoint = new_checkpoint

        elapsed = time.monotonic() - t0
        result.duration_seconds = elapsed
        logger.info(f"Pipeline '{self._name}': completed in {elapsed:.2f}s — "
                    f"{result.resources_created} resources, {result.relationships_created} rels, "
                    f"{result.errors} errors")
        return result

    # ── Pipeline lifecycle (override in subclasses) ────────────────

    @abstractmethod
    def extract(self, context: PipelineContext, checkpoint: PipelineCheckpoint | None = None) -> Generator[T, None, None]:
        """Yield raw records from the source.

        Each record is a domain-specific object (dict, dataclass, etc.).
        The checkpoint tells you where to resume from.
        """
        yield from []  # pragma: no cover

    @abstractmethod
    def resolve(self, context: PipelineContext, record: T) -> list[Resource]:
        """Convert a raw record into Resource nodes.

        Called once per record. Can return multiple resources (e.g. one per entity).
        Should check existing graph state for deduplication.
        """
        return []  # pragma: no cover

    def get_relationships(
        self,
        context: PipelineContext,
        records: list[T],
        resources: list[Resource],
    ) -> list[Relationship]:
        """Generate relationships between resources. Override to create edges."""
        return []


class PipelineRegistry:
    """Auto-discovers and manages registered pipelines."""

    _pipelines: dict[str, KnowledgePipeline] = {}

    @classmethod
    def register(cls, pipeline: KnowledgePipeline) -> None:
        """Register a pipeline by name."""
        if pipeline.name in cls._pipelines:
            logger.warning(f"Overwriting existing pipeline: {pipeline.name}")
        cls._pipelines[pipeline.name] = pipeline
        logger.debug(f"Registered pipeline: {pipeline.name}")

    @classmethod
    def get(cls, name: str) -> KnowledgePipeline | None:
        return cls._pipelines.get(name)

    @classmethod
    def list_pipelines(cls) -> list[dict[str, str]]:
        return [
            {"name": p.name, "description": p.description, "version": p.version}
            for p in cls._pipelines.values()
        ]

    @classmethod
    def create_context(cls, config: KGConfig, dry_run: bool = False, full_rebuild: bool = False) -> PipelineContext:
        """Create a PipelineContext from config with all providers wired up."""
        llm = LLMProviderFactory.create(config)
        embedder = EmbeddingProviderFactory.create(config)
        graph = Neo4jClient(config)
        graph.connect()
        return PipelineContext(
            config=config,
            llm=llm,
            embedder=embedder,
            graph=graph,
            dry_run=dry_run,
            full_rebuild=full_rebuild,
        )
```

### tests/test_pipelines.py

```python
"""Tests for the pipeline framework."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import PipelineCheckpoint, Resource, Relationship
from pipelines.base import (
    KnowledgePipeline,
    PipelineContext,
    PipelineRegistry,
    PipelineResult,
)


class SimplePipeline(KnowledgePipeline):
    """Test pipeline that yields static records."""
    def extract(self, context, checkpoint=None):
        for i in range(3):
            yield {"id": f"rec-{i}", "content": f"Record {i}"}

    def resolve(self, context, record):
        return [
            Resource(
                id=record["id"],
                type="test",
                label=record["content"],
                properties={"index": int(record["id"].split("-")[1])},
            )
        ]

    def get_relationships(self, context, records, resources):
        if len(resources) >= 2:
            return [Relationship(
                source_id=resources[0].id,
                target_id=resources[1].id,
                type="relates_to",
                properties={"order": 1},
            )]
        return []


class EmptyPipeline(KnowledgePipeline):
    """Pipeline that yields nothing."""
    def extract(self, context, checkpoint=None):
        return iter([])
        yield  # make it a generator

    def resolve(self, context, record):
        return []


class FailingPipeline(KnowledgePipeline):
    """Pipeline that fails during resolve."""
    def extract(self, context, checkpoint=None):
        yield {"id": "fail-1"}

    def resolve(self, context, record):
        raise ValueError("Intentional failure")


class TestPipelineContext:
    def test_create_context_wires_providers(self):
        from core.config import KGConfig
        cfg = KGConfig()
        cfg.llm.api_key = "test-key"
        cfg.embedding.provider = "local"

        with patch("core.embedding.LocalEmbeddingProvider"):
            with patch.object(cfg.llm, "api_key", "test"):
                with patch("pipelines.base.LLMProviderFactory.create") as mock_llm:
                    with patch("pipelines.base.EmbeddingProviderFactory.create") as mock_emb:
                        with patch("pipelines.base.Neo4jClient") as mock_graph:
                            mock_graph.return_value = MagicMock()
                            ctx = PipelineRegistry.create_context(cfg)
                            assert isinstance(ctx, PipelineContext)
                            assert ctx.config == cfg


class TestKnowledgePipeline:
    def test_run_extract_resolve_embed_write(self):
        pipeline = SimplePipeline(name="test-simple", description="Simple test pipeline")
        mock_graph = MagicMock()
        mock_graph.get_checkpoint.return_value = None
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
        )

        result = pipeline.run(ctx)
        assert isinstance(result, PipelineResult)
        assert result.pipeline_name == "test-simple"
        assert result.records_processed == 3
        assert result.resources_created == 3
        assert result.relationships_created == 1

    def test_run_empty_pipeline(self):
        pipeline = EmptyPipeline(name="empty")
        mock_graph = MagicMock()
        mock_graph.get_checkpoint.return_value = None

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=MagicMock(),
            graph=mock_graph,
        )

        result = pipeline.run(ctx)
        assert result.records_processed == 0
        assert result.resources_created == 0

    def test_run_with_checkpoint_resume(self):
        pipeline = SimplePipeline(name="test-checkpoint")
        mock_graph = MagicMock()
        mock_graph.get_checkpoint.return_value = PipelineCheckpoint(
            pipeline_name="test-checkpoint",
            last_processed_id="rec-1",
            total_processed=2,
        )
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
        )

        result = pipeline.run(ctx)
        assert result.records_processed == 3

    def test_run_full_rebuild_ignores_checkpoint(self):
        pipeline = SimplePipeline(name="test-rebuild")
        mock_graph = MagicMock()
        mock_graph.get_checkpoint.return_value = PipelineCheckpoint(
            pipeline_name="test-rebuild", last_processed_id="rec-5", total_processed=10
        )
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
            full_rebuild=True,
        )

        result = pipeline.run(ctx)
        assert result.records_processed == 3  # Not 10 + 3, full rebuild

    def test_run_dry_mode(self):
        pipeline = SimplePipeline(name="test-dry")
        mock_graph = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
            dry_run=True,
        )

        result = pipeline.run(ctx)
        assert result.resources_created == 3
        mock_graph.upsert_resource.assert_not_called()

    def test_error_isolation(self):
        pipeline = FailingPipeline(name="test-fail")
        mock_graph = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
        )

        result = pipeline.run(ctx)
        assert result.errors == 1
        assert result.records_processed == 1
        assert result.resources_created == 0

    def test_progress_callback(self):
        pipeline = SimplePipeline(name="test-progress")
        mock_graph = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        calls = []
        def progress(current, total, phase):
            calls.append((current, total, phase))

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
        )

        pipeline.run(ctx, progress_callback=progress)
        assert len(calls) > 0


class TestPipelineRegistry:
    def test_register_and_get(self):
        pipeline = SimplePipeline(name="reg-test")
        PipelineRegistry.register(pipeline)
        assert PipelineRegistry.get("reg-test") is pipeline

    def test_list_pipelines(self):
        PipelineRegistry._pipelines.clear()
        PipelineRegistry.register(SimplePipeline(name="p1"))
        PipelineRegistry.register(SimplePipeline(name="p2"))
        listed = PipelineRegistry.list_pipelines()
        assert len(listed) == 2
        names = [p["name"] for p in listed]
        assert "p1" in names
        assert "p2" in names

    def test_get_nonexistent(self):
        assert PipelineRegistry.get("nope") is None
```

## Instructions for Cursor CLI

1. Replace `pipelines/base.py` with the full implementation above
2. Create `tests/test_pipelines.py` with the test suite
3. Run `uv run python -m pytest tests/test_pipelines.py -v` and report results
4. Run `uv run python -c "from pipelines.base import KnowledgePipeline, PipelineContext, PipelineRegistry; print('Imports OK')"`