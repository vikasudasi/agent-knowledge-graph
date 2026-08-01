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
from core.models import PipelineCheckpoint, Relationship, Resource

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
                logger.info(
                    f"Resuming from checkpoint: {checkpoint.last_processed_id} "
                    f"({checkpoint.total_processed} processed)"
                )

        # Phase 1: Extract
        if progress_callback:
            progress_callback(0, 0, "extracting")
        logger.info(f"Pipeline '{self._name}': starting extract phase")
        records = list(self.extract(context, checkpoint))
        total = len(records)
        if progress_callback:
            progress_callback(0, total, "resolving")
        logger.info(f"Extracted {total} records")

        result.records_processed = total
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

        result.records_processed = total
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
        logger.info(
            f"Pipeline '{self._name}': completed in {elapsed:.2f}s — "
            f"{result.resources_created} resources, {result.relationships_created} rels, "
            f"{result.errors} errors"
        )
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
