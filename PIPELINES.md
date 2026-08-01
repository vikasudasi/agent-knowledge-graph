# Custom Pipeline Guide

## 1) Pipeline Contract

Custom ingestion pipelines subclass `KnowledgePipeline[T]` and participate in a four-phase contract:

1. **Extract**: yield raw source records (`extract`).
2. **Resolve**: map each record to one or more `Resource` nodes (`resolve`).
3. **Embed**: framework generates embeddings when missing.
4. **Write**: framework upserts resources and optional relationships.

Required methods:

- `extract(context, checkpoint)`
- `resolve(context, record)`

Optional method:

- `get_relationships(context, records, resources)`

## 2) Step-by-step: Custom Pipeline

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from core.models import PipelineCheckpoint, Relationship, Resource
from pipelines.base import KnowledgePipeline, PipelineContext, PipelineRegistry


@dataclass
class TicketRecord:
    id: str
    title: str
    owner: str


class TicketPipeline(KnowledgePipeline[TicketRecord]):
    def __init__(self) -> None:
        super().__init__(name="ticket-ingest", description="Ingest support tickets", version="1.0")

    def extract(
        self,
        context: PipelineContext,
        checkpoint: PipelineCheckpoint | None = None,
    ) -> Generator[TicketRecord, None, None]:
        _ = context
        last = checkpoint.last_processed_id if checkpoint else ""
        records = [
            TicketRecord(id="t-1", title="API timeout", owner="alice"),
            TicketRecord(id="t-2", title="Deploy rollback", owner="bob"),
        ]
        for record in records:
            if not last or record.id > last:
                yield record

    def resolve(self, context: PipelineContext, record: TicketRecord) -> list[Resource]:
        _ = context
        return [
            Resource(
                id=f"ticket:{record.id}",
                type="task",
                label=record.title,
                properties={"owner": record.owner, "source": "ticket-system"},
            )
        ]

    def get_relationships(
        self,
        context: PipelineContext,
        records: list[TicketRecord],
        resources: list[Resource],
    ) -> list[Relationship]:
        _ = (context, records)
        rels: list[Relationship] = []
        for resource in resources:
            owner = resource.properties.get("owner", "unknown")
            rels.append(
                Relationship(
                    source_id=resource.id,
                    target_id=f"entity:{owner}",
                    type="assigns",
                    properties={"weight": 1.0},
                )
            )
        return rels


PipelineRegistry.register(TicketPipeline())
```

## 3) Checkpointing

Best practices:

- Use stable, monotonic source IDs when possible.
- Respect `checkpoint.last_processed_id` in `extract`.
- Keep extraction deterministic so replayed records are predictable.
- Let framework persist checkpoints by returning records with stable IDs.

## 4) Error Handling

The framework isolates failures by phase:

- Resolve and embed errors increment pipeline error counts.
- Failing records do not halt the full run by default.
- Graph write errors are captured per resource and relationship.

Recommended strategies:

- Include source record IDs in exception logs.
- Prefer fallback resource creation over hard aborts.
- Use `dry_run` for safe verification before production runs.

## 5) Testing Pipelines

For stable tests, mock all external boundaries:

- **LLM**: return fixed extraction payloads.
- **Embedding**: return deterministic vectors.
- **Graph**: assert on `upsert_resource`, `upsert_relationship`, and checkpoint saves.

Coverage checklist:

- normal pipeline run
- empty extract result
- checkpoint resume behavior
- `dry_run=True` behavior
- resolve, embed, and write error paths
