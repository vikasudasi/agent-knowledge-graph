# Pipelines

`agent-knowledge-graph` uses a pipeline abstraction to ingest agent artifacts into graph-native memory.

## Goals

- Normalize heterogeneous inputs (sessions, files, tool output).
- Enrich records with tags, embeddings, and provenance metadata.
- Upsert graph nodes and relationships predictably.
- Support incremental and full rebuild modes.

## Pipeline Contract

- Implement `Pipeline.run(limit: int | None) -> int`.
- Return the number of processed records.
- Keep ingest operations idempotent where possible.

## Initial Pipeline

`SessionPipeline` is the first pipeline scaffold and targets assistant/user session transcripts.

## Future Pipelines

- Repository file ingest.
- Issue and PR event ingest.
- Tool execution traces.
