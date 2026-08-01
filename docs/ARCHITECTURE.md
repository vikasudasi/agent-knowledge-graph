# Architecture

This document describes the high-level architecture for `agent-knowledge-graph`.

## Layers

1. **CLI layer (`cli/`)**  
   Handles user interaction and command orchestration.
2. **Pipeline layer (`pipelines/`)**  
   Defines ingest pipelines that convert raw artifacts into normalized memory records.
3. **Core services (`core/`)**  
   Owns configuration, embeddings, graph persistence, and LLM interfaces.
4. **Adapters (`adapters/`)**  
   Connects external agent ecosystems (Hermes, MCP, LangChain).
5. **Storage backend (Neo4j + vector indexes)**  
   Persists structured memory entities and semantic embeddings.

## Current Status

Repository scaffolding is complete; functional implementations are intentionally deferred to future milestones.
