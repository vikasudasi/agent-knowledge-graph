# Agent Integrations

This document tracks how different agent runtimes integrate with `agent-knowledge-graph`.

## Hermes

The Hermes adapter will expose plugin hooks for ingesting events and querying memory during agent execution.

## MCP

The MCP adapter will expose memory operations as MCP tools so any MCP-capable client can use graph memory.

## LangChain

The LangChain adapter will provide a tool wrapper that routes natural-language questions through graph retrieval.

## Integration Principles

- Keep adapter contracts narrow and explicit.
- Preserve source attribution and timestamps.
- Avoid adapter-specific logic in core domain services.
