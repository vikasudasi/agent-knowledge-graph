#!/usr/bin/env python3
"""MCP server entrypoint for agent-knowledge-graph — Hermes integration.

Register in ~/.hermes/config.yaml:
    mcp_servers:
      knowledge-graph:
        command: "/root/agent-knowledge-graph/.venv/bin/python"
        args: ["-m", "adapters.mcp_server_entry"]

Then restart Hermes. The tools kg_query, kg_semantic_search, kg_traverse, kg_stats
become available as mcp_knowledge_graph_kg_query etc.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path for local development
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import anyio
from mcp import types as mcp_types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from core.config import load_config
from core.embedding import EmbeddingProviderFactory
from core.graph import Neo4jClient
from core.llm import LLMProviderFactory
from core.query import QueryEngine

# ── Lazy engine singleton ────────────────────────────────────────────────────

_engine: QueryEngine | None = None


def get_engine() -> QueryEngine:
    global _engine
    if _engine is None:
        cfg = load_config(auto_create=False)
        graph = Neo4jClient(cfg)
        graph.connect()
        embedder = EmbeddingProviderFactory.create(cfg)
        llm = LLMProviderFactory.create(cfg)
        _engine = QueryEngine(graph=graph, embedder=embedder, llm=llm)
    return _engine


# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool(
        name="kg_query",
        description="Ask a natural-language question about the knowledge graph"
        " — translates to Cypher and returns results",
        inputSchema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question about the knowledge graph",
                }
            },
            "required": ["question"],
        },
    ),
    Tool(
        name="kg_semantic_search",
        description="Search the knowledge graph by semantic meaning — returns the most similar resources",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to find semantically similar resources",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="kg_traverse",
        description="Traverse relationships from a starting node in the knowledge graph",
        inputSchema={
            "type": "object",
            "properties": {
                "start_id": {
                    "type": "string",
                    "description": "ID of the starting node to traverse from",
                },
                "hops": {
                    "type": "integer",
                    "description": "Number of relationship hops (default: 1)",
                    "default": 1,
                },
            },
            "required": ["start_id"],
        },
    ),
    Tool(
        name="kg_stats",
        description="Return statistics about the knowledge graph —"
        " node/relationship counts, vector index status, checkpoints",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# ── Handler ──────────────────────────────────────────────────────────────────


async def handle_call_tool(ctx, req: mcp_types.CallToolRequestParams) -> CallToolResult:
    name = req.name
    args = req.arguments or {}

    try:
        if name == "kg_query":
            result = get_engine().nl_query(args["question"])
            data = {
                "question": args["question"],
                "cypher": result.cypher or "",
                "results": result.results or [],
                "error": result.error or "",
                "execution_time_ms": result.execution_time_ms or 0,
            }
        elif name == "kg_semantic_search":
            result = get_engine().semantic(args["query"], top_k=int(args.get("top_k", 5)))
            data = {
                "query": args["query"],
                "results": [
                    {
                        "id": node.id,
                        "type": node.type,
                        "label": node.label,
                        "score": result.scores[idx] if result.scores else None,
                    }
                    for idx, node in enumerate(result.nodes)
                ],
                "execution_time_ms": result.execution_time_ms or 0,
            }
        elif name == "kg_traverse":
            result = get_engine().traverse(args["start_id"], hops=int(args.get("hops", 1)))
            data = {
                "start_id": args["start_id"],
                "nodes": [{"id": node.id, "type": node.type, "label": node.label} for node in result.nodes],
                "relationships": [
                    {"source": rel.source_id, "target": rel.target_id, "type": rel.type} for rel in result.relationships
                ],
                "execution_time_ms": result.execution_time_ms or 0,
            }
        elif name == "kg_stats":
            cfg = load_config(auto_create=False)
            graph = Neo4jClient(cfg)
            graph.connect()
            try:
                stats = graph.get_stats()
            finally:
                graph.close()
            data = {
                "node_count": stats.node_count,
                "relationship_count": stats.relationship_count,
                "vector_index_ready": stats.vector_index_ready,
                "checkpoints": {
                    name: {
                        "last_processed_id": cp.last_processed_id,
                        "total_processed": cp.total_processed,
                    }
                    for name, cp in stats.last_checkpoints.items()
                },
            }
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )

        return CallToolResult(content=[TextContent(type="text", text=json.dumps(data, indent=2))])
    except Exception as exc:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {exc}")],
            isError=True,
        )


# ── Server lifecycle ─────────────────────────────────────────────────────────


async def main() -> None:
    from mcp import stdio_server

    # Register request handlers via constructor (SDK default routing)
    async def list_tools(ctx, req: mcp_types.PaginatedRequestParams | None) -> mcp_types.ListToolsResult:
        return mcp_types.ListToolsResult(tools=TOOLS)

    server = Server(
        "agent-knowledge-graph",
        version="0.1.0",
        on_list_tools=list_tools,
        on_call_tool=handle_call_tool,
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="agent-knowledge-graph",
                server_version="0.1.0",
                capabilities=server.get_capabilities(),
            ),
        )


if __name__ == "__main__":
    anyio.run(main)
