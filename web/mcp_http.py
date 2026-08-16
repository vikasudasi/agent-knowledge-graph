"""Authenticated HTTP/SSE MCP server scoped per agent API key."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import mcp_types as mcp_types
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser, RequireAuthMiddleware
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.authentication import AuthCredentials, AuthenticationBackend
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from core.config import KGConfig, load_config
from core.embedding import EmbeddingProviderFactory
from core.graph import Neo4jClient
from core.llm import LLMProviderFactory
from core.query import QueryEngine
from web.auth import AGENT_KEY_PREFIX, hash_agent_key
from web.db import Agent, MetadataStore


@dataclass
class McpLifespanState:
    store: MetadataStore
    config: KGConfig


class AgentKeyTokenVerifier:
    """Validate kg_* agent API keys and return MCP AccessToken metadata."""

    def __init__(self, store: MetadataStore) -> None:
        self._store = store

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token.startswith(AGENT_KEY_PREFIX):
            return None
        agent = self._store.get_agent_by_key_hash(hash_agent_key(token))
        if agent is None:
            return None
        return AccessToken(
            token=token,
            client_id=agent.id,
            scopes=["mcp"],
            subject=agent.user_id,
            claims={"agent_id": agent.id, "user_id": agent.user_id},
        )


class AgentKeyAuthBackend(AuthenticationBackend):
    """Accept Authorization Bearer or X-Agent-Key for MCP requests."""

    def __init__(self, verifier: AgentKeyTokenVerifier) -> None:
        self._verifier = verifier

    async def authenticate(self, conn):  # type: ignore[no-untyped-def]
        token = None
        auth_header = conn.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        if token is None:
            token = conn.headers.get("x-agent-key", "").strip()
        if not token or not token.startswith(AGENT_KEY_PREFIX):
            return None
        auth_info = await self._verifier.verify_token(token)
        if auth_info is None:
            return None
        return AuthCredentials(auth_info.scopes), AuthenticatedUser(auth_info)


TOOLS: list[Tool] = [
    Tool(
        name="kg_query",
        description="Ask a natural-language question about the knowledge graph",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Natural language question"},
            },
            "required": ["question"],
        },
    ),
    Tool(
        name="kg_semantic_search",
        description="Search the knowledge graph by semantic meaning",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="kg_traverse",
        description="Traverse relationships from a starting node",
        input_schema={
            "type": "object",
            "properties": {
                "start_id": {"type": "string"},
                "hops": {"type": "integer", "default": 1},
            },
            "required": ["start_id"],
        },
    ),
    Tool(
        name="kg_stats",
        description="Return knowledge graph statistics",
        input_schema={"type": "object", "properties": {}},
    ),
]


def _require_agent(ctx: ServerRequestContext[McpLifespanState, Request]) -> AccessToken:
    request = ctx.request
    if request is None or not hasattr(request, "user") or not isinstance(request.user, AuthenticatedUser):
        raise PermissionError("Missing or invalid agent API key")
    return request.user.access_token


def _get_agent(state: McpLifespanState, agent_id: str) -> Agent:
    agent = state.store.get_agent_by_id(agent_id)
    if agent is None or agent.revoked:
        raise PermissionError("Agent not found or revoked")
    return agent


def _engine_for_agent(state: McpLifespanState, agent: Agent) -> tuple[QueryEngine, list[str]]:
    graph_ids = state.store.resolve_agent_graph_scope(agent)
    if not graph_ids:
        raise PermissionError("Agent has no accessible graphs")
    graph = Neo4jClient(state.config)
    graph.connect()
    embedder = EmbeddingProviderFactory.create(state.config)
    llm = LLMProviderFactory.create(state.config)
    engine = QueryEngine(graph=graph, embedder=embedder, llm=llm, graph_ids=graph_ids)
    return engine, graph_ids


def create_mcp_apps(
    store: MetadataStore,
    config: KGConfig | None = None,
) -> tuple[Starlette, Starlette, Server[McpLifespanState]]:
    """Create streamable HTTP and SSE Starlette apps for authenticated MCP."""
    cfg = config or load_config(auto_create=False)
    verifier = AgentKeyTokenVerifier(store)
    auth_backend = AgentKeyAuthBackend(verifier)
    auth_settings = AuthSettings(
        issuer_url=AnyHttpUrl("http://localhost"),
        resource_server_url=AnyHttpUrl("http://localhost/mcp"),
        required_scopes=["mcp"],
    )

    @asynccontextmanager
    async def lifespan(server: Server[McpLifespanState]) -> Any:
        yield McpLifespanState(store=store, config=cfg)

    async def list_tools(
        ctx: ServerRequestContext[McpLifespanState, Request],
        req: mcp_types.PaginatedRequestParams | None,
    ) -> mcp_types.ListToolsResult:
        _require_agent(ctx)
        return mcp_types.ListToolsResult(tools=TOOLS)

    async def call_tool(
        ctx: ServerRequestContext[McpLifespanState, Request],
        req: mcp_types.CallToolRequestParams,
    ) -> CallToolResult:
        token = _require_agent(ctx)
        agent_id = str(token.claims.get("agent_id", token.client_id) if token.claims else token.client_id)
        state = ctx.lifespan_context
        engine: QueryEngine | None = None
        try:
            agent = _get_agent(state, agent_id)
            engine, graph_ids = _engine_for_agent(state, agent)
            name = req.name
            args = req.arguments or {}
            if name == "kg_query":
                nl_result = engine.nl_query(str(args["question"]))
                data = {
                    "question": args["question"],
                    "cypher": nl_result.cypher or "",
                    "results": nl_result.results or [],
                    "error": nl_result.error or "",
                    "execution_time_ms": nl_result.execution_time_ms or 0,
                    "graph_ids": graph_ids,
                }
            elif name == "kg_semantic_search":
                search_result = engine.semantic(str(args["query"]), top_k=int(args.get("top_k", 5)))
                data = {
                    "query": args["query"],
                    "results": [
                        {
                            "id": node.id,
                            "type": node.type,
                            "label": node.label,
                            "score": search_result.scores[idx] if search_result.scores else None,
                        }
                        for idx, node in enumerate(search_result.nodes)
                    ],
                    "execution_time_ms": search_result.execution_time_ms or 0,
                    "graph_ids": graph_ids,
                }
            elif name == "kg_traverse":
                traverse_result = engine.traverse(str(args["start_id"]), hops=int(args.get("hops", 1)))
                data = {
                    "start_id": args["start_id"],
                    "nodes": [{"id": n.id, "type": n.type, "label": n.label} for n in traverse_result.nodes],
                    "relationships": [
                        {"source": rel.source_id, "target": rel.target_id, "type": rel.type}
                        for rel in traverse_result.relationships
                    ],
                    "execution_time_ms": traverse_result.execution_time_ms or 0,
                    "graph_ids": graph_ids,
                }
            elif name == "kg_stats":
                stats = engine._graph.get_stats(graph_ids=graph_ids)
                data = {
                    "node_count": stats.node_count,
                    "relationship_count": stats.relationship_count,
                    "vector_index_ready": stats.vector_index_ready,
                    "graph_ids": graph_ids,
                    "checkpoints": {
                        cp_name: {
                            "last_processed_id": cp.last_processed_id,
                            "total_processed": cp.total_processed,
                        }
                        for cp_name, cp in stats.last_checkpoints.items()
                    },
                }
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                    is_error=True,
                )
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(data, indent=2))])
        except PermissionError as exc:
            return CallToolResult(content=[TextContent(type="text", text=str(exc))], is_error=True)
        except Exception as exc:
            return CallToolResult(content=[TextContent(type="text", text=f"Error: {exc}")], is_error=True)
        finally:
            if engine is not None:
                engine._graph.close()

    server: Server[McpLifespanState] = Server(
        "agent-knowledge-graph",
        version="0.1.0",
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )

    streamable_app = server.streamable_http_app(
        streamable_http_path="",
        token_verifier=verifier,
        auth=auth_settings,
        stateless_http=True,
        host="0.0.0.0",
    )
    streamable_app.user_middleware = [
        Middleware(AuthenticationMiddleware, backend=auth_backend),
        Middleware(AuthContextMiddleware),
    ]

    sse_transport = SseServerTransport("/messages/", security_settings=TransportSecuritySettings())

    async def handle_sse(scope: Scope, receive: Receive, send: Send) -> None:
        async with sse_transport.connect_sse(scope, receive, send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    async def sse_endpoint(request: Request) -> Response:
        await handle_sse(request.scope, request.receive, request._send)  # noqa: SLF001
        return Response()

    sse_middleware = [
        Middleware(AuthenticationMiddleware, backend=auth_backend),
        Middleware(AuthContextMiddleware),
    ]
    sse_routes: list[Route | Mount] = [
        Route(
            "/sse",
            endpoint=RequireAuthMiddleware(sse_endpoint, ["mcp"], None),
            methods=["GET"],
        ),
        Mount(
            "/messages/",
            app=RequireAuthMiddleware(sse_transport.handle_post_message, ["mcp"], None),
        ),
    ]
    sse_app = Starlette(routes=sse_routes, middleware=sse_middleware)

    return streamable_app, sse_app, server


__all__ = ["AgentKeyTokenVerifier", "create_mcp_apps"]
