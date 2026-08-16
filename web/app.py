"""FastAPI web dashboard for multi-user knowledge graph management."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import FastAPI, Form, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from core.config import KGConfig, load_config
from core.graph import Neo4jClient
from web.auth import (
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    LoginForm,
    SessionManager,
    SignupForm,
    generate_agent_key,
    get_or_create_session_secret,
    hash_agent_key,
    hash_password,
    key_prefix,
    verify_password,
)
from web.db import MetadataStore, metadata_db_path
from web.mcp_http import create_mcp_apps

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
GRAPH_VIZ_NODE_LIMIT = 200
NODES_PAGE_SIZE = 50

TYPE_COLORS: dict[str, str] = {
    "session": "#16a34a",
    "person": "#ec4899",
    "project": "#6366f1",
    "tool": "#2563eb",
    "concept": "#f59e0b",
    "file": "#0d9488",
    "artifact": "#9333ea",
    "task": "#0891b2",
    "skill": "#f43f5e",
    "function": "#84cc16",
}
DEFAULT_TYPE_COLOR = "#94a3b8"


def _redirect_with_flash(path: str, flash: str, message: str) -> RedirectResponse:
    separator = "&" if "?" in path else "?"
    url = f"{path}{separator}flash={quote(flash)}&msg={quote(message)}"
    return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)


def _set_session_cookie(response: Response, request: Request, token: str) -> None:
    secure = request.url.scheme == "https"
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_MAX_AGE,
        secure=secure,
        path="/",
    )


def _clear_session_cookie(response: Response, request: Request) -> None:
    secure = request.url.scheme == "https"
    response.delete_cookie(SESSION_COOKIE, path="/", secure=secure)


def _serialize_node_row(row: dict[str, Any]) -> dict[str, Any]:
    properties = row.get("properties") or {}
    if not isinstance(properties, dict):
        properties = dict(properties)
    for key in ("id", "type", "label", "graph_id", "ingested_at", "embedding"):
        properties.pop(key, None)
    return {
        "id": row.get("id", ""),
        "type": row.get("type", ""),
        "label": row.get("label", ""),
        "properties": properties,
        "created_at": row.get("ingested_at") or "",
    }


def create_app(config: KGConfig | None = None) -> FastAPI:
    """Build the FastAPI application."""
    cfg = config or load_config(auto_create=False)
    data_dir = Path(cfg.storage.data_dir).expanduser()
    store = MetadataStore(metadata_db_path(str(data_dir)))
    session_secret = get_or_create_session_secret(store)
    sessions = SessionManager(session_secret)
    streamable_mcp, sse_mcp, mcp_server = create_mcp_apps(store, cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        async with mcp_server.session_manager.run():
            yield

    app = FastAPI(title="agent-knowledge-graph", lifespan=lifespan)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/mcp/sse", sse_mcp)
    app.mount("/mcp", streamable_mcp)

    @app.middleware("http")
    async def attach_user_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.user = None
        session = request.cookies.get(SESSION_COOKIE)
        if session:
            user_id = sessions.read_session_token(session)
            if user_id is not None:
                user = store.get_user_by_id(user_id)
                if user is not None:
                    request.state.user = user
        return await call_next(request)

    def require_user(request: Request):  # type: ignore[no-untyped-def]
        if request.state.user is not None:
            return request.state.user
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    def _graph_client() -> Neo4jClient:
        client = Neo4jClient(cfg)
        client.connect()
        return client

    def _fetch_graph_stats(graphs: list[Any]) -> list[dict[str, Any]]:
        graph_stats: list[dict[str, Any]] = []
        graph_client = _graph_client()
        try:
            for graph in graphs:
                stats = graph_client.get_stats(graph_id=graph.id)
                graph_stats.append(
                    {
                        "id": graph.id,
                        "name": graph.name,
                        "description": graph.description,
                        "created_at": graph.created_at,
                        "node_count": stats.node_count,
                        "relationship_count": stats.relationship_count,
                    }
                )
        except Exception:
            for graph in graphs:
                graph_stats.append(
                    {
                        "id": graph.id,
                        "name": graph.name,
                        "description": graph.description,
                        "created_at": graph.created_at,
                        "node_count": 0,
                        "relationship_count": 0,
                    }
                )
        finally:
            graph_client.close()
        return graph_stats

    def _delete_graph_data(graph_id: str) -> None:
        graph_client = _graph_client()
        try:
            graph_client.run_cypher(
                "MATCH (n:Resource) WHERE n.graph_id = $graph_id DETACH DELETE n",
                graph_id=graph_id,
            )
            graph_client.run_cypher(
                "MATCH (c:PipelineCheckpoint) WHERE c.graph_id = $graph_id DELETE c",
                graph_id=graph_id,
            )
        except Exception:
            pass
        finally:
            graph_client.close()

    @app.exception_handler(404)
    async def not_found_handler(request: Request, _exc: Exception) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "404.html",
            {"title": "Page not found"},
            status_code=status.HTTP_404_NOT_FOUND,
        )

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> Response:
        user = request.state.user
        if user is not None:
            return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        return templates.TemplateResponse(request, "home.html", {"title": "agent-knowledge-graph", "user": None})

    @app.get("/signup", response_class=HTMLResponse)
    async def signup_form(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "signup.html", {"error": None})

    @app.post("/signup")
    async def signup_submit(
        request: Request,
        email: Annotated[str, Form()],
        password: Annotated[str, Form()],
        confirm_password: Annotated[str, Form()],
    ) -> Response:
        try:
            form = SignupForm(email=email, password=password, confirm_password=confirm_password)
        except ValidationError as exc:
            return templates.TemplateResponse(
                request,
                "signup.html",
                {"error": exc.errors()[0]["msg"]},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        if form.password != form.confirm_password:
            return templates.TemplateResponse(
                request,
                "signup.html",
                {"error": "Passwords do not match"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        if store.get_user_by_email(form.email):
            return templates.TemplateResponse(
                request,
                "signup.html",
                {"error": "An account with this email already exists"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        user = store.create_user(uuid.uuid4().hex, form.email, hash_password(form.password))
        token = sessions.create_session_token(user.id)
        response = RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        _set_session_cookie(response, request, token)
        return response

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login")
    async def login_submit(
        request: Request,
        email: Annotated[str, Form()],
        password: Annotated[str, Form()],
    ) -> Response:
        try:
            form = LoginForm(email=email, password=password)
        except ValidationError:
            return templates.TemplateResponse(
                request,
                "login.html",
                {"error": "Invalid email or password"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        user = store.get_user_by_email(form.email)
        if user is None or not verify_password(form.password, user.password_hash):
            return templates.TemplateResponse(
                request,
                "login.html",
                {"error": "Invalid email or password"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        token = sessions.create_session_token(user.id)
        response = RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        _set_session_cookie(response, request, token)
        return response

    @app.post("/logout")
    async def logout(request: Request) -> Response:
        response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
        _clear_session_cookie(response, request)
        return response

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        graphs = store.list_graphs(user.id)
        graph_stats = _fetch_graph_stats(graphs)
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {"user": user, "graphs": graph_stats},
        )

    @app.get("/graphs", response_class=HTMLResponse)
    async def graphs_list(request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        graphs = store.list_graphs(user.id)
        graph_stats = _fetch_graph_stats(graphs)
        return templates.TemplateResponse(
            request,
            "graphs.html",
            {"user": user, "graphs": graph_stats},
        )

    @app.get("/graphs/new", response_class=HTMLResponse)
    async def graph_new_form(request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        return templates.TemplateResponse(request, "graph_new.html", {"user": user, "error": None})

    @app.post("/graphs/new")
    async def graph_new_submit(
        request: Request,
        name: Annotated[str, Form()],
        description: Annotated[str, Form()] = "",
    ) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        if not name.strip():
            return templates.TemplateResponse(
                request,
                "graph_new.html",
                {"user": user, "error": "Graph name is required"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        graph_id = uuid.uuid4().hex
        store.create_graph(graph_id, user.id, name.strip(), description.strip())
        graph_client = Neo4jClient(cfg)
        try:
            graph_client.connect()
            graph_client.initialize_schema()
        except Exception:
            pass
        finally:
            graph_client.close()
        return _redirect_with_flash("/dashboard", "success", "Graph created successfully")

    @app.get("/graphs/{graph_id}", response_class=HTMLResponse)
    async def graph_detail(graph_id: str, request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        graph = store.get_graph(graph_id, user.id)
        if graph is None:
            return templates.TemplateResponse(
                request,
                "404.html",
                {"title": "Graph not found"},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        node_count = 0
        relationship_count = 0
        viz_data = {"nodes": [], "links": []}
        type_distribution: list[dict[str, Any]] = []
        top_nodes: list[dict[str, Any]] = []
        node_types: list[str] = []
        graph_client = _graph_client()
        try:
            stats = graph_client.get_stats(graph_id=graph_id)
            node_count = stats.node_count
            relationship_count = stats.relationship_count
            node_rows = graph_client.run_cypher(
                """
                MATCH (n:Resource)
                WHERE n.graph_id = $graph_id
                OPTIONAL MATCH (n)-[r:RELATES]-()
                WITH n, count(r) AS degree
                RETURN n.id AS id, n.label AS label, n.type AS type, degree
                ORDER BY degree DESC
                LIMIT $limit
                """,
                {"graph_id": graph_id, "limit": GRAPH_VIZ_NODE_LIMIT},
                graph_id=graph_id,
            )
            type_dist_rows = graph_client.run_cypher(
                """
                MATCH (n:Resource) WHERE n.graph_id = $graph_id
                RETURN n.type AS type, count(n) AS count
                ORDER BY count DESC LIMIT 8
                """,
                {"graph_id": graph_id},
                graph_id=graph_id,
            )
            type_distribution = [
                {"type": row.get("type") or "unknown", "count": int(row.get("count", 0))} for row in type_dist_rows
            ]
            top_node_rows = graph_client.run_cypher(
                """
                MATCH (n:Resource)-[r:RELATES]-()
                WHERE n.graph_id = $graph_id
                RETURN n.id AS id, n.label AS label, n.type AS type, count(r) AS degree
                ORDER BY degree DESC LIMIT 10
                """,
                {"graph_id": graph_id},
                graph_id=graph_id,
            )
            top_nodes = [
                {
                    "id": row["id"],
                    "label": row.get("label") or row["id"],
                    "type": row.get("type", ""),
                    "degree": int(row.get("degree", 0)),
                }
                for row in top_node_rows
                if row.get("id")
            ]
            type_rows = graph_client.run_cypher(
                """
                MATCH (n:Resource) WHERE n.graph_id = $graph_id
                RETURN DISTINCT n.type AS type
                ORDER BY type
                """,
                {"graph_id": graph_id},
                graph_id=graph_id,
            )
            node_types = [row["type"] for row in type_rows if row.get("type")]
            node_ids = [row["id"] for row in node_rows if row.get("id")]
            rel_rows: list[dict[str, Any]] = []
            if node_ids:
                rel_rows = graph_client.run_cypher(
                    """
                    MATCH (a:Resource)-[r:RELATES]->(b:Resource)
                    WHERE a.graph_id = $graph_id AND b.graph_id = $graph_id
                      AND a.id IN $node_ids AND b.id IN $node_ids
                    RETURN a.id AS source, b.id AS target, r.type AS type
                    """,
                    {"graph_id": graph_id, "node_ids": node_ids},
                    graph_id=graph_id,
                )
            viz_data = {
                "nodes": [
                    {
                        "id": row["id"],
                        "label": row.get("label") or row["id"],
                        "type": row.get("type", ""),
                        "degree": int(row.get("degree", 0)),
                    }
                    for row in node_rows
                    if row.get("id")
                ],
                "links": [
                    {"source": row["source"], "target": row["target"], "type": row.get("type", "RELATES")}
                    for row in rel_rows
                    if row.get("source") and row.get("target")
                ],
            }
        except Exception:
            pass
        finally:
            graph_client.close()
        return templates.TemplateResponse(
            request,
            "graph_detail.html",
            {
                "user": user,
                "graph": graph,
                "node_count": node_count,
                "relationship_count": relationship_count,
                "viz_data_json": json.dumps(viz_data),
                "page_size": NODES_PAGE_SIZE,
                "viz_limit": GRAPH_VIZ_NODE_LIMIT,
                "type_colors": TYPE_COLORS,
                "default_type_color": DEFAULT_TYPE_COLOR,
                "type_distribution": type_distribution,
                "top_nodes": top_nodes,
                "node_types": node_types,
            },
        )

    @app.get("/graphs/{graph_id}/api/nodes")
    async def graph_nodes_api(
        graph_id: str,
        request: Request,
        offset: int = 0,
        limit: int = NODES_PAGE_SIZE,
        q: str | None = None,
        type: str | None = None,
    ) -> Response:
        user = request.state.user
        if user is None:
            return JSONResponse({"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)
        graph = store.get_graph(graph_id, user.id)
        if graph is None:
            return JSONResponse({"error": "Graph not found"}, status_code=status.HTTP_404_NOT_FOUND)
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        search_q = (q or "").strip()
        type_filter = (type or "").strip()
        graph_client = _graph_client()
        try:
            where_clauses = ["n.graph_id = $graph_id"]
            params: dict[str, Any] = {"graph_id": graph_id, "offset": offset, "limit": limit}
            if search_q:
                where_clauses.append("toLower(n.label) CONTAINS toLower($q)")
                params["q"] = search_q
            if type_filter:
                where_clauses.append("n.type = $type")
                params["type"] = type_filter
            where_sql = " AND ".join(where_clauses)
            rows = graph_client.run_cypher(
                f"""
                MATCH (n:Resource)
                WHERE {where_sql}
                RETURN n.id AS id, n.type AS type, n.label AS label,
                       properties(n) AS properties, n.ingested_at AS ingested_at
                ORDER BY n.ingested_at DESC
                SKIP $offset LIMIT $limit
                """,
                params,
                graph_id=graph_id,
            )
            total_row = graph_client.run_cypher(
                f"MATCH (n:Resource) WHERE {where_sql} RETURN count(n) AS total",
                {k: v for k, v in params.items() if k not in ("offset", "limit")},
                graph_id=graph_id,
            )
            total = int(total_row[0]["total"]) if total_row else 0
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            graph_client.close()
        nodes = [_serialize_node_row(row) for row in rows]
        return JSONResponse(
            {
                "nodes": nodes,
                "offset": offset,
                "limit": limit,
                "total": total,
                "has_more": offset + len(nodes) < total,
            }
        )

    @app.get("/graphs/{graph_id}/api/neighbors/{node_id}")
    async def graph_neighbors_api(graph_id: str, node_id: str, request: Request) -> Response:
        user = request.state.user
        if user is None:
            return JSONResponse({"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)
        graph = store.get_graph(graph_id, user.id)
        if graph is None:
            return JSONResponse({"error": "Graph not found"}, status_code=status.HTTP_404_NOT_FOUND)
        graph_client = _graph_client()
        try:
            node_rows = graph_client.run_cypher(
                """
                MATCH (n:Resource {id: $node_id})
                WHERE n.graph_id = $graph_id
                RETURN n.id AS id, n.type AS type, n.label AS label,
                       properties(n) AS properties, n.ingested_at AS ingested_at
                """,
                {"graph_id": graph_id, "node_id": node_id},
                graph_id=graph_id,
            )
            if not node_rows:
                return JSONResponse({"error": "Node not found"}, status_code=status.HTTP_404_NOT_FOUND)
            rel_rows = graph_client.run_cypher(
                """
                MATCH (n:Resource {id: $node_id})-[r:RELATES]-(m:Resource)
                WHERE n.graph_id = $graph_id AND m.graph_id = $graph_id
                RETURN n.id AS source_id, m.id AS target_id, r.type AS type,
                       properties(r) AS properties,
                       m.id AS neighbor_id, m.type AS neighbor_type,
                       m.label AS neighbor_label, properties(m) AS neighbor_properties,
                       m.ingested_at AS neighbor_ingested_at,
                       startNode(r).id AS rel_start, endNode(r).id AS rel_end
                """,
                {"graph_id": graph_id, "node_id": node_id},
                graph_id=graph_id,
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            graph_client.close()
        node = _serialize_node_row(node_rows[0])
        relationships: list[dict[str, Any]] = []
        neighbors: list[dict[str, Any]] = []
        seen_neighbors: set[str] = set()
        for row in rel_rows:
            neighbor_id = row.get("neighbor_id", "")
            if neighbor_id and neighbor_id not in seen_neighbors:
                seen_neighbors.add(neighbor_id)
                neighbor_props = row.get("neighbor_properties") or {}
                if not isinstance(neighbor_props, dict):
                    neighbor_props = dict(neighbor_props)
                for key in ("id", "type", "label", "graph_id", "ingested_at", "embedding"):
                    neighbor_props.pop(key, None)
                neighbors.append(
                    {
                        "id": neighbor_id,
                        "type": row.get("neighbor_type", ""),
                        "label": row.get("neighbor_label", ""),
                        "properties": neighbor_props,
                        "created_at": row.get("neighbor_ingested_at") or "",
                    }
                )
            rel_props = row.get("properties") or {}
            if not isinstance(rel_props, dict):
                rel_props = dict(rel_props)
            relationships.append(
                {
                    "source_id": row.get("rel_start") or row.get("source_id", ""),
                    "target_id": row.get("rel_end") or row.get("target_id", ""),
                    "type": row.get("type", "RELATES"),
                    "properties": rel_props,
                }
            )
        return JSONResponse({"node": node, "relationships": relationships, "neighbors": neighbors})

    @app.post("/graphs/{graph_id}/rename")
    async def graph_rename(
        graph_id: str,
        request: Request,
        name: Annotated[str, Form()],
    ) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        if not name.strip():
            return _redirect_with_flash("/graphs", "error", "Graph name is required")
        updated = store.update_graph(graph_id, user.id, name=name.strip())
        if updated is None:
            return templates.TemplateResponse(
                request,
                "404.html",
                {"title": "Graph not found"},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return _redirect_with_flash("/graphs", "success", "Graph renamed successfully")

    @app.post("/graphs/{graph_id}/delete")
    async def graph_delete(graph_id: str, request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        graph = store.get_graph(graph_id, user.id)
        if graph is None:
            return templates.TemplateResponse(
                request,
                "404.html",
                {"title": "Graph not found"},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        _delete_graph_data(graph_id)
        store.delete_graph(graph_id, user.id)
        return _redirect_with_flash("/dashboard", "success", "Graph deleted successfully")

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        return templates.TemplateResponse(
            request,
            "settings.html",
            {"user": user, "error": None},
        )

    @app.post("/settings/change-password")
    async def settings_change_password(
        request: Request,
        current_password: Annotated[str, Form()],
        new_password: Annotated[str, Form()],
        confirm_password: Annotated[str, Form()],
    ) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        if not verify_password(current_password, user.password_hash):
            return templates.TemplateResponse(
                request,
                "settings.html",
                {"user": user, "error": "Current password is incorrect"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        if len(new_password) < 8:
            return templates.TemplateResponse(
                request,
                "settings.html",
                {"user": user, "error": "New password must be at least 8 characters"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        if new_password != confirm_password:
            return templates.TemplateResponse(
                request,
                "settings.html",
                {"user": user, "error": "New passwords do not match"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        store.update_user(user.id, password_hash=hash_password(new_password))
        return _redirect_with_flash("/settings", "success", "Password changed successfully")

    @app.post("/settings/delete-account")
    async def settings_delete_account(request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        for graph in store.list_graphs(user.id):
            _delete_graph_data(graph.id)
        store.delete_user(user.id)
        response = _redirect_with_flash("/", "success", "Your account has been deleted")
        _clear_session_cookie(response, request)
        return response

    @app.get("/agents", response_class=HTMLResponse)
    async def agents_list(request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        agents = store.list_agents(user.id)
        return templates.TemplateResponse(request, "agents.html", {"user": user, "agents": agents})

    @app.post("/agents/{agent_id}/revoke")
    async def agent_revoke(agent_id: str, request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        store.revoke_agent(agent_id, user.id)
        return _redirect_with_flash("/agents", "success", "Agent revoked successfully")

    @app.get("/agents/new", response_class=HTMLResponse)
    async def agent_new_form(request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        graphs = store.list_graphs(user.id)
        return templates.TemplateResponse(
            request,
            "agent_new.html",
            {"user": user, "graphs": graphs, "error": None},
        )

    @app.post("/agents/new")
    async def agent_new_submit(
        request: Request,
        name: Annotated[str, Form()],
        graph_ids: Annotated[list[str] | None, Form()] = None,
    ) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        if not name.strip():
            graphs = store.list_graphs(user.id)
            return templates.TemplateResponse(
                request,
                "agent_new.html",
                {"user": user, "graphs": graphs, "error": "Agent name is required"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        selected_graphs = graph_ids or []
        for graph_id in selected_graphs:
            if store.get_graph(graph_id, user.id) is None:
                graphs = store.list_graphs(user.id)
                return templates.TemplateResponse(
                    request,
                    "agent_new.html",
                    {"user": user, "graphs": graphs, "error": "Invalid graph selection"},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
        raw_key = generate_agent_key()
        agent = store.create_agent(
            uuid.uuid4().hex,
            user.id,
            name.strip(),
            hash_agent_key(raw_key),
            key_prefix(raw_key),
            selected_graphs or None,
        )
        return templates.TemplateResponse(
            request,
            "agent_created.html",
            {"user": user, "agent": agent, "raw_key": raw_key},
        )

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_page(request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        agents = [agent for agent in store.list_agents(user.id) if not agent.revoked]
        base_url = str(request.base_url).rstrip("/")
        mcp_url = f"{base_url}/mcp"
        placeholder_key = "kg_YOUR_AGENT_KEY_HERE"
        agent_key = placeholder_key
        if agents:
            agent_key = f"{agents[0].key_prefix}… (use your saved key from agent creation)"
        return templates.TemplateResponse(
            request,
            "setup.html",
            {
                "user": user,
                "mcp_url": mcp_url,
                "agent_key": agent_key,
                "placeholder_key": placeholder_key,
            },
        )

    return app


__all__ = ["create_app"]
