"""FastAPI web dashboard for multi-user knowledge graph management."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Form, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from core.config import KGConfig, load_config
from core.graph import Neo4jClient
from web.auth import (
    SESSION_COOKIE,
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

    def require_user(request: Request):  # type: ignore[no-untyped-def]
        session = request.cookies.get(SESSION_COOKIE)
        if not session:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
        user_id = sessions.read_session_token(session)
        if user_id is None:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
        user = store.get_user_by_id(user_id)
        if user is None:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
        request.state.user = user
        return user

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "home.html", {"title": "agent-knowledge-graph"})

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
        response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax")
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
        response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax")
        return response

    @app.post("/logout")
    async def logout() -> Response:
        response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
        response.delete_cookie(SESSION_COOKIE)
        return response

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request) -> Response:
        user = require_user(request)
        if isinstance(user, RedirectResponse):
            return user
        graphs = store.list_graphs(user.id)
        graph_stats: list[dict[str, Any]] = []
        graph_client = Neo4jClient(cfg)
        try:
            graph_client.connect()
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
        return templates.TemplateResponse(
            request,
            "dashboard.html",
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
        return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)

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
        return RedirectResponse("/agents", status_code=status.HTTP_303_SEE_OTHER)

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
