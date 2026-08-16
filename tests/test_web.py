"""Tests for web dashboard auth, agents, graph scoping, and MCP auth."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.graph import Neo4jClient
from core.models import Resource
from web.app import create_app
from web.auth import generate_agent_key, hash_agent_key, hash_password, verify_password
from web.db import MetadataStore
from web.mcp_http import AgentKeyTokenVerifier


@pytest.fixture()
def tmp_store(tmp_path: Path) -> MetadataStore:
    return MetadataStore(tmp_path / "metadata.db")


@pytest.fixture()
def web_app(tmp_store: MetadataStore, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("web.app.metadata_db_path", lambda _data_dir: tmp_store._db_path)

    def _store_factory(_path):  # type: ignore[no-untyped-def]
        return tmp_store

    monkeypatch.setattr("web.app.MetadataStore", _store_factory)
    monkeypatch.setattr("web.app.create_mcp_apps", lambda store, config=None: _mock_mcp_apps(store))
    cfg = MagicMock()
    cfg.storage.data_dir = str(tmp_store._db_path.parent)
    with patch("web.app.load_config", return_value=cfg), patch("web.app.Neo4jClient") as mock_graph:
        mock_graph.return_value.connect.return_value = None
        mock_graph.return_value.get_stats.return_value = MagicMock(
            node_count=0,
            relationship_count=0,
            vector_index_ready=False,
            last_checkpoints={},
        )
        mock_graph.return_value.initialize_schema.return_value = None
        mock_graph.return_value.close.return_value = None
        yield create_app(cfg)


def _mock_mcp_apps(store: MetadataStore):
    from contextlib import asynccontextmanager

    from mcp.server.lowlevel.server import Server
    from starlette.applications import Starlette
    from starlette.routing import Route

    server: Server[object] = Server("test")

    async def ok(request):  # type: ignore[no-untyped-def]
        from starlette.responses import JSONResponse

        return JSONResponse({"ok": True})

    streamable = Starlette(routes=[Route("/", ok)])
    sse = Starlette(routes=[Route("/", ok)])

    @asynccontextmanager
    async def run_ctx():  # type: ignore[no-untyped-def]
        yield

    session_manager = MagicMock()
    session_manager.run = run_ctx
    server._session_manager = session_manager  # noqa: SLF001
    return streamable, sse, server


@pytest.mark.asyncio
async def test_signup_login_logout_flow(web_app, tmp_store: MetadataStore) -> None:
    transport = ASGITransport(app=web_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        signup = await client.post(
            "/signup",
            data={
                "email": "user@example.com",
                "password": "secret123",
                "confirm_password": "secret123",
            },
            follow_redirects=False,
        )
        assert signup.status_code == 303
        assert signup.headers["location"] == "/dashboard"
        assert "kg_session" in signup.cookies

        logout = await client.post("/logout", cookies=signup.cookies, follow_redirects=False)
        assert logout.status_code == 303
        assert logout.headers["location"] == "/"

        login = await client.post(
            "/login",
            data={"email": "user@example.com", "password": "secret123"},
            follow_redirects=False,
        )
        assert login.status_code == 303
        assert "kg_session" in login.cookies

        dashboard = await client.get("/dashboard", cookies=login.cookies)
        assert dashboard.status_code == 200
        assert "Your graphs" in dashboard.text


@pytest.mark.asyncio
async def test_signup_rejects_duplicate_email(web_app) -> None:
    transport = ASGITransport(app=web_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "email": "dup@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
        }
        first = await client.post("/signup", data=payload, follow_redirects=False)
        assert first.status_code == 303
        second = await client.post("/signup", data=payload, follow_redirects=False)
        assert second.status_code == 400
        assert "already exists" in second.text


def test_agent_key_hash_round_trip() -> None:
    raw = generate_agent_key()
    assert raw.startswith("kg_")
    assert len(raw) == 43
    hashed = hash_agent_key(raw)
    assert hashed != raw
    assert hash_agent_key(raw) == hashed


def test_password_hash_and_verify() -> None:
    hashed = hash_password("secret123")
    assert verify_password("secret123", hashed)
    assert not verify_password("wrong", hashed)


@pytest.mark.asyncio
async def test_agent_create_and_revoke(web_app, tmp_store: MetadataStore) -> None:
    user = tmp_store.create_user(uuid.uuid4().hex, "agent@example.com", hash_password("secret123"))
    graph = tmp_store.create_graph(uuid.uuid4().hex, user.id, "Test Graph")

    transport = ASGITransport(app=web_app)
    token = web_app.state  # noqa: F841
    from web.auth import SessionManager, get_or_create_session_secret

    sessions = SessionManager(get_or_create_session_secret(tmp_store))
    cookie = sessions.create_session_token(user.id)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create = await client.post(
            "/agents/new",
            data={"name": "My Agent", "graph_ids": graph.id},
            cookies={"kg_session": cookie},
        )
        assert create.status_code == 200
        assert "kg_" in create.text
        agents = tmp_store.list_agents(user.id)
        assert len(agents) == 1
        assert agents[0].key_prefix.startswith("kg_")

        listing = await client.get("/agents", cookies={"kg_session": cookie})
        assert listing.status_code == 200
        assert "My Agent" in listing.text

        await client.post(f"/agents/{agents[0].id}/revoke", cookies={"kg_session": cookie}, follow_redirects=False)
        refreshed = tmp_store.get_agent_by_id(agents[0].id)
        assert refreshed is not None
        assert refreshed.revoked is True


class TestGraphIdScoping:
    def test_upsert_sets_graph_id(self, mock_config):
        client = Neo4jClient(mock_config)
        client._driver = MagicMock()
        resource = Resource(id="r1", type="entity", label="One", properties={})
        client.upsert_resource(resource, graph_id="graph-abc")
        mock_session = client._driver.session.return_value.__enter__.return_value
        query = str(mock_session.run.call_args_list[0])
        assert "graph_id" in query

    def test_vector_search_applies_graph_filter_after_search(self, mock_config):
        client = Neo4jClient(mock_config)
        client._driver = MagicMock()
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = []
        client.vector_search([0.1, 0.2], top_k=5, graph_id="graph-xyz")
        cypher = str(mock_session.run.call_args[0][0])
        assert "SEARCH" in cypher
        assert "graph_id" in cypher
        assert cypher.index("SEARCH") < cypher.index("graph_id")

    def test_unscoped_vector_search_unchanged(self, mock_config):
        client = Neo4jClient(mock_config)
        client._driver = MagicMock()
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = []
        client.vector_search([0.1, 0.2], top_k=5)
        cypher = str(mock_session.run.call_args[0][0])
        assert "graph_id" not in cypher

    def test_get_stats_scoped(self, mock_config):
        client = Neo4jClient(mock_config)
        client._driver = MagicMock()
        mock_session = client._driver.session.return_value.__enter__.return_value

        def mock_run(cypher, params=None):
            result = MagicMock()
            if "count(r)" in cypher:
                result.single.return_value = {"count": 3}
            elif "SHOW INDEXES" in cypher:
                result.__iter__.return_value = []
            elif "PipelineCheckpoint" in cypher:
                result.__iter__.return_value = []
            return result

        mock_session.run.side_effect = mock_run
        stats = client.get_stats(graph_id="scoped-graph")
        assert stats.node_count == 3
        first_call = str(mock_session.run.call_args_list[0][0][0])
        assert "graph_id" in first_call


@pytest.fixture()
def mock_config():
    from core.config import KGConfig

    return KGConfig()


@pytest.mark.asyncio
async def test_mcp_auth_rejects_invalid_key(tmp_store: MetadataStore) -> None:
    verifier = AgentKeyTokenVerifier(tmp_store)
    assert await verifier.verify_token("kg_invalid") is None
    assert await verifier.verify_token("not-a-key") is None


@pytest.mark.asyncio
async def test_mcp_auth_accepts_valid_key(tmp_store: MetadataStore) -> None:
    user = tmp_store.create_user(uuid.uuid4().hex, "mcp@example.com", hash_password("secret123"))
    graph_id = uuid.uuid4().hex
    tmp_store.create_graph(graph_id, user.id, "Graph")
    raw_key = generate_agent_key()
    agent = tmp_store.create_agent(
        uuid.uuid4().hex,
        user.id,
        "MCP Agent",
        hash_agent_key(raw_key),
        raw_key[:12],
        [graph_id],
    )
    verifier = AgentKeyTokenVerifier(tmp_store)
    token = await verifier.verify_token(raw_key)
    assert token is not None
    assert token.client_id == agent.id
    assert token.subject == user.id

    tmp_store.revoke_agent(agent.id, user.id)
    assert await verifier.verify_token(raw_key) is None
