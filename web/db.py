"""SQLite metadata store for users, agents, and graphs."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class User:
    id: str
    email: str
    password_hash: str
    created_at: str


@dataclass
class Agent:
    id: str
    user_id: str
    name: str
    key_hash: str
    key_prefix: str
    created_at: str
    revoked: bool = False


@dataclass
class GraphRecord:
    id: str
    user_id: str
    name: str
    description: str
    created_at: str


class MetadataStore:
    """SQLite-backed store for web dashboard metadata."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    key_prefix TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS graphs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_graphs (
                    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
                    graph_id TEXT NOT NULL REFERENCES graphs(id) ON DELETE CASCADE,
                    PRIMARY KEY (agent_id, graph_id)
                );
                """
            )

    def get_meta(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def create_user(self, user_id: str, email: str, password_hash: str) -> User:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (user_id, email.lower(), password_hash, created_at),
            )
        return User(id=user_id, email=email.lower(), password_hash=password_hash, created_at=created_at)

    def get_user_by_email(self, email: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
            return None if row is None else self._row_to_user(row)

    def get_user_by_id(self, user_id: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return None if row is None else self._row_to_user(row)

    def update_user(self, user_id: str, *, password_hash: str | None = None) -> User | None:
        if password_hash is None:
            return self.get_user_by_id(user_id)
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (password_hash, user_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_user_by_id(user_id)

    def delete_user(self, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cursor.rowcount > 0

    def create_graph(self, graph_id: str, user_id: str, name: str, description: str = "") -> GraphRecord:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO graphs (id, user_id, name, description, created_at) VALUES (?, ?, ?, ?, ?)",
                (graph_id, user_id, name, description, created_at),
            )
        return GraphRecord(
            id=graph_id,
            user_id=user_id,
            name=name,
            description=description,
            created_at=created_at,
        )

    def list_graphs(self, user_id: str) -> list[GraphRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM graphs WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
            return [self._row_to_graph(row) for row in rows]

    def get_graph(self, graph_id: str, user_id: str | None = None) -> GraphRecord | None:
        with self._connect() as conn:
            if user_id is None:
                row = conn.execute("SELECT * FROM graphs WHERE id = ?", (graph_id,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM graphs WHERE id = ? AND user_id = ?",
                    (graph_id, user_id),
                ).fetchone()
            return None if row is None else self._row_to_graph(row)

    def update_graph(
        self,
        graph_id: str,
        user_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> GraphRecord | None:
        graph = self.get_graph(graph_id, user_id)
        if graph is None:
            return None
        new_name = name if name is not None else graph.name
        new_description = description if description is not None else graph.description
        with self._connect() as conn:
            conn.execute(
                "UPDATE graphs SET name = ?, description = ? WHERE id = ? AND user_id = ?",
                (new_name, new_description, graph_id, user_id),
            )
        return self.get_graph(graph_id, user_id)

    def delete_graph(self, graph_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM graphs WHERE id = ? AND user_id = ?",
                (graph_id, user_id),
            )
            return cursor.rowcount > 0

    def create_agent(
        self,
        agent_id: str,
        user_id: str,
        name: str,
        key_hash: str,
        key_prefix: str,
        graph_ids: list[str] | None = None,
    ) -> Agent:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agents (id, user_id, name, key_hash, key_prefix, created_at, revoked)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (agent_id, user_id, name, key_hash, key_prefix, created_at),
            )
            if graph_ids:
                conn.executemany(
                    "INSERT INTO agent_graphs (agent_id, graph_id) VALUES (?, ?)",
                    [(agent_id, graph_id) for graph_id in graph_ids],
                )
        return Agent(
            id=agent_id,
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            created_at=created_at,
            revoked=False,
        )

    def list_agents(self, user_id: str) -> list[Agent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agents WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
            return [self._row_to_agent(row) for row in rows]

    def get_agent_by_id(self, agent_id: str, user_id: str | None = None) -> Agent | None:
        with self._connect() as conn:
            if user_id is None:
                row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM agents WHERE id = ? AND user_id = ?",
                    (agent_id, user_id),
                ).fetchone()
            return None if row is None else self._row_to_agent(row)

    def revoke_agent(self, agent_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE agents SET revoked = 1 WHERE id = ? AND user_id = ? AND revoked = 0",
                (agent_id, user_id),
            )
            return cursor.rowcount > 0

    def get_agent_by_key_hash(self, key_hash: str) -> Agent | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agents WHERE key_hash = ? AND revoked = 0",
                (key_hash,),
            ).fetchone()
            return None if row is None else self._row_to_agent(row)

    def get_agent_graph_ids(self, agent_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT graph_id FROM agent_graphs WHERE agent_id = ?",
                (agent_id,),
            ).fetchall()
            return [str(row["graph_id"]) for row in rows]

    def resolve_agent_graph_scope(self, agent: Agent) -> list[str]:
        """Return graph IDs the agent may access (explicit scope or all user graphs)."""
        scoped = self.get_agent_graph_ids(agent.id)
        if scoped:
            return scoped
        return [graph.id for graph in self.list_graphs(agent.user_id)]

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(
            id=str(row["id"]),
            email=str(row["email"]),
            password_hash=str(row["password_hash"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _row_to_agent(row: sqlite3.Row) -> Agent:
        return Agent(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            name=str(row["name"]),
            key_hash=str(row["key_hash"]),
            key_prefix=str(row["key_prefix"]),
            created_at=str(row["created_at"]),
            revoked=bool(row["revoked"]),
        )

    @staticmethod
    def _row_to_graph(row: sqlite3.Row) -> GraphRecord:
        return GraphRecord(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            name=str(row["name"]),
            description=str(row["description"]),
            created_at=str(row["created_at"]),
        )


def metadata_db_path(data_dir: str) -> Path:
    """Return the SQLite metadata database path under storage.data_dir."""
    return Path(data_dir).expanduser() / "metadata.db"


__all__ = [
    "Agent",
    "GraphRecord",
    "MetadataStore",
    "User",
    "metadata_db_path",
]
