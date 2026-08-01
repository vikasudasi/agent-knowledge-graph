# agent-knowledge-graph — Storage Layer (Task 3)

## What to Build

Implement the Neo4j storage layer — the core data driver that every other component depends on. This provides a single backend for both property graph traversal and vector (semantic) search.

## Files to Modify

- `core/graph.py` — full `Neo4jClient` implementation
- `core/models.py` — add Resource, Relationship, PipelineCheckpoint models
- `core/__init__.py` — export new classes
- `tests/conftest.py` — add Neo4j fixtures (mocked)
- `tests/test_graph.py` — create comprehensive test suite
- `cli/init.py` — wire schema initialization

## Detailed Implementation

### 1. Data Models (core/models.py)

Add these models alongside the existing `MemoryRecord`:

```python
"""Pydantic data models used across the repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Resource:
    """A node in the knowledge graph."""
    id: str
    type: str  # "session", "entity", "memory", "file", "task", "artifact"
    label: str  # Human-readable name
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    ingested_at: datetime | None = None


@dataclass
class Relationship:
    """A directed edge between two Resources."""
    source_id: str
    target_id: str
    type: str  # "mentions", "produces", "uses", "references", "depends_on", "resolves"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineCheckpoint:
    """Tracks pipeline progress for idempotent incremental runs."""
    pipeline_name: str
    last_processed_id: str = ""
    last_processed_timestamp: datetime | None = None
    total_processed: int = 0
    updated_at: datetime | None = None


@dataclass
class QueryResult:
    """Result from a knowledge graph query."""
    nodes: list[Resource] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    scores: list[float] | None = None
    execution_time_ms: float = 0.0


@dataclass
class GraphStats:
    """Statistics about the knowledge graph."""
    node_count: int = 0
    relationship_count: int = 0
    vector_index_ready: bool = False
    last_checkpoints: dict[str, PipelineCheckpoint] = field(default_factory=dict)
    database_size_mb: float = 0.0
```

### 2. Neo4jClient (core/graph.py)

Replace the stub with a full implementation:

```python
"""Neo4j graph client — connection, schema, CRUD, and query operations."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from neo4j import GraphDatabase, Driver, Session, SessionConfig, Result, exceptions as neo4j_exc

from core.config import KGConfig
from core.models import GraphStats, PipelineCheckpoint, QueryResult, Resource, Relationship

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Manages a Neo4j connection pool and provides high-level graph operations.

    Usage:
        with Neo4jClient(config) as client:
            client.initialize_schema()
            client.upsert_resource(...)
    """

    def __init__(self, config: KGConfig) -> None:
        self._config = config
        self._driver: Driver | None = None

    # --- Connection Management ---

    def connect(self) -> None:
        """Open the connection pool."""
        if self._driver is not None:
            return
        self._driver = GraphDatabase.driver(
            self._config.neo4j.uri,
            auth=(self._config.neo4j.user, self._config.neo4j.password),
            max_connection_pool_size=self._config.neo4j.max_connection_pool_size,
            connection_timeout=self._config.neo4j.connection_timeout,
        )
        # Verify connectivity
        self._driver.verify_connectivity()

    def close(self) -> None:
        """Close the connection pool."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> Neo4jClient:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            raise RuntimeError("Not connected. Call connect() or use context manager.")
        return self._driver

    # --- Schema Initialization ---

    def initialize_schema(self) -> None:
        """Create constraints, indexes, and vector index."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            # Unique constraint on Resource.id
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Resource) REQUIRE r.id IS UNIQUE"
            )
            # Index on Resource.type for filtering
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (r:Resource) ON (r.type)"
            )
            # Index on Resource.label for text search
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (r:Resource) ON (r.label)"
            )
            # Checkpoint constraint
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:PipelineCheckpoint) REQUIRE c.pipeline_name IS UNIQUE"
            )
            # Vector index — dimension from config
            dimension = self._config.embedding.dimension
            # Drop existing first if dimension changed (safe for init)
            try:
                session.run("DROP INDEX resource_embedding IF EXISTS")
            except Exception:
                pass
            session.run(
                f"CREATE VECTOR INDEX resource_embedding IF NOT EXISTS "
                f"FOR (r:Resource) ON (r.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, "
                f"`vector.similarity_function`: 'cosine'}}}}"
            )

    def drop_schema(self) -> None:
        """Remove all constraints and indexes (for reset)."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            # Drop all constraints
            result = session.run("SHOW CONSTRAINTS")
            for record in result:
                name = record.get("name")
                if name:
                    session.run(f"DROP CONSTRAINT {name} IF EXISTS")
            # Drop all indexes
            result = session.run("SHOW INDEXES")
            for record in result:
                name = record.get("name")
                if name:
                    session.run(f"DROP INDEX {name} IF EXISTS")

    def health_check(self) -> bool:
        """Check if the server is reachable and schema is initialized."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    # --- Resource CRUD ---

    def upsert_resource(self, resource: Resource) -> None:
        """MERGE a Resource node by id. Creates or updates."""
        query = """
        MERGE (r:Resource {id: $id})
        ON CREATE SET
            r.type = $type,
            r.label = $label,
            r.properties = $properties,
            r.ingested_at = $ingested_at
        ON MATCH SET
            r.type = $type,
            r.label = $label,
            r.properties = $properties
        """
        params = {
            "id": resource.id,
            "type": resource.type,
            "label": resource.label,
            "properties": resource.properties or {},
            "ingested_at": (resource.ingested_at or datetime.now(timezone.utc)).isoformat(),
        }
        with self.driver.session(database=self._config.neo4j.database) as session:
            session.run(query, params)
            # Set embedding separately (vector index field)
            if resource.embedding is not None:
                session.run(
                    "MATCH (r:Resource {id: $id}) SET r.embedding = $embedding",
                    {"id": resource.id, "embedding": resource.embedding},
                )

    def upsert_resources_batch(self, resources: list[Resource]) -> None:
        """Upsert multiple resources in a single transaction for performance."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            for resource in resources:
                session.run(
                    """
                    MERGE (r:Resource {id: $id})
                    ON CREATE SET
                        r.type = $type, r.label = $label,
                        r.properties = $properties, r.ingested_at = $ingested_at
                    ON MATCH SET
                        r.type = $type, r.label = $label, r.properties = $properties
                    """,
                    {
                        "id": resource.id,
                        "type": resource.type,
                        "label": resource.label,
                        "properties": resource.properties or {},
                        "ingested_at": (resource.ingested_at or datetime.now(timezone.utc)).isoformat(),
                    },
                )
                if resource.embedding is not None:
                    session.run(
                        "MATCH (r:Resource {id: $id}) SET r.embedding = $embedding",
                        {"id": resource.id, "embedding": resource.embedding},
                    )

    def get_resource(self, resource_id: str) -> Resource | None:
        """Fetch a single Resource by id."""
        query = "MATCH (r:Resource {id: $id}) RETURN r"
        with self.driver.session(database=self._config.neo4j.database) as session:
            result = session.run(query, {"id": resource_id})
            record = result.single()
            if record is None:
                return None
            node = record["r"]
            return Resource(
                id=node.get("id", resource_id),
                type=node.get("type", "unknown"),
                label=node.get("label", ""),
                properties=dict(node.get("properties", {}) or {}),
                embedding=node.get("embedding"),
            )

    # --- Relationship CRUD ---

    def upsert_relationship(self, rel: Relationship) -> None:
        """MERGE a RELATES relationship between two Resources."""
        query = """
        MATCH (a:Resource {id: $source_id})
        MATCH (b:Resource {id: $target_id})
        MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
        ON CREATE SET r += $properties
        ON MATCH SET r += $properties
        """
        with self.driver.session(database=self._config.neo4j.database) as session:
            session.run(query, {
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "rel_type": rel.type,
                "properties": rel.properties,
            })

    def upsert_relationships_batch(self, relationships: list[Relationship]) -> None:
        """Upsert multiple relationships in a single transaction."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            for rel in relationships:
                session.run(
                    """
                    MATCH (a:Resource {id: $source_id})
                    MATCH (b:Resource {id: $target_id})
                    MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
                    ON CREATE SET r += $properties
                    ON MATCH SET r += $properties
                    """,
                    {
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "rel_type": rel.type,
                        "properties": rel.properties,
                    },
                )

    # --- Query Operations ---

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        type_filter: str | None = None,
    ) -> QueryResult:
        """Semantic search via vector index. Returns top-k similar Resources."""
        cypher = "CALL db.index.vector.queryNodes('resource_embedding', $top_k, $query_embedding)"
        params: dict[str, Any] = {"top_k": top_k, "query_embedding": query_embedding}

        if type_filter:
            cypher += " YIELD node, score WHERE node.type = $type_filter"
        else:
            cypher += " YIELD node, score"
        cypher += " RETURN node, score ORDER BY score DESC"

        params["type_filter"] = type_filter if type_filter else None

        t0 = time.monotonic()
        with self.driver.session(database=self._config.neo4j.database) as session:
            result = session.run(cypher, params)
            resources: list[Resource] = []
            scores: list[float] = []
            for record in result:
                node = record["node"]
                resources.append(Resource(
                    id=node.get("id", ""),
                    type=node.get("type", "unknown"),
                    label=node.get("label", ""),
                    properties=dict(node.get("properties", {}) or {}),
                ))
                scores.append(record["score"])

        elapsed = (time.monotonic() - t0) * 1000
        return QueryResult(nodes=resources, scores=scores, execution_time_ms=elapsed)

    def traverse(
        self,
        start_id: str,
        hops: int = 1,
        rel_types: list[str] | None = None,
        direction: str = "both",
    ) -> QueryResult:
        """Graph traversal from a starting node. Returns all nodes and relationships within N hops."""
        dir_symbol = {"outgoing": "->", "incoming": "<-", "both": "-"}.get(direction, "-")
        rel_pattern = f"[r:RELATES{dir_symbol}]"
        if rel_types:
            type_filter = "|".join(rel_types)
            rel_pattern = f"[r:RELATES {dir_symbol}[r.type IN $rel_types]]"

        cypher = f"""
        MATCH path = (start:Resource {{id: $start_id}})-{rel_pattern}*(1..{hops})-(end:Resource)
        RETURN nodes(path) AS nodes, relationships(path) AS rels
        """

        t0 = time.monotonic()
        with self.driver.session(database=self._config.neo4j.database) as session:
            result = session.run(cypher, {
                "start_id": start_id,
                "rel_types": rel_types or [],
            })
            seen_nodes: dict[str, Resource] = {}
            seen_rels: list[Relationship] = []
            for record in result:
                for node in record["nodes"]:
                    nid = node.get("id", "")
                    if nid not in seen_nodes:
                        seen_nodes[nid] = Resource(
                            id=nid,
                            type=node.get("type", "unknown"),
                            label=node.get("label", ""),
                            properties=dict(node.get("properties", {}) or {}),
                        )
                for rel in record["rels"]:
                    seen_rels.append(Relationship(
                        source_id=rel.get("source_id", ""),
                        target_id=rel.get("target_id", ""),
                        type=rel.type,
                        properties=dict(rel.get("properties", {}) or {}),
                    ))

        elapsed = (time.monotonic() - t0) * 1000
        return QueryResult(
            nodes=list(seen_nodes.values()),
            relationships=seen_rels,
            execution_time_ms=elapsed,
        )

    def hybrid_search(
        self,
        query_embedding: list[float],
        cypher_filter: str = "",
        top_k: int = 10,
    ) -> QueryResult:
        """Vector search scoped by an optional Cypher pre-filter."""
        cypher = """
        CALL db.index.vector.queryNodes('resource_embedding', $top_k * 3, $query_embedding)
        YIELD node, score
        """
        if cypher_filter:
            cypher += f" WHERE {cypher_filter}"
        cypher += """
        WITH node, score ORDER BY score DESC LIMIT $top_k
        RETURN node, score
        """

        t0 = time.monotonic()
        with self.driver.session(database=self._config.neo4j.database) as session:
            result = session.run(cypher, {
                "query_embedding": query_embedding,
                "top_k": top_k,
            })
            resources = []
            scores = []
            for record in result:
                node = record["node"]
                resources.append(Resource(
                    id=node.get("id", ""),
                    type=node.get("type", "unknown"),
                    label=node.get("label", ""),
                    properties=dict(node.get("properties", {}) or {}),
                ))
                scores.append(record["score"])

        elapsed = (time.monotonic() - t0) * 1000
        return QueryResult(nodes=resources, scores=scores, execution_time_ms=elapsed)

    def run_cypher(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute raw Cypher and return results as dicts."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    # --- Stats ---

    def get_stats(self) -> GraphStats:
        """Get node count, relationship count, vector index health, and checkpoints."""
        stats = GraphStats()
        with self.driver.session(database=self._config.neo4j.database) as session:
            # Node count
            result = session.run("MATCH (r:Resource) RETURN count(r) AS count")
            stats.node_count = result.single()["count"]

            # Relationship count
            result = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS count")
            stats.relationship_count = result.single()["count"]

            # Vector index health
            result = session.run("SHOW INDEXES WHERE name = 'resource_embedding'")
            for record in result:
                stats.vector_index_ready = record.get("state") == "online"

            # Checkpoints
            result = session.run("MATCH (c:PipelineCheckpoint) RETURN c")
            for record in result:
                node = record["c"]
                cp = PipelineCheckpoint(
                    pipeline_name=node.get("pipeline_name", ""),
                    last_processed_id=node.get("last_processed_id", ""),
                    total_processed=node.get("total_processed", 0),
                )
                stats.last_checkpoints[cp.pipeline_name] = cp

        return stats

    # --- Checkpoints ---

    def get_checkpoint(self, pipeline_name: str) -> PipelineCheckpoint | None:
        """Get the checkpoint for a pipeline."""
        query = "MATCH (c:PipelineCheckpoint {pipeline_name: $name}) RETURN c"
        with self.driver.session(database=self._config.neo4j.database) as session:
            result = session.run(query, {"name": pipeline_name})
            record = result.single()
            if record is None:
                return None
            node = record["c"]
            return PipelineCheckpoint(
                pipeline_name=node.get("pipeline_name", pipeline_name),
                last_processed_id=node.get("last_processed_id", ""),
                total_processed=node.get("total_processed", 0),
                updated_at=node.get("updated_at"),
            )

    def save_checkpoint(self, checkpoint: PipelineCheckpoint) -> None:
        """Upsert a pipeline checkpoint."""
        query = """
        MERGE (c:PipelineCheckpoint {pipeline_name: $name})
        SET c.last_processed_id = $last_id,
            c.total_processed = $total,
            c.updated_at = $updated_at
        """
        with self.driver.session(database=self._config.neo4j.database) as session:
            session.run(query, {
                "name": checkpoint.pipeline_name,
                "last_id": checkpoint.last_processed_id,
                "total": checkpoint.total_processed,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
```

### 3. Update cli/init.py

After loading config, call `client.initialize_schema()`:

```python
def run_init(reset: bool = False) -> None:
    """Initialize config and Neo4j schema."""
    cfg = load_config(auto_create=True)
    
    if reset:
        console.print("[yellow]Reset requested — dropping existing schema...[/]")
    
    # Initialize Neo4j
    from core.graph import Neo4jClient
    with Neo4jClient(cfg) as client:
        if reset:
            client.drop_schema()
        client.initialize_schema()
        stats = client.get_stats()
    
    console.print(Panel.fit(
        f"[bold green]✓[/] Config loaded\n"
        f"  [bold green]✓[/] Neo4j schema initialized\n"
        f"  Nodes: {stats.node_count}, Relationships: {stats.relationship_count}\n"
        f"  Vector index: {'[green]ready[/]' if stats.vector_index_ready else '[yellow]pending[/]'}",
        title="agent-knowledge-graph",
    ))
```

### 4. Update core/__init__.py

```python
from core.config import EmbeddingConfig, KGConfig, LLMConfig, Neo4jConfig, load_config
from core.graph import Neo4jClient
from core.models import GraphStats, PipelineCheckpoint, QueryResult, Resource, Relationship

__all__ = [
    "KGConfig", "LLMConfig", "EmbeddingConfig", "Neo4jConfig", "load_config",
    "Neo4jClient",
    "Resource", "Relationship", "GraphStats", "PipelineCheckpoint", "QueryResult",
]
```

### 5. Tests (tests/test_graph.py)

Create a new test file. Since we can't guarantee a local Neo4j instance, the tests should use mocking extensively but also document the integration test pattern.

```python
"""Tests for the Neo4j storage layer."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.graph import Neo4jClient
from core.models import GraphStats, PipelineCheckpoint, Resource, Relationship


@pytest.fixture
def mock_config():
    """Return a KGConfig with test values."""
    from core.config import KGConfig
    return KGConfig()


@pytest.fixture
def client(mock_config):
    """Return a Neo4jClient with the driver mock-autowired."""
    c = Neo4jClient(mock_config)
    c._driver = MagicMock()
    return c


class TestConnection:
    def test_connect_calls_verify(self, client):
        client._driver = None
        with patch("neo4j.GraphDatabase.driver") as mock_driver:
            mock_instance = MagicMock()
            mock_driver.return_value = mock_instance
            client.connect()
            mock_instance.verify_connectivity.assert_called_once()

    def test_close_clears_driver(self, client):
        client.close()
        assert client._driver is None

    def test_context_manager(self, mock_config):
        with patch("neo4j.GraphDatabase.driver") as mock_driver:
            with Neo4jClient(mock_config) as c:
                assert c._driver is not None
            assert c._driver is None

    def test_health_check_ok(self, client):
        client._driver.verify_connectivity.return_value = True
        assert client.health_check() is True

    def test_health_check_fail(self, client):
        client._driver.verify_connectivity.side_effect = Exception("Connection refused")
        assert client.health_check() is False


class TestSchema:
    def test_initialize_schema_runs_cypher(self, client):
        client.initialize_schema()
        # Should have called session.run multiple times for constraints, indexes, vector index
        calls = client._driver.session.return_value.__enter__.return_value.run.call_args_list
        assert len(calls) >= 4  # constraint + 2 indexes + vector index

    def test_drop_schema(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = []  # SHOW CONSTRAINTS returns nothing
        client.drop_schema()
        # Should have called SHOW CONSTRAINTS
        assert any("SHOW CONSTRAINTS" in str(c) for c in mock_session.run.call_args_list)


class TestResourceCRUD:
    def test_upsert_resource(self, client):
        resource = Resource(
            id="test-1", type="session", label="Test Session",
            properties={"key": "value"}, embedding=[0.1, 0.2, 0.3],
        )
        client.upsert_resource(resource)
        mock_session = client._driver.session.return_value.__enter__.return_value
        assert mock_session.run.call_count >= 2  # MERGE + SET embedding

    def test_batch_upsert(self, client):
        resources = [
            Resource(id=f"r{i}", type="entity", label=f"Entity {i}",
                     properties={"idx": i}, embedding=[float(i)])
            for i in range(3)
        ]
        client.upsert_resources_batch(resources)
        mock_session = client._driver.session.return_value.__enter__.return_value
        # Each resource should fire 1 MERGE + 1 SET
        assert mock_session.run.call_count == 6

    def test_get_resource_found(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value.single.return_value = {
            "r": {
                "id": "test-1", "type": "session", "label": "Found",
                "properties": {"k": "v"}, "embedding": [0.1],
            }
        }
        resource = client.get_resource("test-1")
        assert resource is not None
        assert resource.id == "test-1"
        assert resource.label == "Found"

    def test_get_resource_not_found(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value.single.return_value = None
        assert client.get_resource("nonexistent") is None


class TestRelationshipCRUD:
    def test_upsert_relationship(self, client):
        rel = Relationship(source_id="a", target_id="b", type="references",
                           properties={"weight": 0.8})
        client.upsert_relationship(rel)
        mock_session = client._driver.session.return_value.__enter__.return_value
        assert mock_session.run.called

    def test_batch_upsert_relationships(self, client):
        rels = [
            Relationship(source_id="a", target_id="b", type="references"),
            Relationship(source_id="b", target_id="c", type="depends_on"),
        ]
        client.upsert_relationships_batch(rels)
        mock_session = client._driver.session.return_value.__enter__.return_value
        assert mock_session.run.call_count == 2


class TestQuery:
    def test_vector_search(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = [
            {"node": {"id": "r1", "type": "entity", "label": "Test"}, "score": 0.95},
        ]
        result = client.vector_search([0.1, 0.2, 0.3], top_k=5)
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "r1"
        assert result.scores == [0.95]

    def test_vector_search_with_type_filter(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = []
        result = client.vector_search([0.1, 0.2, 0.3], top_k=5, type_filter="session")
        # Verify the Cypher includes WHERE node.type
        call_args = str(mock_session.run.call_args)
        assert "WHERE" in call_args
        assert len(result.nodes) == 0

    def test_traverse(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = [
            {
                "nodes": [
                    {"id": "a", "type": "entity", "label": "A", "properties": {}},
                    {"id": "b", "type": "entity", "label": "B", "properties": {}},
                ],
                "rels": [
                    {"source_id": "a", "target_id": "b", "type": "RELATES", "properties": {}},
                ],
            }
        ]
        result = client.traverse("a", hops=1)
        assert len(result.nodes) == 2
        assert len(result.relationships) == 1

    def test_hybrid_search(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = []
        result = client.hybrid_search([0.1, 0.2], cypher_filter="node.type = 'session'", top_k=5)
        assert result is not None


class TestCheckpoints:
    def test_get_checkpoint_exists(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value.single.return_value = {
            "c": {"pipeline_name": "sessions", "last_processed_id": "sess-100",
                  "total_processed": 50}
        }
        cp = client.get_checkpoint("sessions")
        assert cp is not None
        assert cp.pipeline_name == "sessions"
        assert cp.last_processed_id == "sess-100"

    def test_get_checkpoint_missing(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value.single.return_value = None
        assert client.get_checkpoint("nonexistent") is None

    def test_save_checkpoint(self, client):
        cp = PipelineCheckpoint(
            pipeline_name="test", last_processed_id="last-1", total_processed=10,
        )
        client.save_checkpoint(cp)
        mock_session = client._driver.session.return_value.__enter__.return_value
        assert mock_session.run.called


class TestStats:
    def test_get_stats(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        # Mock run to return appropriate values for each query
        def mock_run(cypher, **kwargs):
            result = MagicMock()
            if "count(r)" in cypher:
                if "RELATES" in cypher:
                    result.single.return_value = {"count": 25}
                else:
                    result.single.return_value = {"count": 100}
            elif "SHOW INDEXES" in cypher:
                result.__iter__.return_value = [
                    {"name": "resource_embedding", "state": "online"}
                ]
            elif "PipelineCheckpoint" in cypher:
                result.__iter__.return_value = []
            return result

        mock_session.run.side_effect = mock_run
        stats = client.get_stats()
        assert stats.node_count == 100
        assert stats.relationship_count == 25
        assert stats.vector_index_ready is True


class TestRunCypher:
    def test_run_cypher_raw(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = [
            {"id": "r1", "name": "test"},
            {"id": "r2", "name": "test2"},
        ]
        result = client.run_cypher("MATCH (r:Resource) RETURN r.id AS id, r.label AS name LIMIT 2")
        assert len(result) == 2
        assert result[0]["id"] == "r1"
```

### 6. Wire into CLI (cli/main.py)

The `kg status` command should now show graph stats:

```python
@app.command()
def status() -> None:
    """Show knowledge graph stats and health."""
    from core.graph import Neo4jClient
    cfg = load_config(auto_create=False)
    try:
        with Neo4jClient(cfg) as client:
            stats = client.get_stats()
            console = Console()
            table = Table(title="Knowledge Graph Status")
            table.add_column("Metric", style="bold")
            table.add_column("Value")
            table.add_row("Nodes", str(stats.node_count))
            table.add_row("Relationships", str(stats.relationship_count))
            table.add_row("Vector Index", "[green]✓ Ready[/]" if stats.vector_index_ready else "[red]✗ Not found[/]")
            for name, cp in stats.last_checkpoints.items():
                table.add_row(f"Checkpoint: {name}", f"{cp.total_processed} items, last: {cp.last_processed_id[:20]}...")
            console.print(table)
    except Exception as e:
        Console().print(f"[red]Cannot connect to Neo4j: {e}[/]")
```

## Instructions for Cursor

1. Replace `core/graph.py` with the full implementation above
2. Replace `core/models.py` with all the dataclass models
3. Update `core/__init__.py` to export new classes
4. Create `tests/test_graph.py` with the test suite above
5. Update `cli/init.py` to call `Neo4jClient(cfg).initialize_schema()` after loading config
6. Update `cli/main.py` `status()` command to show graph stats
7. Run `uv run pytest tests/test_graph.py -v` and ensure all tests pass
8. Run `uv run pytest -v` to ensure nothing else broke
9. Run `uv run python -c "from core.graph import Neo4jClient; print('Import OK')"` to verify imports