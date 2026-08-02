"""Neo4j graph client — connection, schema, CRUD, and query operations."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

from neo4j import Driver, GraphDatabase

from core.config import KGConfig
from core.models import GraphStats, PipelineCheckpoint, QueryResult, Relationship, Resource


class Neo4jClient:
    """Manage a Neo4j connection pool and provide high-level graph operations."""

    def __init__(self, config: KGConfig) -> None:
        self._config = config
        self._driver: Driver | None = None

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

    def initialize_schema(self) -> None:
        """Create constraints, indexes, and vector index."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Resource) REQUIRE r.id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (r:Resource) ON (r.type)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (r:Resource) ON (r.label)")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:PipelineCheckpoint) REQUIRE c.pipeline_name IS UNIQUE")

            dimension = self._config.embedding.dimension
            session.run("DROP INDEX resource_embedding IF EXISTS")
            session.run(
                "CREATE VECTOR INDEX resource_embedding IF NOT EXISTS "
                "FOR (r:Resource) ON (r.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: $dimension, "
                "`vector.similarity_function`: 'cosine'}}",
                {"dimension": dimension},
            )

    def drop_schema(self) -> None:
        """Remove all constraints and indexes."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            constraints = session.run("SHOW CONSTRAINTS")
            for record in constraints:
                name = record.get("name")
                if name:
                    session.run(f"DROP CONSTRAINT {name} IF EXISTS")

            indexes = session.run("SHOW INDEXES")
            for record in indexes:
                name = record.get("name")
                if name:
                    session.run(f"DROP INDEX {name} IF EXISTS")

    def health_check(self) -> bool:
        """Check if Neo4j is reachable."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def upsert_resource(self, resource: Resource) -> None:
        """Merge a Resource node by id."""
        query = """
        MERGE (r:Resource {id: $id})
        ON CREATE SET
            r.type = $type,
            r.label = $label,
            r.properties_json = $properties_json,
            r.ingested_at = $ingested_at
        ON MATCH SET
            r.type = $type,
            r.label = $label,
            r.properties_json = $properties_json
        """
        with self.driver.session(database=self._config.neo4j.database) as session:
            session.run(
                query,
                {
                    "id": resource.id,
                    "type": resource.type,
                    "label": resource.label,
                    "properties_json": json.dumps(resource.properties or {}),
                    "ingested_at": (resource.ingested_at or datetime.now(UTC)).isoformat(),
                },
            )
            if resource.embedding is not None:
                session.run(
                    "MATCH (r:Resource {id: $id}) SET r.embedding = $embedding",
                    {"id": resource.id, "embedding": resource.embedding},
                )

    def upsert_resources_batch(self, resources: list[Resource]) -> None:
        """Upsert multiple Resource nodes."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            for resource in resources:
                session.run(
                    """
                    MERGE (r:Resource {id: $id})
                    ON CREATE SET
                        r.type = $type,
                        r.label = $label,
                        r.properties_json = $properties_json,
                        r.ingested_at = $ingested_at
                    ON MATCH SET
                        r.type = $type,
                        r.label = $label,
                        r.properties_json = $properties_json
                    """,
                    {
                        "id": resource.id,
                        "type": resource.type,
                        "label": resource.label,
                        "properties_json": json.dumps(resource.properties or {}),
                        "ingested_at": (resource.ingested_at or datetime.now(UTC)).isoformat(),
                    },
                )
                if resource.embedding is not None:
                    session.run(
                        "MATCH (r:Resource {id: $id}) SET r.embedding = $embedding",
                        {"id": resource.id, "embedding": resource.embedding},
                    )

    def get_resource(self, resource_id: str) -> Resource | None:
        """Fetch a Resource by id."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            result = session.run("MATCH (r:Resource {id: $id}) RETURN r", {"id": resource_id})
            record = result.single()
            if record is None:
                return None
            node = record["r"]
            raw_props = node.get("properties_json")
            if isinstance(raw_props, str):
                try:
                    parsed = json.loads(raw_props)
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = {}
            return Resource(
                id=node.get("id", resource_id),
                type=node.get("type", "unknown"),
                label=node.get("label", ""),
                properties=parsed,
                embedding=node.get("embedding"),
            )

    def upsert_relationship(self, rel: Relationship) -> None:
        """Merge a RELATES relationship between two Resource nodes."""
        query = """
        MATCH (a:Resource {id: $source_id})
        MATCH (b:Resource {id: $target_id})
        MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
        ON CREATE SET r.properties_json = $properties_json
        ON MATCH SET r.properties_json = $properties_json
        """
        with self.driver.session(database=self._config.neo4j.database) as session:
            session.run(
                query,
                {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "rel_type": rel.type,
                    "properties_json": json.dumps(rel.properties or {}),
                },
            )

    def upsert_relationships_batch(self, relationships: list[Relationship]) -> None:
        """Upsert multiple relationships."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            for rel in relationships:
                session.run(
                    """
                    MATCH (a:Resource {id: $source_id})
                    MATCH (b:Resource {id: $target_id})
                    MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
                    ON CREATE SET r.properties_json = $properties_json
                    ON MATCH SET r.properties_json = $properties_json
                    """,
                    {
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "rel_type": rel.type,
                        "properties_json": json.dumps(rel.properties or {}),
                    },
                )

    @staticmethod
    def _extract_properties(record_node: dict[str, Any] | Any) -> dict[str, Any]:
        """Extract properties from a Neo4j node/relationship record, handling JSON serialization."""
        raw = record_node.get("properties_json")
        if isinstance(raw, str):
            try:
                return dict(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                return {}
        return dict(raw or {})

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        type_filter: str | None = None,
    ) -> QueryResult:
        """Run semantic search through Neo4j vector index using SEARCH clause."""
        cypher = "MATCH (n:Resource)\nSEARCH n IN ( VECTOR INDEX resource_embedding FOR $query_embedding LIMIT $top_k )\nSCORE AS score"
        if type_filter:
            cypher += "\nWHERE n.type = $type_filter"
        cypher += "\nRETURN n, score ORDER BY score DESC"

        params: dict[str, Any] = {"top_k": top_k, "query_embedding": query_embedding}
        if type_filter:
            params["type_filter"] = type_filter

        t0 = time.monotonic()
        resources: list[Resource] = []
        scores: list[float] = []
        with self.driver.session(database=self._config.neo4j.database) as session:
            for record in session.run(cypher, params):
                node = record["n"]
                resources.append(
                    Resource(
                        id=node.get("id", ""),
                        type=node.get("type", "unknown"),
                        label=node.get("label", ""),
                        properties=self._extract_properties(node),
                    )
                )
                scores.append(float(record["score"]))

        elapsed = (time.monotonic() - t0) * 1000
        return QueryResult(nodes=resources, scores=scores, execution_time_ms=elapsed)

    def traverse(
        self,
        start_id: str,
        hops: int = 1,
        rel_types: list[str] | None = None,
        direction: str = "both",
    ) -> QueryResult:
        """Traverse graph neighborhood from a start node."""
        if direction not in {"both", "incoming", "outgoing"}:
            direction = "both"

        # Neo4j does not allow parameters in the variable-length relationship
        # quantifier ([*1..$hops] is rejected) — the depth must be a literal.
        try:
            hops = int(hops)
        except (TypeError, ValueError):
            hops = 1
        hops = max(1, min(hops, 10))  # guard against unbounded deep traversals

        if direction == "outgoing":
            pattern = f"-[r:RELATES*1..{hops}]->"
        elif direction == "incoming":
            pattern = f"<-[r:RELATES*1..{hops}]-"
        else:
            pattern = f"-[r:RELATES*1..{hops}]-"

        cypher = (
            "MATCH path = (start:Resource {id: $start_id})"
            f"{pattern}(end:Resource) "
            "RETURN nodes(path) AS nodes, relationships(path) AS rels"
        )

        t0 = time.monotonic()
        seen_nodes: dict[str, Resource] = {}
        seen_rels: list[Relationship] = []

        with self.driver.session(database=self._config.neo4j.database) as session:
            for record in session.run(cypher, {"start_id": start_id}):
                for node in record["nodes"]:
                    node_id = node.get("id", "")
                    if node_id and node_id not in seen_nodes:
                        seen_nodes[node_id] = Resource(
                            id=node_id,
                            type=node.get("type", "unknown"),
                            label=node.get("label", ""),
                            properties=self._extract_properties(node),
                        )
                for rel in record["rels"]:
                    rel_type = rel.get("type")
                    if rel_types and rel_type not in rel_types:
                        continue
                    seen_rels.append(
                        Relationship(
                            source_id=rel.get("source_id", ""),
                            target_id=rel.get("target_id", ""),
                            type=rel_type or "RELATES",
                            properties=self._extract_properties(rel),
                        )
                    )

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
        """Run vector search with optional post-filter using SEARCH clause."""
        cypher = "MATCH (n:Resource)\nSEARCH n IN ( VECTOR INDEX resource_embedding FOR $query_embedding LIMIT $top_k )\nSCORE AS score"
        if cypher_filter:
            cypher += f"\nWHERE {cypher_filter}"
        cypher += "\nWITH n, score ORDER BY score DESC LIMIT $top_k\nRETURN n, score"

        t0 = time.monotonic()
        resources: list[Resource] = []
        scores: list[float] = []
        with self.driver.session(database=self._config.neo4j.database) as session:
            for record in session.run(cypher, {"query_embedding": query_embedding, "top_k": top_k}):
                node = record["n"]
                resources.append(
                    Resource(
                        id=node.get("id", ""),
                        type=node.get("type", "unknown"),
                        label=node.get("label", ""),
                        properties=self._extract_properties(node),
                    )
                )
                scores.append(float(record["score"]))

        elapsed = (time.monotonic() - t0) * 1000
        return QueryResult(nodes=resources, scores=scores, execution_time_ms=elapsed)

    def run_cypher(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute raw Cypher and return row dicts."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def get_stats(self) -> GraphStats:
        """Return node/relationship counts, vector-index state, and checkpoints."""
        stats = GraphStats()
        with self.driver.session(database=self._config.neo4j.database) as session:
            node_count = session.run("MATCH (r:Resource) RETURN count(r) AS count").single()
            rel_count = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS count").single()

            if node_count is not None:
                stats.node_count = int(node_count["count"])
            if rel_count is not None:
                stats.relationship_count = int(rel_count["count"])

            for record in session.run("SHOW INDEXES WHERE name = 'resource_embedding'"):
                # Neo4j returns the index state uppercase ("ONLINE") — compare
                # case-insensitively so a healthy index isn't reported as false.
                stats.vector_index_ready = str(record.get("state", "")).lower() == "online"

            for record in session.run("MATCH (c:PipelineCheckpoint) RETURN c"):
                node = record["c"]
                checkpoint = PipelineCheckpoint(
                    pipeline_name=node.get("pipeline_name", ""),
                    last_processed_id=node.get("last_processed_id", ""),
                    total_processed=node.get("total_processed", 0),
                    updated_at=node.get("updated_at"),
                )
                stats.last_checkpoints[checkpoint.pipeline_name] = checkpoint

        return stats

    def get_checkpoint(self, pipeline_name: str) -> PipelineCheckpoint | None:
        """Fetch the checkpoint for a pipeline."""
        query = "MATCH (c:PipelineCheckpoint {pipeline_name: $name}) RETURN c"
        with self.driver.session(database=self._config.neo4j.database) as session:
            record = session.run(query, {"name": pipeline_name}).single()
            if record is None:
                return None
            node = record["c"]
            return PipelineCheckpoint(
                pipeline_name=node.get("pipeline_name", pipeline_name),
                last_processed_id=node.get("last_processed_id", ""),
                last_processed_timestamp=node.get("last_processed_timestamp"),
                total_processed=node.get("total_processed", 0),
                updated_at=node.get("updated_at"),
            )

    def save_checkpoint(self, checkpoint: PipelineCheckpoint) -> None:
        """Upsert a pipeline checkpoint."""
        query = """
        MERGE (c:PipelineCheckpoint {pipeline_name: $name})
        SET c.last_processed_id = $last_id,
            c.last_processed_timestamp = $last_processed_timestamp,
            c.total_processed = $total,
            c.updated_at = $updated_at
        """
        with self.driver.session(database=self._config.neo4j.database) as session:
            session.run(
                query,
                {
                    "name": checkpoint.pipeline_name,
                    "last_id": checkpoint.last_processed_id,
                    "last_processed_timestamp": (
                        checkpoint.last_processed_timestamp.isoformat() if checkpoint.last_processed_timestamp else None
                    ),
                    "total": checkpoint.total_processed,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
