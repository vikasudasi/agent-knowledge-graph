"""Neo4j graph client — connection, schema, CRUD, and query operations."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

from neo4j import Driver, GraphDatabase

from core.config import KGConfig
from core.models import GraphStats, PipelineCheckpoint, QueryResult, Relationship, Resource

DEFAULT_GRAPH_ID = "default"


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

    @staticmethod
    def _graph_filter_clause(
        graph_id: str | None = None,
        graph_ids: list[str] | None = None,
        *,
        node_alias: str = "n",
        prefix: str = "WHERE",
    ) -> tuple[str, dict[str, Any]]:
        """Build graph_id scoping clause. None graph_id preserves legacy unscoped behavior."""
        if graph_ids:
            if len(graph_ids) == 1:
                return f"{prefix} {node_alias}.graph_id = $graph_id", {"graph_id": graph_ids[0]}
            return f"{prefix} {node_alias}.graph_id IN $graph_ids", {"graph_ids": graph_ids}
        if graph_id is not None:
            if graph_id == DEFAULT_GRAPH_ID:
                return (
                    f"{prefix} ({node_alias}.graph_id IS NULL OR {node_alias}.graph_id = $graph_id)",
                    {"graph_id": DEFAULT_GRAPH_ID},
                )
            return f"{prefix} {node_alias}.graph_id = $graph_id", {"graph_id": graph_id}
        return "", {}

    def initialize_schema(self) -> None:
        """Create constraints, indexes, and vector index."""
        with self.driver.session(database=self._config.neo4j.database) as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Resource) REQUIRE r.id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (r:Resource) ON (r.type)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (r:Resource) ON (r.label)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (r:Resource) ON (r.graph_id)")
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

    def upsert_resource(self, resource: Resource, graph_id: str | None = None) -> None:
        """Merge a Resource node by id."""
        graph_set = ""
        params: dict[str, Any] = {
            "id": resource.id,
            "type": resource.type,
            "label": resource.label,
            "properties_json": json.dumps(resource.properties or {}),
            "ingested_at": (resource.ingested_at or datetime.now(UTC)).isoformat(),
        }
        if graph_id is not None:
            graph_set = ", r.graph_id = $graph_id"
            params["graph_id"] = graph_id
        query = f"""
        MERGE (r:Resource {{id: $id}})
        ON CREATE SET
            r.type = $type,
            r.label = $label,
            r.properties_json = $properties_json,
            r.ingested_at = $ingested_at{graph_set}
        ON MATCH SET
            r.type = $type,
            r.label = $label,
            r.properties_json = $properties_json{graph_set}
        """
        with self.driver.session(database=self._config.neo4j.database) as session:
            session.run(
                query,
                params,
            )
            if resource.embedding is not None:
                session.run(
                    "MATCH (r:Resource {id: $id}) SET r.embedding = $embedding",
                    {"id": resource.id, "embedding": resource.embedding},
                )

    def upsert_resources_batch(self, resources: list[Resource], graph_id: str | None = None) -> None:
        """Upsert multiple Resource nodes."""
        graph_set = ""
        if graph_id is not None:
            graph_set = ", r.graph_id = $graph_id"
        with self.driver.session(database=self._config.neo4j.database) as session:
            for resource in resources:
                params: dict[str, Any] = {
                    "id": resource.id,
                    "type": resource.type,
                    "label": resource.label,
                    "properties_json": json.dumps(resource.properties or {}),
                    "ingested_at": (resource.ingested_at or datetime.now(UTC)).isoformat(),
                }
                if graph_id is not None:
                    params["graph_id"] = graph_id
                session.run(
                    f"""
                    MERGE (r:Resource {{id: $id}})
                    ON CREATE SET
                        r.type = $type,
                        r.label = $label,
                        r.properties_json = $properties_json,
                        r.ingested_at = $ingested_at{graph_set}
                    ON MATCH SET
                        r.type = $type,
                        r.label = $label,
                        r.properties_json = $properties_json{graph_set}
                    """,
                    params,
                )
                if resource.embedding is not None:
                    session.run(
                        "MATCH (r:Resource {id: $id}) SET r.embedding = $embedding",
                        {"id": resource.id, "embedding": resource.embedding},
                    )

    def get_resource(
        self,
        resource_id: str,
        graph_id: str | None = None,
        graph_ids: list[str] | None = None,
    ) -> Resource | None:
        """Fetch a Resource by id."""
        graph_clause, graph_params = self._graph_filter_clause(graph_id, graph_ids, node_alias="r", prefix="AND")
        query = f"MATCH (r:Resource {{id: $id}}){graph_clause} RETURN r"
        params: dict[str, Any] = {"id": resource_id, **graph_params}
        with self.driver.session(database=self._config.neo4j.database) as session:
            result = session.run(query, params)
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
        graph_id: str | None = None,
        graph_ids: list[str] | None = None,
    ) -> QueryResult:
        """Run semantic search through Neo4j vector index using SEARCH clause."""
        cypher = (
            "MATCH (n:Resource)\nSEARCH n IN ( VECTOR INDEX resource_embedding"
            " FOR $query_embedding LIMIT $top_k )\nSCORE AS score"
        )
        where_parts: list[str] = []
        params: dict[str, Any] = {"top_k": top_k, "query_embedding": query_embedding}
        graph_clause, graph_params = self._graph_filter_clause(graph_id, graph_ids, node_alias="n")
        if graph_clause:
            where_parts.append(graph_clause.removeprefix("WHERE ").strip())
            params.update(graph_params)
        if type_filter:
            where_parts.append("n.type = $type_filter")
            params["type_filter"] = type_filter
        if where_parts:
            cypher += "\nWHERE " + " AND ".join(where_parts)
        cypher += "\nRETURN n, score ORDER BY score DESC"

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
        graph_id: str | None = None,
        graph_ids: list[str] | None = None,
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

        graph_clause, graph_params = self._graph_filter_clause(graph_id, graph_ids, node_alias="start", prefix="WHERE")
        end_graph_clause, end_graph_params = self._graph_filter_clause(
            graph_id, graph_ids, node_alias="end", prefix="WHERE"
        )
        cypher = "MATCH (start:Resource {id: $start_id})"
        if graph_clause:
            cypher += f"\n{graph_clause}"
        cypher += f"\nMATCH path = (start){pattern}(end:Resource)"
        if end_graph_clause:
            end_condition = end_graph_clause.removeprefix("WHERE ").strip().replace("end.", "end.", 1)
            cypher += f"\nWHERE {end_condition}"
        cypher += "\nRETURN nodes(path) AS nodes, relationships(path) AS rels"
        params: dict[str, Any] = {"start_id": start_id, **graph_params, **end_graph_params}

        t0 = time.monotonic()
        seen_nodes: dict[str, Resource] = {}
        seen_rels: list[Relationship] = []

        with self.driver.session(database=self._config.neo4j.database) as session:
            for record in session.run(cypher, params):
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
                    # Neo4j Relationship objects expose their endpoints via
                    # start_node / end_node (Graph node objects), NOT via
                    # source_id/target_id properties. Fall back to properties
                    # for mocked/dict-like records.
                    start_node = getattr(rel, "start_node", None)
                    end_node = getattr(rel, "end_node", None)
                    seen_rels.append(
                        Relationship(
                            source_id=start_node.get("id", "") if start_node is not None else rel.get("source_id", ""),
                            target_id=end_node.get("id", "") if end_node is not None else rel.get("target_id", ""),
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
        graph_id: str | None = None,
        graph_ids: list[str] | None = None,
    ) -> QueryResult:
        """Run vector search with optional post-filter using SEARCH clause."""
        cypher = (
            "MATCH (n:Resource)\nSEARCH n IN ( VECTOR INDEX resource_embedding"
            " FOR $query_embedding LIMIT $top_k )\nSCORE AS score"
        )
        where_parts: list[str] = []
        params: dict[str, Any] = {"query_embedding": query_embedding, "top_k": top_k}
        graph_clause, graph_params = self._graph_filter_clause(graph_id, graph_ids, node_alias="n")
        if graph_clause:
            where_parts.append(graph_clause.removeprefix("WHERE ").strip())
            params.update(graph_params)
        if cypher_filter:
            where_parts.append(cypher_filter)
        if where_parts:
            cypher += "\nWHERE " + " AND ".join(where_parts)
        cypher += "\nWITH n, score ORDER BY score DESC LIMIT $top_k\nRETURN n, score"

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

    @staticmethod
    def _node_to_dict(node: Any) -> dict[str, Any]:
        """Convert a Neo4j Node to a plain JSON-serializable dict."""
        return {
            "id": node.get("id", ""),
            "type": node.get("type", "unknown"),
            "label": node.get("label", ""),
            "properties": Neo4jClient._extract_properties(node),
        }

    @staticmethod
    def _relationship_to_dict(rel: Any) -> dict[str, Any]:
        """Convert a Neo4j Relationship to a plain JSON-serializable dict."""
        start_node = getattr(rel, "start_node", None)
        end_node = getattr(rel, "end_node", None)
        return {
            "source_id": start_node.get("id", "") if start_node is not None else rel.get("source_id", ""),
            "target_id": end_node.get("id", "") if end_node is not None else rel.get("target_id", ""),
            "type": rel.get("type", "RELATES"),
            "properties": Neo4jClient._extract_properties(rel),
        }

    @classmethod
    def _serialize_value(cls, value: Any) -> Any:
        """Recursively convert Neo4j types to JSON-serializable Python values.

        Handles Node, Relationship, Path, lists, tuples, and dicts so that the
        MCP layer (which json.dumps the result) never receives a raw GraphObject.
        """
        from neo4j.graph import Node, Path, Relationship

        if isinstance(value, Node):
            return cls._node_to_dict(value)
        if isinstance(value, Relationship):
            return cls._relationship_to_dict(value)
        if isinstance(value, Path):
            # A path is a sequence of alternating nodes and relationships.
            return [cls._serialize_value(item) for item in value if isinstance(item, (Node, Relationship))]
        if isinstance(value, (list, tuple)):
            return [cls._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._serialize_value(item) for key, item in value.items()}
        return value

    def run_cypher(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        graph_id: str | None = None,
        graph_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute raw Cypher and return row dicts.

        Values that are Neo4j Graph objects (Node, Relationship, Path) are
        converted to plain JSON-serializable dicts so downstream callers can
        safely json.dumps the result (e.g. the MCP server entrypoint).
        """
        merged_params = dict(params or {})
        if graph_ids:
            merged_params.setdefault("graph_ids", graph_ids)
        elif graph_id is not None:
            merged_params.setdefault("graph_id", graph_id)
        with self.driver.session(database=self._config.neo4j.database) as session:
            result = session.run(cypher, merged_params)
            return [
                {key: self._serialize_value(value) for key, value in record.items()}  # type: ignore[no-untyped-call]
                for record in result
            ]

    def get_stats(
        self,
        graph_id: str | None = None,
        graph_ids: list[str] | None = None,
    ) -> GraphStats:
        """Return node/relationship counts, vector-index state, and checkpoints."""
        stats = GraphStats()
        graph_clause, graph_params = self._graph_filter_clause(graph_id, graph_ids, node_alias="r", prefix="WHERE")
        rel_graph_clause, rel_graph_params = self._graph_filter_clause(
            graph_id, graph_ids, node_alias="a", prefix="WHERE"
        )
        with self.driver.session(database=self._config.neo4j.database) as session:
            node_count = session.run(
                f"MATCH (r:Resource){graph_clause} RETURN count(r) AS count",
                graph_params,
            ).single()
            rel_count = session.run(
                f"MATCH (a:Resource)-[r:RELATES]->(b:Resource){rel_graph_clause} RETURN count(r) AS count",
                rel_graph_params,
            ).single()

            if node_count is not None:
                stats.node_count = int(node_count["count"])
            if rel_count is not None:
                stats.relationship_count = int(rel_count["count"])

            for record in session.run("SHOW INDEXES WHERE name = 'resource_embedding'"):
                # Neo4j returns the index state uppercase ("ONLINE") — compare
                # case-insensitively so a healthy index isn't reported as false.
                stats.vector_index_ready = str(record.get("state", "")).lower() == "online"

            cp_clause, cp_params = self._graph_filter_clause(graph_id, graph_ids, node_alias="c", prefix="WHERE")
            for record in session.run(f"MATCH (c:PipelineCheckpoint){cp_clause} RETURN c", cp_params):
                node = record["c"]
                checkpoint = PipelineCheckpoint(
                    pipeline_name=node.get("pipeline_name", ""),
                    last_processed_id=node.get("last_processed_id", ""),
                    total_processed=node.get("total_processed", 0),
                    updated_at=node.get("updated_at"),
                )
                stats.last_checkpoints[checkpoint.pipeline_name] = checkpoint

        return stats

    def get_checkpoint(
        self,
        pipeline_name: str,
        graph_id: str | None = None,
        graph_ids: list[str] | None = None,
    ) -> PipelineCheckpoint | None:
        """Fetch the checkpoint for a pipeline."""
        graph_clause, graph_params = self._graph_filter_clause(graph_id, graph_ids, node_alias="c", prefix="AND")
        query = f"MATCH (c:PipelineCheckpoint {{pipeline_name: $name}}){graph_clause} RETURN c"
        params: dict[str, Any] = {"name": pipeline_name, **graph_params}
        with self.driver.session(database=self._config.neo4j.database) as session:
            record = session.run(query, params).single()
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

    def save_checkpoint(self, checkpoint: PipelineCheckpoint, graph_id: str | None = None) -> None:
        """Upsert a pipeline checkpoint."""
        graph_set = ""
        params: dict[str, Any] = {
            "name": checkpoint.pipeline_name,
            "last_id": checkpoint.last_processed_id,
            "last_processed_timestamp": (
                checkpoint.last_processed_timestamp.isoformat() if checkpoint.last_processed_timestamp else None
            ),
            "total": checkpoint.total_processed,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if graph_id is not None:
            graph_set = ", c.graph_id = $graph_id"
            params["graph_id"] = graph_id
        query = f"""
        MERGE (c:PipelineCheckpoint {{pipeline_name: $name}})
        SET c.last_processed_id = $last_id,
            c.last_processed_timestamp = $last_processed_timestamp,
            c.total_processed = $total,
            c.updated_at = $updated_at{graph_set}
        """
        with self.driver.session(database=self._config.neo4j.database) as session:
            session.run(
                query,
                params,
            )
