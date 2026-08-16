"""Tests for the Neo4j storage layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.graph import Neo4jClient
from core.models import PipelineCheckpoint, Relationship, Resource


@pytest.fixture
def mock_config():
    """Return a KGConfig with test values."""
    from core.config import KGConfig

    return KGConfig()


@pytest.fixture
def client(mock_config):
    """Return a Neo4jClient with a mocked driver."""
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
            assert mock_driver.called

    def test_health_check_ok(self, client):
        client._driver.verify_connectivity.return_value = True
        assert client.health_check() is True

    def test_health_check_fail(self, client):
        client._driver.verify_connectivity.side_effect = Exception("Connection refused")
        assert client.health_check() is False


class TestSchema:
    def test_initialize_schema_runs_cypher(self, client):
        client.initialize_schema()
        calls = client._driver.session.return_value.__enter__.return_value.run.call_args_list
        assert len(calls) >= 4

    def test_drop_schema(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = []
        client.drop_schema()
        assert any("SHOW CONSTRAINTS" in str(c) for c in mock_session.run.call_args_list)


class TestResourceCRUD:
    def test_upsert_resource(self, client):
        resource = Resource(
            id="test-1",
            type="session",
            label="Test Session",
            properties={"key": "value"},
            embedding=[0.1, 0.2, 0.3],
        )
        client.upsert_resource(resource)
        mock_session = client._driver.session.return_value.__enter__.return_value
        assert mock_session.run.call_count >= 2

    def test_batch_upsert(self, client):
        resources = [
            Resource(
                id=f"r{i}",
                type="entity",
                label=f"Entity {i}",
                properties={"idx": i},
                embedding=[float(i)],
            )
            for i in range(3)
        ]
        client.upsert_resources_batch(resources)
        mock_session = client._driver.session.return_value.__enter__.return_value
        assert mock_session.run.call_count == 6

    def test_get_resource_found(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value.single.return_value = {
            "r": {
                "id": "test-1",
                "type": "session",
                "label": "Found",
                "properties": {"k": "v"},
                "embedding": [0.1],
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
        rel = Relationship(
            source_id="a",
            target_id="b",
            type="references",
            properties={"weight": 0.8},
        )
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
        mock_session.run.return_value = [{"n": {"id": "r1", "type": "entity", "label": "Test"}, "score": 0.95}]
        result = client.vector_search([0.1, 0.2, 0.3], top_k=5)
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "r1"
        assert result.scores == [0.95]

    def test_vector_search_with_type_filter(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = []
        result = client.vector_search([0.1, 0.2, 0.3], top_k=5, type_filter="session")
        call_args = str(mock_session.run.call_args)
        assert "WHERE" in call_args
        assert len(result.nodes) == 0

    def test_traverse(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value

        class _FakeRel(dict):
            def __init__(self):
                super().__init__({"type": "RELATES"})
                self.start_node = {"id": "a"}
                self.end_node = {"id": "b"}

        mock_session.run.return_value = [
            {
                "nodes": [
                    {"id": "a", "type": "entity", "label": "A", "properties": {}},
                    {"id": "b", "type": "entity", "label": "B", "properties": {}},
                ],
                "rels": [_FakeRel()],
            }
        ]
        result = client.traverse("a", hops=1)
        assert len(result.nodes) == 2
        assert len(result.relationships) == 1
        assert result.relationships[0].source_id == "a"
        assert result.relationships[0].target_id == "b"

    def test_traverse_populates_edge_endpoints_from_neoj_objects(self, client):
        """Regression: endpoint IDs must be read from start_node/end_node, not
        from non-existent source_id/target_id properties."""
        mock_session = client._driver.session.return_value.__enter__.return_value

        class _FakeRel(dict):
            def __init__(self):
                super().__init__({"type": "RELATES"})
                self.start_node = {"id": "from-node"}
                self.end_node = {"id": "to-node"}

        mock_session.run.return_value = [
            {
                "nodes": [
                    {"id": "from-node", "type": "entity", "label": "From", "properties": {}},
                    {"id": "to-node", "type": "entity", "label": "To", "properties": {}},
                ],
                "rels": [_FakeRel()],
            }
        ]
        result = client.traverse("from-node", hops=1)
        assert len(result.relationships) == 1
        assert result.relationships[0].source_id == "from-node"
        assert result.relationships[0].target_id == "to-node"

    def test_hybrid_search(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = []
        result = client.hybrid_search([0.1, 0.2], cypher_filter="node.type = 'session'", top_k=5)
        assert result is not None


class TestCheckpoints:
    def test_get_checkpoint_exists(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value.single.return_value = {
            "c": {
                "pipeline_name": "sessions",
                "last_processed_id": "sess-100",
                "total_processed": 50,
            }
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
        cp = PipelineCheckpoint(pipeline_name="test", last_processed_id="last-1", total_processed=10)
        client.save_checkpoint(cp)
        mock_session = client._driver.session.return_value.__enter__.return_value
        assert mock_session.run.called


class TestStats:
    def test_get_stats(self, client):
        mock_session = client._driver.session.return_value.__enter__.return_value

        def mock_run(cypher, params=None, **kwargs):
            result = MagicMock()
            if "count(r)" in cypher:
                if "RELATES" in cypher:
                    result.single.return_value = {"count": 25}
                else:
                    result.single.return_value = {"count": 100}
            elif "SHOW INDEXES" in cypher:
                result.__iter__.return_value = [{"name": "resource_embedding", "state": "online"}]
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

    def test_run_cypher_serializes_neo4j_node(self, client):
        """Regression: raw Neo4j Node objects must be converted to plain dicts
        so the result can be json.dumps'd (MCP layer does this)."""
        import json

        from neo4j.graph import Graph, Node

        mock_session = client._driver.session.return_value.__enter__.return_value
        node = Node(
            Graph(),
            "node:meta",
            801,
            ["Resource"],
            {
                "id": "entity:meta",
                "type": "project",
                "label": "AI company",
                "properties_json": '{"canonical_name": "Meta"}',
            },
        )
        mock_session.run.return_value = [{"n": node}]
        result = client.run_cypher("MATCH (n:Resource {id:'entity:meta'}) RETURN n")
        payload = json.dumps(result)  # must not raise
        assert result[0]["n"]["id"] == "entity:meta"
        assert result[0]["n"]["type"] == "project"
        assert result[0]["n"]["label"] == "AI company"
        assert result[0]["n"]["properties"]["canonical_name"] == "Meta"
        assert "AI company" in payload

    def test_run_cypher_serializes_relationship_endpoints(self, client):
        """Regression: relationship endpoint ids come from start_node/end_node."""
        import json

        from neo4j.graph import Graph, Node, Relationship

        mock_session = client._driver.session.return_value.__enter__.return_value

        class _FakeRel(Relationship):
            def __init__(self, start_id, end_id, rtype="mentions"):
                graph = Graph()
                super().__init__(graph, "rel:1", 99, {"type": rtype, "properties_json": "{}"})
                self._start_node = Node(graph, "node:s", 1, ["Resource"], {"id": start_id})
                self._end_node = Node(graph, "node:e", 2, ["Resource"], {"id": end_id})

        mock_session.run.return_value = [{"r": _FakeRel("a", "b")}]
        result = client.run_cypher("MATCH ()-[r:RELATES]->() RETURN r LIMIT 1")
        json.dumps(result)  # must not raise
        assert result[0]["r"]["source_id"] == "a"
        assert result[0]["r"]["target_id"] == "b"
        assert result[0]["r"]["type"] == "mentions"

    def test_run_cypher_serializes_nested_and_path(self, client):
        import json

        from neo4j.graph import Graph, Node, Relationship

        mock_session = client._driver.session.return_value.__enter__.return_value

        class _FakeRel(Relationship):
            def __init__(self, start_id, end_id, rtype="mentions"):
                graph = Graph()
                super().__init__(graph, "rel:1", 99, {"type": rtype, "properties_json": "{}"})
                self._start_node = Node(graph, "node:s", 1, ["Resource"], {"id": start_id})
                self._end_node = Node(graph, "node:e", 2, ["Resource"], {"id": end_id})

        def _node(_id):
            return Node(
                Graph(),
                f"node:{_id}",
                1,
                ["Resource"],
                {"id": _id, "type": "entity", "label": _id, "properties_json": "{}"},
            )

        fake_rel = _FakeRel("a", "b")
        fake_path = [_node("a"), fake_rel, _node("b")]
        mock_session.run.return_value = [
            {
                "items": [_node("a"), fake_rel],
                "path": fake_path,
                "scalar": 42,
            }
        ]
        result = client.run_cypher("RETURN 0")
        json.dumps(result)  # must not raise
        assert result[0]["scalar"] == 42
        assert result[0]["items"][0]["id"] == "a"
        assert result[0]["items"][1]["source_id"] == "a"
        assert result[0]["path"][1]["source_id"] == "a"
