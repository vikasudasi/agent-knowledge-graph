"""Neo4j graph client abstractions for agent-knowledge-graph."""

from __future__ import annotations


class Neo4jGraphClient:
    """Execute Cypher and manage graph schema state."""

    def __init__(self, uri: str, username: str, password: str) -> None:
        self.uri = uri
        self.username = username
        self.password = password

    def run_query(self, cypher: str, parameters: dict[str, object] | None = None) -> list[dict[str, object]]:
        """Run a Cypher query and return row dictionaries."""
        raise NotImplementedError("Neo4jGraphClient.run_query is not yet implemented")
