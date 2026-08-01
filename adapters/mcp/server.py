"""MCP server adapter for exposing graph memory tools."""

from __future__ import annotations


class MCPMemoryServer:
    """Expose memory query and ingest operations as MCP tools."""

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start the MCP server loop."""
        raise NotImplementedError("MCPMemoryServer.serve is not yet implemented")
