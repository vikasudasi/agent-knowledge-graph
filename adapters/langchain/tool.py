"""LangChain tool wrapper for querying graph memory."""

from __future__ import annotations


class LangChainMemoryTool:
    """Provide a callable LangChain-compatible memory lookup tool."""

    def invoke(self, question: str) -> str:
        """Resolve a natural language memory query."""
        raise NotImplementedError("LangChainMemoryTool.invoke is not yet implemented")
