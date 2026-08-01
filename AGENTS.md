# Agent Adapter Guide

## 1) Hermes Plugin

The Hermes adapter exposes `KnowledgeGraphPlugin`, providing tool-call style methods:

- `kg_query(question)`
- `kg_semantic_search(query, top_k=5)`
- `kg_traverse(start_id, hops=1)`
- `kg_stats()`

### Installation

```bash
uv sync --extra hermes
```

### Usage Sketch

```python
from adapters.hermes_plugin import KnowledgeGraphPlugin

plugin = KnowledgeGraphPlugin()
print(plugin.query("What did we decide about retries?"))
```

## 2) MCP Server

MCP integration is provided by `create_mcp_handlers()` and can be mounted by your MCP runtime.

### Handler Map

- `kg_query`
- `kg_semantic_search`
- `kg_traverse`

### Lifecycle Hooks

- `close()` for graceful graph cleanup
- `get_graph()` for optional diagnostics

### Configuration Notes

Set `KG_NEO4J_*` and `KG_LLM_API_KEY` for full query features.

## 3) LangChain Tools

LangChain integration is optional and only enabled when `langchain_core` is installed.

### Installation

```bash
pip install langchain-core
```

### Available Tool Classes

- `KGQueryTool`
- `KGSemanticSearchTool`
- `KGTraverseTool`

Each tool lazily resolves a shared `QueryEngine` and returns JSON-serialized outputs.

## 4) Comparing Adapters

| Capability | Hermes Plugin | MCP Handlers | LangChain Tools |
|---|---|---|---|
| Call style | Direct Python methods | Async handler functions | Tool classes (`BaseTool`) |
| Runtime target | Hermes plugin lifecycle | MCP-compatible clients | LangChain agents and chains |
| Dependency footprint | Hermes extra | MCP runtime | `langchain-core` |
| Query ops | NL, semantic, traverse, stats | NL, semantic, traverse | NL, semantic, traverse |
| Lifecycle close hook | Implicit object lifetime | Explicit `close()` | Implicit process lifetime |

Choose Hermes for plugin-native workflows, MCP for protocol-level interoperability, and LangChain when your orchestration stack is already tool-based.
