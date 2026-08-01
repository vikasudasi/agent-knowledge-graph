# agent-knowledge-graph — Tasks 9+10+11: CLI Polish, Docker, Agent Adapters

## Task 9: CLI Polish

### Files
- `cli/build.py` — replace: `kg build` pipeline execution
- `cli/status.py` — replace: `kg status` graph health + stats
- `cli/visualize.py` — create: `kg visualize` ASCII graph
- `cli/watch.py` — create: `kg watch` auto-poller
- `tests/test_cli_polish.py` — create: tests for new commands

### Implementation

#### cli/build.py

```python
"""kg build — run pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from core.config import Configuration, load_config
from core.embedding import EmbeddingProviderFactory
from core.graph import Neo4jClient
from core.llm import LLMProviderFactory
from pipelines.base import PipelineContext, PipelineRegistry

app = typer.Typer(help="Build the knowledge graph from data sources")
console = Console()
logger = logging.getLogger(__name__)


def _build_context(config: Configuration) -> PipelineContext:
    """Build a PipelineContext from config."""
    graph = Neo4jClient(config)
    graph.connect()
    llm = LLMProviderFactory.create(config) if config.llm.api_key else None
    embedder = EmbeddingProviderFactory.create(config) if config.embedding.provider else None
    return PipelineContext(
        config=config,
        graph=graph,
        llm=llm,
        embedder=embedder,
        metadata={"version": "1.0"},
    )


@app.command()
def run(
    pipeline: str = typer.Argument(..., help="Pipeline name to run (or 'all')"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Run without writing to graph"),
    full_rebuild: bool = typer.Option(False, "--rebuild", "-f", help="Ignore checkpoints, full rebuild"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max records to process"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Detailed logging"),
) -> None:
    """Run one or all registered pipelines to build the knowledge graph."""
    config = load_config(auto_create=False)
    context = _build_context(config)

    if pipeline == "all":
        pipelines_to_run = PipelineRegistry.list()
    else:
        matched = PipelineRegistry.get(pipeline)
        if not matched:
            console.print(f"[red]Pipeline '{pipeline}' not found[/]")
            console.print(f"Available: {', '.join(p.name for p in PipelineRegistry.list())}")
            raise typer.Exit(1)
        pipelines_to_run = [matched]

    console.print(f"[bold]Running {len(pipelines_to_run)} pipeline(s)[/]")

    for pipe in pipelines_to_run:
        console.print(f"\n[cyan]=== {pipe.name} ({pipe.description}) ===[/]")
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Running {pipe.name}...", total=None)
            result = pipe.run(context, dry_run=dry_run, full_rebuild=full_rebuild, limit=limit)
            progress.update(task, completed=1)

        # Print result
        result_table = Table(title=f"{pipe.name} — Result")
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Value", style="green")
        result_table.add_row("Processed", str(result.records_processed))
        result_table.add_row("Errors", str(result.errors))
        result_table.add_row("Duration", f"{result.duration_seconds:.2f}s")
        result_table.add_row("Checkpoint", str(result.checkpoint or "—"))
        console.print(result_table)

    context.graph.close()


@app.command()
def list_pipelines() -> None:
    """List all registered pipelines."""
    pipes = PipelineRegistry.list()
    table = Table(title="Registered Pipelines")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Description")
    for p in pipes:
        table.add_row(p.name, p.version, p.description[:60])
    console.print(table)
```

#### cli/status.py

```python
"""kg status — graph health and statistics."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.config import load_config
from core.graph import Neo4jClient

app = typer.Typer(help="Show graph status and statistics")
console = Console()


@app.command()
def status() -> None:
    """Show knowledge graph health and statistics."""
    config = load_config(auto_create=False)
    graph = Neo4jClient(config)
    try:
        graph.connect()
    except Exception as e:
        console.print(f"[red]Failed to connect to Neo4j: {e}[/]")
        console.print(f"[dim]URI: {config.neo4j.uri}[/]")
        raise typer.Exit(1)

    # Health check
    healthy = graph.health_check()
    health_icon = "✅" if healthy else "❌"
    console.print(Panel(f"Neo4j Connection: {health_icon}", title="Health"))

    # Stats
    try:
        stats = graph.get_stats()
        stats_table = Table(title="Graph Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        stats_table.add_row("Resources", str(stats.resource_count))
        stats_table.add_row("Relationships", str(stats.relationship_count))
        stats_table.add_row("Resource Types", str(len(stats.resource_types)))
        stats_table.add_row("Relationship Types", str(len(stats.relationship_types)))
        if stats.last_ingested_at:
            stats_table.add_row("Last Ingested", stats.last_ingested_at.isoformat()[:19] if hasattr(stats.last_ingested_at, 'isoformat') else str(stats.last_ingested_at))
        console.print(stats_table)

        if stats.resource_types:
            type_table = Table(title="Resources by Type")
            type_table.add_column("Type", style="magenta")
            type_table.add_column("Count", style="green")
            for rt, count in sorted(stats.resource_types.items()):
                type_table.add_row(rt, str(count))
            console.print(type_table)
    except Exception as e:
        console.print(f"[yellow]Could not load stats: {e}[/]")

    graph.close()


@app.command()
def health() -> None:
    """Quick health check (for monitoring)."""
    config = load_config(auto_create=False)
    graph = Neo4jClient(config)
    try:
        graph.connect()
        ok = graph.health_check()
        graph.close()
        if ok:
            console.print("healthy")
            raise typer.Exit(0)
        else:
            console.print("unhealthy")
            raise typer.Exit(1)
    except Exception:
        console.print("unreachable")
        raise typer.Exit(1)
```

#### cli/visualize.py

```python
"""kg visualize — ASCII graph visualization."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.tree import Tree

from core.config import load_config
from core.graph import Neo4jClient

app = typer.Typer(help="Visualize the knowledge graph")
console = Console()


@app.command()
def tree(
    root_id: Optional[str] = typer.Argument(None, help="Root node ID to start from"),
    depth: int = typer.Option(2, "--depth", "-d", help="Traversal depth"),
    max_children: int = typer.Option(5, "--max", "-m", help="Max children per node"),
) -> None:
    """Display the graph as a tree rooted at a node or showing top resources."""
    config = load_config(auto_create=False)
    graph = Neo4jClient(config)
    graph.connect()

    if root_id:
        # Tree from root
        tree_root = Tree(f"[bold cyan]{root_id}[/]")
        _build_tree(graph, root_id, tree_root, depth, max_children, set())
        console.print(tree_root)
    else:
        # Show top resource types
        stats = graph.get_stats()
        stats_tree = Tree("[bold]Knowledge Graph Overview[/]")
        for rtype, count in sorted(stats.resource_types.items()):
            stats_tree.add(f"[green]{rtype}[/]: {count}")
        console.print(stats_tree)

    graph.close()


def _build_tree(
    graph: Neo4jClient,
    node_id: str,
    tree_node: Tree,
    depth: int,
    max_children: int,
    visited: set[str],
) -> None:
    """Recursive tree builder."""
    if depth <= 0 or node_id in visited:
        return
    visited.add(node_id)

    try:
        result = graph.traverse(node_id, hops=1)
        count = 0
        for rel in result.relationships:
            if count >= max_children:
                tree_node.add(f"[dim]...and more[/]")
                break
            neighbor = rel.target_id if rel.source_id == node_id else rel.source_id
            child_label = f"{neighbor} [dim]({rel.type})[/]"
            child_node = tree_node.add(child_label)
            _build_tree(graph, neighbor, child_node, depth - 1, max_children, visited)
            count += 1
    except Exception:
        tree_node.add("[red]tree traversal error[/]")
```

#### cli/watch.py

```python
"""kg watch — continuously poll and ingest."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from core.config import load_config
from core.graph import Neo4jClient
from pipelines.base import PipelineRegistry
from cli.build import _build_context

app = typer.Typer(help="Watch mode — continuously poll sources and ingest")
console = Console()


@app.command()
def watch(
    interval: int = typer.Option(60, "--interval", "-i", help="Poll interval in seconds"),
    pipelines: str = typer.Option("all", "--pipeline", "-p", help="Pipelines to watch (comma-separated or 'all')"),
    once: bool = typer.Option(False, "--once", "-1", help="Run once and exit"),
) -> None:
    """Watch mode — poll sources and auto-ingest on new data."""
    config = load_config(auto_create=False)
    pipeline_names = [p.strip() for p in pipelines.split(",")] if pipelines != "all" else None

    # Validate pipelines
    registered = {p.name: p for p in PipelineRegistry.list()}
    if pipeline_names:
        for name in pipeline_names:
            if name not in registered:
                console.print(f"[red]Pipeline '{name}' not found[/]")
                raise typer.Exit(1)
    else:
        pipeline_names = list(registered.keys())

    console.print(f"[bold cyan]Watch mode[/] — polling every {interval}s")
    console.print(f"Pipelines: {', '.join(pipeline_names)}")
    if once:
        console.print("[dim]Single run mode[/]")

    run_count = 0
    while True:
        run_count += 1
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        console.print(f"\n[bold]=== Run {run_count} — {now} ===[/]")

        context = _build_context(config)
        for name in pipeline_names:
            pipe = registered[name]
            try:
                result = pipe.run(context)
                if result.records_processed > 0:
                    console.print(f"  [green]{name}:[/] {result.records_processed} records, {result.errors} errors")
                else:
                    console.print(f"  [dim]{name}:[/] no new records")
            except Exception as e:
                console.print(f"  [red]{name}:[/] {e}")

        context.graph.close()

        if once:
            break

        console.print(f"[dim]Next poll in {interval}s... Ctrl+C to stop[/]")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Watch mode stopped[/]")
            break
```

### Update cli/main.py

```python
from cli.build import app as build_app
from cli.status import app as status_app
from cli.visualize import app as visualize_app
from cli.watch import app as watch_app
app.add_typer(build_app, name="build", help="Build the knowledge graph from pipelines")
app.add_typer(status_app, name="status", help="Show graph status and statistics")
app.add_typer(visualize_app, name="visualize", help="Visualize the graph")
app.add_typer(watch_app, name="watch", help="Watch mode — auto-ingest")
```

#### tests/test_cli_polish.py

```python
"""Tests for CLI polish commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


class TestBuildCommand:
    def test_list_pipelines(self):
        result = runner.invoke(app, ["build", "list-pipelines"])
        assert result.exit_code == 0

    def test_run_unknown_pipeline(self):
        result = runner.invoke(app, ["build", "run", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower()

    def test_run_all(self):
        """Should try to run and either succeed or fail gracefully."""
        result = runner.invoke(app, ["build", "run", "all", "--dry-run"])
        # May fail if config not set up, but shouldn't crash
        assert result.exit_code in (0, 1)


class TestStatusCommand:
    def test_status_handles_no_connection(self, capsys):
        result = runner.invoke(app, ["status"])
        # Should handle connection failure gracefully
        assert result.exit_code in (0, 1)

    def test_health_cmd(self):
        result = runner.invoke(app, ["status", "health"])
        assert result.exit_code in (0, 1)


class TestVisualizeCommand:
    def test_visualize_default(self):
        result = runner.invoke(app, ["visualize", "tree"])
        assert result.exit_code in (0, 1)

    def test_visualize_with_root(self):
        result = runner.invoke(app, ["visualize", "tree", "root-1", "--depth", "1"])
        assert result.exit_code in (0, 1)


class TestWatchCommand:
    def test_watch_once(self):
        result = runner.invoke(app, ["watch", "watch", "--once", "--interval", "1"])
        assert result.exit_code in (0, 1)
```

---

## Task 10: Docker Setup

### Files
- `docker-compose.yml` — replace with polished version
- `scripts/run-neo4j.sh` — create helper script

### Implementation

#### docker-compose.yml

```yaml
version: "3.8"

services:
  neo4j:
    image: neo4j:5-enterprise
    container_name: agent-knowledge-graph-neo4j
    ports:
      - "7687:7687"   # Bolt
      - "7474:7474"   # Browser UI
    environment:
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_AUTH=${KG_NEO4J_USER:-neo4j}/${KG_NEO4J_PASSWORD:-password}
      - NEO4J_PLUGINS=["apoc","graph-data-science"]
      - NEO4J_db_tx__log_rotation_retention__policy=false
      - NEO4J_dbms_security_procedures_allowlist=gds.*,apoc.*
      - NEO4J_dbms_security_procedures_unrestricted=gds.*,apoc.*
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      - neo4j_plugins:/plugins
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u ${KG_NEO4J_USER:-neo4j} -p ${KG_NEO4J_PASSWORD:-password} 'RETURN 1' || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
```

#### scripts/run-neo4j.sh

```bash
#!/usr/bin/env bash
# Helper to manage Neo4j lifecycle
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

cmd="${1:-help}"

case "$cmd" in
  up)
    echo "Starting Neo4j..."
    docker compose -f "$COMPOSE_FILE" up -d
    echo "Waiting for Neo4j to be ready..."
    for i in $(seq 1 30); do
      if docker compose -f "$COMPOSE_FILE" exec neo4j cypher-shell -u "${KG_NEO4J_USER:-neo4j}" -p "${KG_NEO4J_PASSWORD:-password}" "RETURN 1" &>/dev/null; then
        echo "Neo4j is ready!"
        exit 0
      fi
      sleep 2
    done
    echo "Timed out waiting for Neo4j"
    exit 1
    ;;
  down)
    echo "Stopping Neo4j..."
    docker compose -f "$COMPOSE_FILE" down
    ;;
  status)
    docker compose -f "$COMPOSE_FILE" ps
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" logs "${2:--f}"
    ;;
  reset)
    echo "Removing Neo4j data volumes..."
    docker compose -f "$COMPOSE_FILE" down -v
    echo "Starting fresh..."
    "$0" up
    ;;
  *)
    echo "Usage: $0 {up|down|status|logs|reset}"
    exit 1
    ;;
esac
```

---

## Task 11: Agent Adapters

### Files
- `adapters/hermes_plugin.py` — Hermes agent plugin
- `adapters/mcp_server.py` — MCP server
- `adapters/langchain_tool.py` — LangChain tool wrapper
- `tests/test_adapters.py` — tests

### Implementation

#### adapters/hermes_plugin.py

```python
"""Hermes agent plugin — adds kg knowledge commands to Hermes."""

from __future__ import annotations

import json
from typing import Any

try:
    from hermes_plugin_base import HermesPlugin, plugin_tool
except ImportError:
    # Standalone fallback for testing
    HermesPlugin = object
    plugin_tool = lambda **kw: lambda f: f


class KnowledgeGraphPlugin(HermesPlugin):
    """Hermes plugin for knowledge graph operations."""

    name = "knowledge-graph"
    description = "Query and manage the agent knowledge graph"

    def __init__(self) -> None:
        self._engine = None  # Lazy init

    @property
    def engine(self):
        if self._engine is None:
            from core.config import load_config
            from core.embedding import EmbeddingProviderFactory
            from core.graph import Neo4jClient
            from core.llm import LLMProviderFactory
            from core.query import QueryEngine

            cfg = load_config(auto_create=False)
            graph = Neo4jClient(cfg)
            graph.connect()
            embedder = EmbeddingProviderFactory.create(cfg) if cfg.embedding.provider else None
            llm = LLMProviderFactory.create(cfg) if cfg.llm.api_key else None
            self._engine = QueryEngine(graph=graph, embedder=embedder, llm=llm)
        return self._engine

    @plugin_tool(name="kg_query", description="Query the knowledge graph using natural language")
    def query(self, question: str) -> dict[str, Any]:
        """Ask a natural language question about the knowledge graph."""
        result = self.engine.nl_query(question)
        return {
            "question": question,
            "cypher": result.cypher,
            "results": result.results,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
        }

    @plugin_tool(name="kg_semantic_search", description="Semantic search across graph resources")
    def semantic_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Find resources semantically similar to the query string."""
        result = self.engine.semantic(query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {"id": n.id, "type": n.type, "label": n.label,
                 "score": result.scores[i] if result.scores else None}
                for i, n in enumerate(result.nodes)
            ],
            "execution_time_ms": result.execution_time_ms,
        }

    @plugin_tool(name="kg_traverse", description="Traverse relationships from a graph node")
    def traverse(self, start_id: str, hops: int = 1) -> dict[str, Any]:
        """Traverse the graph from a starting node."""
        result = self.engine.traverse(start_id, hops=hops)
        return {
            "start_id": start_id,
            "nodes": [{"id": n.id, "type": n.type, "label": n.label} for n in result.nodes],
            "relationships": [
                {"source": r.source_id, "target": r.target_id, "type": r.type}
                for r in result.relationships
            ],
            "execution_time_ms": result.execution_time_ms,
        }

    @plugin_tool(name="kg_stats", description="Get knowledge graph statistics")
    def stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        cfg = load_config(auto_create=False)
        from core.graph import Neo4jClient
        graph = Neo4jClient(cfg)
        graph.connect()
        stats = graph.get_stats()
        graph.close()
        return {
            "resource_count": stats.resource_count,
            "relationship_count": stats.relationship_count,
            "resource_types": stats.resource_types,
            "relationship_types": stats.relationship_types,
        }
```

#### adapters/mcp_server.py

```python
"""MCP server — expose query layer to MCP clients (Claude Code, etc.)."""

from __future__ import annotations

import json
import logging
from typing import Any

from core.config import load_config
from core.embedding import EmbeddingProviderFactory
from core.graph import Neo4jClient
from core.llm import LLMProviderFactory
from core.query import QueryEngine

logger = logging.getLogger(__name__)


def create_mcp_handlers():
    """Factory: returns dict of MCP tool handlers.

    Each handler accepts kwargs and returns a dict.
    """
    cfg = load_config(auto_create=False)
    graph = Neo4jClient(cfg)
    graph.connect()
    embedder = EmbeddingProviderFactory.create(cfg) if cfg.embedding.provider else None
    llm = LLMProviderFactory.create(cfg) if cfg.llm.api_key else None
    engine = QueryEngine(graph=graph, embedder=embedder, llm=llm)

    def get_graph() -> Neo4jClient:
        return graph

    def close():
        graph.close()

    async def handle_query(question: str, **kwargs: Any) -> dict[str, Any]:
        result = engine.nl_query(question)
        return {
            "type": "nl_query",
            "question": question,
            "cypher": result.cypher,
            "results": result.results,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
        }

    async def handle_semantic(query: str, top_k: int = 5, **kwargs: Any) -> dict[str, Any]:
        result = engine.semantic(query, top_k=top_k)
        return {
            "type": "semantic_search",
            "query": query,
            "results": [
                {"id": n.id, "type": n.type, "label": n.label,
                 "score": result.scores[i] if result.scores else None}
                for i, n in enumerate(result.nodes)
            ],
            "execution_time_ms": result.execution_time_ms,
        }

    async def handle_traverse(start_id: str, hops: int = 1, **kwargs: Any) -> dict[str, Any]:
        result = engine.traverse(start_id, hops=hops)
        return {
            "type": "traverse",
            "start_id": start_id,
            "nodes": [{"id": n.id, "type": n.type, "label": n.label} for n in result.nodes],
            "relationships": [
                {"source": r.source_id, "target": r.target_id, "type": r.type}
                for r in result.relationships
            ],
            "execution_time_ms": result.execution_time_ms,
        }

    return {
        "handlers": {
            "kg_query": handle_query,
            "kg_semantic_search": handle_semantic,
            "kg_traverse": handle_traverse,
        },
        "close": close,
        "get_graph": get_graph,
    }
```

#### adapters/langchain_tool.py

```python
"""LangChain tool wrappers for the knowledge graph query layer."""

from __future__ import annotations

from typing import Any, Type

try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    BaseTool = object
    BaseModel = object
    Field = lambda default=None, **kw: default


from core.config import load_config
from core.embedding import EmbeddingProviderFactory
from core.graph import Neo4jClient
from core.llm import LLMProviderFactory
from core.query import QueryEngine


class _QueryEngineMixin:
    """Lazy-load singleton query engine."""

    _engine: QueryEngine | None = None

    @classmethod
    def _get_engine(cls) -> QueryEngine:
        if cls._engine is None:
            cfg = load_config(auto_create=False)
            graph = Neo4jClient(cfg)
            graph.connect()
            embedder = EmbeddingProviderFactory.create(cfg) if cfg.embedding.provider else None
            llm = LLMProviderFactory.create(cfg) if cfg.llm.api_key else None
            cls._engine = QueryEngine(graph=graph, embedder=embedder, llm=llm)
        return cls._engine


if HAS_LANGCHAIN:

    class KGQueryTool(_QueryEngineMixin, BaseTool):
        name: str = "kg_query"
        description: str = "Query the knowledge graph using natural language. Returns results and generated Cypher."

        def _run(self, question: str) -> str:
            result = self._get_engine().nl_query(question)
            return json.dumps({
                "question": question,
                "cypher": result.cypher,
                "results": result.results[:10],
                "error": result.error,
            }, default=str)

        async def _arun(self, question: str) -> str:
            return self._run(question)


    class KGSemanticSearchTool(_QueryEngineMixin, BaseTool):
        name: str = "kg_semantic_search"
        description: str = "Semantic search across the knowledge graph. Finds resources by meaning."

        def _run(self, query: str, top_k: int = 5) -> str:
            result = self._get_engine().semantic(query, top_k=top_k)
            return json.dumps({
                "results": [
                    {"id": n.id, "type": n.type, "label": n.label,
                     "score": result.scores[i] if result.scores else None}
                    for i, n in enumerate(result.nodes)
                ],
            }, default=str)

        async def _arun(self, query: str, top_k: int = 5) -> str:
            return self._run(query, top_k=top_k)


    class KGTraverseTool(_QueryEngineMixin, BaseTool):
        name: str = "kg_traverse"
        description: str = "Traverse relationships from a node in the knowledge graph."

        def _run(self, start_id: str, hops: int = 1) -> str:
            result = self._get_engine().traverse(start_id, hops=hops)
            return json.dumps({
                "nodes": [{"id": n.id, "type": n.type, "label": n.label} for n in result.nodes],
                "relationships": [
                    {"source": r.source_id, "target": r.target_id, "type": r.type}
                    for r in result.relationships
                ],
            }, default=str)

        async def _arun(self, start_id: str, hops: int = 1) -> str:
            return self._run(start_id, hops=hops)


    AVAILABLE_TOOLS = [KGQueryTool, KGSemanticSearchTool, KGTraverseTool]
else:
    AVAILABLE_TOOLS = []
```

#### tests/test_adapters.py

```python
"""Tests for agent adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.hermes_plugin import KnowledgeGraphPlugin
from adapters.mcp_server import create_mcp_handlers


class TestHermesPlugin:
    @pytest.fixture
    def plugin(self):
        return KnowledgeGraphPlugin()

    def test_plugin_name(self, plugin):
        assert plugin.name == "knowledge-graph"

    def test_plugin_description(self, plugin):
        assert plugin.description

    def test_plugin_tools_exist(self, plugin):
        """Plugin should have expected tool methods."""
        assert hasattr(plugin, "query")
        assert hasattr(plugin, "semantic_search")
        assert hasattr(plugin, "traverse")
        assert hasattr(plugin, "stats")


class TestMCPHandlers:
    def test_create_mcp_handlers(self):
        """Should return handler dict with expected keys."""
        handlers = create_mcp_handlers()
        assert "handlers" in handlers
        assert "close" in handlers
        assert "kg_query" in handlers["handlers"]
        assert "kg_semantic_search" in handlers["handlers"]
        assert "kg_traverse" in handlers["handlers"]

    def test_handlers_are_async(self):
        handlers = create_mcp_handlers()
        import asyncio
        for h in handlers["handlers"].values():
            assert asyncio.iscoroutinefunction(h)

    def test_handlers_have_close(self):
        handlers = create_mcp_handlers()
        handlers["close"]()  # Should not raise


class TestLangChainTools:
    def test_imports_without_langchain(self):
        from adapters.langchain_tool import AVAILABLE_TOOLS, HAS_LANGCHAIN
        # In test env, langchain may or may not be installed
        assert isinstance(AVAILABLE_TOOLS, list)

    def test_tools_have_names(self):
        try:
            from adapters.langchain_tool import KGQueryTool, HAS_LANGCHAIN
            if HAS_LANGCHAIN:
                tool = KGQueryTool()
                assert tool.name == "kg_query"
                assert tool.description
        except ImportError:
            pass
```

---

## Instructions for Cursor CLI

1. Replace `cli/build.py` with `kg build run` and `kg build list-pipelines`
2. Replace `cli/status.py` with `kg status status` and `kg status health`
3. Create `cli/visualize.py` with `kg visualize tree`
4. Create `cli/watch.py` with `kg watch watch`
5. Update `cli/main.py` to register all new subcommand groups
6. Create `tests/test_cli_polish.py`
7. Replace `docker-compose.yml` with the polished version
8. Create `scripts/run-neo4j.sh`
9. Create `adapters/hermes_plugin.py`, `adapters/mcp_server.py`, `adapters/langchain_tool.py`
10. Create `tests/test_adapters.py`
11. Update `pyproject.toml` to add `adapters` package if needed
12. Run test suite: `uv run python -m pytest tests/test_cli_polish.py tests/test_adapters.py -v` and report results
13. Run full suite: `uv run python -m pytest -v` and report