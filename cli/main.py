"""Typer application entry point for the kg CLI."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from cli.init import run_init
from core.config import load_config

app = typer.Typer(
    name="kg",
    help="agent-knowledge-graph — Persistent knowledge for AI agents",
    rich_markup_mode="rich",
)


@app.command()
def init(
    reset: Annotated[bool, typer.Option("--reset", help="Reset Neo4j and re-initialize")] = False,
) -> None:
    """Initialize Neo4j and create schema."""
    run_init(reset=reset)


@app.command()
def build(
    pipeline: Annotated[str, typer.Argument(help="Pipeline name (sessions, files)")],
    full: Annotated[bool, typer.Option("--full", help="Full rebuild from scratch")] = False,
    limit: Annotated[int | None, typer.Option("--limit", help="Max items to process")] = None,
) -> None:
    """Run an ingestion pipeline."""
    _ = full
    _ = limit
    typer.echo(f"[kg] build {pipeline} not yet implemented")


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Natural language question")],
    cypher: Annotated[bool, typer.Option("--cypher", help="Raw Cypher mode")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="JSON output format")] = False,
) -> None:
    """Query the knowledge graph."""
    _ = question
    _ = cypher
    _ = json_output
    typer.echo("[kg] query not yet implemented")


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
            table.add_row(
                "Vector Index",
                "[green]✓ Ready[/]" if stats.vector_index_ready else "[red]✗ Not found[/]",
            )
            for name, cp in stats.last_checkpoints.items():
                table.add_row(
                    f"Checkpoint: {name}",
                    f"{cp.total_processed} items, last: {cp.last_processed_id[:20]}...",
                )
            console.print(table)
    except Exception as exc:
        Console().print(f"[red]Cannot connect to Neo4j: {exc}[/]")


if __name__ == "__main__":
    app()
