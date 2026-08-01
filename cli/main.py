"""Typer application entry point for the kg CLI."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from cli.init import run_init
from cli.llm import app as llm_app
from cli.query import app as query_app
from core.config import load_config

app = typer.Typer(
    name="kg",
    help="agent-knowledge-graph — Persistent knowledge for AI agents",
    rich_markup_mode="rich",
)
app.add_typer(llm_app, name="llm", help="LLM provider commands (ping, extract)")
app.add_typer(query_app, name="query", help="Query the knowledge graph")


@app.command()
def init(
    reset: Annotated[bool, typer.Option("--reset", help="Reset Neo4j and re-initialize")] = False,
    with_docker: Annotated[
        bool,
        typer.Option("--with-docker", help="Start Neo4j via Docker Compose before init"),
    ] = False,
) -> None:
    """Initialize Neo4j and create schema."""
    run_init(reset=reset, with_docker=with_docker)


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


# ---------------------------------------------------------------------------
# Docker sub-command group
# ---------------------------------------------------------------------------

docker_app = typer.Typer(help="Manage Neo4j Docker Compose lifecycle")


@docker_app.command()
def up() -> None:
    """Start Neo4j via Docker Compose."""
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            Console().print("[green]✓ Neo4j started[/]")
        else:
            Console().print(f"[red]Failed to start Neo4j: {result.stderr.strip()}[/]")
            raise typer.Exit(1)
    except FileNotFoundError:
        Console().print("[red]docker compose not found — is Docker installed?[/]")
        raise typer.Exit(1)


@docker_app.command()
def down() -> None:
    """Stop Neo4j via Docker Compose."""
    import subprocess

    result = subprocess.run(
        ["docker", "compose", "down"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        Console().print("[green]✓ Neo4j stopped[/]")
    else:
        Console().print(f"[red]Failed to stop Neo4j: {result.stderr.strip()}[/]")
        raise typer.Exit(1)


@docker_app.command(name="status")
def _docker_status() -> None:
    """Show Neo4j container status."""
    import subprocess

    result = subprocess.run(
        ["docker", "compose", "ps", "--format", "table {{.Name}}\t{{.Status}}\t{{.Ports}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.stdout.strip():
        Console().print(result.stdout)
    else:
        Console().print("[yellow]No Neo4j containers running[/]")


@docker_app.command(name="reset")
def reset() -> None:
    """Stop Neo4j and remove volumes (data loss!)."""
    import subprocess

    Console().print("[yellow]Stopping Neo4j and removing all data volumes...[/]")
    result = subprocess.run(
        ["docker", "compose", "down", "-v"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        Console().print("[green]✓ Neo4j stopped and volumes removed[/]")
        Console().print("[dim]Run 'kg docker up' to start fresh.[/]")
    else:
        Console().print(f"[red]Failed: {result.stderr.strip()}[/]")
        raise typer.Exit(1)


app.add_typer(docker_app, name="docker", help="Manage Neo4j Docker Compose lifecycle")

if __name__ == "__main__":
    app()