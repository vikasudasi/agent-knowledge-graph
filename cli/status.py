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
    except Exception as exc:
        console.print(f"[red]Failed to connect to Neo4j: {exc}[/]")
        console.print(f"[dim]URI: {config.neo4j.uri}[/]")
        raise typer.Exit(1) from exc

    try:
        healthy = graph.health_check()
        health_icon = "OK" if healthy else "FAIL"
        panel_style = "green" if healthy else "red"
        console.print(Panel(f"Neo4j Connection: {health_icon}", title="Health", style=panel_style))

        stats = graph.get_stats()
        stats_table = Table(title="Graph Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        stats_table.add_row("Nodes", str(stats.node_count))
        stats_table.add_row("Relationships", str(stats.relationship_count))
        stats_table.add_row("Vector Index", "ready" if stats.vector_index_ready else "missing")
        stats_table.add_row("Checkpoints", str(len(stats.last_checkpoints)))
        console.print(stats_table)

        if stats.last_checkpoints:
            cp_table = Table(title="Pipeline Checkpoints")
            cp_table.add_column("Pipeline", style="magenta")
            cp_table.add_column("Last Processed ID")
            cp_table.add_column("Total Processed", style="green")
            for name, cp in sorted(stats.last_checkpoints.items()):
                cp_table.add_row(name, cp.last_processed_id or "—", str(cp.total_processed))
            console.print(cp_table)
    except Exception as exc:
        console.print(f"[yellow]Could not load stats: {exc}[/]")
        raise typer.Exit(1) from exc
    finally:
        graph.close()


@app.command()
def health() -> None:
    """Quick health check (for monitoring)."""
    config = load_config(auto_create=False)
    graph = Neo4jClient(config)
    try:
        graph.connect()
        ok = graph.health_check()
    except Exception:
        console.print("unreachable")
        raise typer.Exit(1)
    finally:
        graph.close()

    if ok:
        console.print("healthy")
        raise typer.Exit(0)

    console.print("unhealthy")
    raise typer.Exit(1)
