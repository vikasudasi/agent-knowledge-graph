"""KG init command — bootstrap configuration and Neo4j."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from core.config import load_config
from core.graph import Neo4jClient

console = Console()


def run_init(reset: bool = False) -> None:
    """Initialize config and Neo4j schema."""
    try:
        cfg = load_config(auto_create=True)
        if reset:
            console.print("[yellow]Reset requested — dropping existing schema...[/]")

        with Neo4jClient(cfg) as client:
            if reset:
                client.drop_schema()
            client.initialize_schema()
            stats = client.get_stats()

        console.print(
            Panel.fit(
                f"[bold green]✓[/] Config loaded\n"
                f"  [bold green]✓[/] Neo4j schema initialized\n"
                f"  Nodes: {stats.node_count}, Relationships: {stats.relationship_count}\n"
                f"  Vector index: {'[green]ready[/]' if stats.vector_index_ready else '[yellow]pending[/]'}",
                title="agent-knowledge-graph",
            )
        )
    except Exception as exc:
        console.print(f"[bold red]✗[/] Config error: {exc}")
        raise typer.Exit(1) from exc
