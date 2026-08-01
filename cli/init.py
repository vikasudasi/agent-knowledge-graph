"""KG init command — bootstrap configuration and Neo4j."""

from __future__ import annotations

import shutil
import subprocess

import typer
from rich.console import Console
from rich.panel import Panel

from core.config import load_config
from core.graph import Neo4jClient

console = Console()


def _docker_compose_up() -> bool:
    """Run docker compose up -d in the project root.

    Returns True if Docker Compose was started successfully or was already
    running, False if Docker Compose is not available.
    """
    if not shutil.which("docker") and not shutil.which("docker compose"):
        console.print("[yellow]Docker not found — skipping auto-start.[/]")
        return False

    try:
        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            console.print("[green]✓ Neo4j container started via Docker Compose[/]")
            return True
        else:
            console.print(f"[red]Docker Compose error: {result.stderr.strip()}[/]")
            return False
    except FileNotFoundError:
        console.print("[yellow]docker compose command not found — skipping.[/]")
        return False
    except subprocess.TimeoutExpired:
        console.print("[yellow]docker compose timed out — continuing anyway.[/]")
        return True


def run_init(reset: bool = False, with_docker: bool = False) -> None:
    """Initialize config and Neo4j schema.

    Args:
        reset: If True, drop existing schema before re-initializing.
        with_docker: If True, attempt to start Neo4j via Docker Compose first.
    """
    try:
        if with_docker:
            console.print("[dim]Checking Docker Compose availability...[/]")
            started = _docker_compose_up()
            if not started:
                console.print("[yellow]Proceeding with existing Neo4j instance (if any).[/]")

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