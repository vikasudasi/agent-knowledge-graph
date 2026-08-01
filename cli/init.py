"""KG init command — bootstrap configuration and Neo4j."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from core.config import load_config

console = Console()


def run_init(reset: bool = False) -> None:
    """Initialize config and Neo4j.

    If reset is True, overwrite existing config with defaults.
    """

    _ = reset
    try:
        cfg = load_config(auto_create=True)
        console.print(
            Panel.fit(
                f"[bold green]✓[/] Config loaded from {cfg._config_path}\n"
                f"  LLM provider: {cfg.llm.provider}\n"
                f"  Embedding: {cfg.embedding.provider} ({cfg.embedding.local_model})\n"
                f"  Neo4j: {cfg.neo4j.uri}",
                title="agent-knowledge-graph",
            )
        )
    except Exception as exc:
        console.print(f"[bold red]✗[/] Config error: {exc}")
        raise typer.Exit(1) from exc
