"""kg build — run pipelines."""

from __future__ import annotations

import logging
from typing import Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from core.config import KGConfig, load_config
from core.embedding import EmbeddingProviderFactory
from core.graph import Neo4jClient
from core.llm import LLMProviderFactory
from pipelines.base import PipelineContext, PipelineRegistry
from pipelines.session import SessionIngestPipeline

app = typer.Typer(help="Build the knowledge graph from data sources")
console = Console()
logger = logging.getLogger(__name__)


def _build_context(
    config: KGConfig,
    *,
    dry_run: bool = False,
    full_rebuild: bool = False,
    limit: int | None = None,
) -> PipelineContext:
    """Build a PipelineContext from config."""
    graph = Neo4jClient(config)
    graph.connect()
    embedder = EmbeddingProviderFactory.create(config)
    llm = LLMProviderFactory.create(config)
    return PipelineContext(
        config=config,
        graph=graph,
        embedder=embedder,
        llm=llm,
        dry_run=dry_run,
        full_rebuild=full_rebuild,
        max_records=limit,
    )


def _registered_pipelines() -> list[Any]:
    """Return pipeline instances, bootstrapping defaults when empty."""
    entries = PipelineRegistry.list_pipelines()
    if entries:
        return [PipelineRegistry.get(item["name"]) for item in entries if PipelineRegistry.get(item["name"])]

    # Safety fallback in case registry side-effect imports were skipped.
    fallback = SessionIngestPipeline()
    PipelineRegistry.register(fallback)
    return [fallback]


@app.command()
def run(
    pipeline: str = typer.Argument(..., help="Pipeline name to run (or 'all')"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Run without writing to graph"),
    full_rebuild: bool = typer.Option(False, "--rebuild", "-f", help="Ignore checkpoints, full rebuild"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Max records to process"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Detailed logging"),
) -> None:
    """Run one or all registered pipelines to build the knowledge graph."""
    _ = limit

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    config = load_config(auto_create=False)

    available = _registered_pipelines()
    if pipeline == "all":
        pipelines_to_run = available
    else:
        matched = PipelineRegistry.get(pipeline)
        if not matched:
            console.print(f"[red]Pipeline '{pipeline}' not found[/]")
            names = ", ".join(p.name for p in available)
            console.print(f"Available: {names}")
            raise typer.Exit(1)
        pipelines_to_run = [matched]

    console.print(f"[bold]Running {len(pipelines_to_run)} pipeline(s)[/]")
    context = _build_context(config, dry_run=dry_run, full_rebuild=full_rebuild, limit=limit)
    try:
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
                task = progress.add_task(f"Running {pipe.name}...", total=1)
                result = pipe.run(context, full_rebuild=full_rebuild)
                progress.update(task, completed=1)

            result_table = Table(title=f"{pipe.name} — Result")
            result_table.add_column("Metric", style="cyan")
            result_table.add_column("Value", style="green")
            result_table.add_row("Processed", str(result.records_processed))
            result_table.add_row("Resources", str(result.resources_created))
            result_table.add_row("Relationships", str(result.relationships_created))
            result_table.add_row("Errors", str(result.errors))
            result_table.add_row("Duration", f"{result.duration_seconds:.2f}s")
            result_table.add_row("Checkpoint", str(result.checkpoint.last_processed_id if result.checkpoint else "—"))
            console.print(result_table)
    finally:
        context.graph.close()


@app.command(name="list-pipelines")
def list_pipelines() -> None:
    """List all registered pipelines."""
    pipes = _registered_pipelines()
    table = Table(title="Registered Pipelines")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Description")
    for p in pipes:
        table.add_row(p.name, p.version, p.description[:60])
    console.print(table)
