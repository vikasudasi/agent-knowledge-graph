"""kg watch — continuously poll and ingest."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import typer
from rich.console import Console

from cli.build import _build_context
from core.config import load_config
from pipelines.base import PipelineRegistry
from pipelines.session import SessionIngestPipeline

app = typer.Typer(help="Watch mode — continuously poll sources and ingest")
console = Console()


def _pipeline_map() -> dict:
    entries = PipelineRegistry.list_pipelines()
    if not entries:
        fallback = SessionIngestPipeline()
        PipelineRegistry.register(fallback)
        return {fallback.name: fallback}
    return {item["name"]: pipe for item in entries if (pipe := PipelineRegistry.get(item["name"])) is not None}


@app.command()
def watch(
    interval: int = typer.Option(60, "--interval", "-i", help="Poll interval in seconds"),
    pipelines: str = typer.Option(
        "all",
        "--pipeline",
        "-p",
        help="Pipelines to watch (comma-separated or 'all')",
    ),
    once: bool = typer.Option(False, "--once", "-1", help="Run once and exit"),
) -> None:
    """Watch mode — poll sources and auto-ingest on new data."""
    config = load_config(auto_create=False)
    registered = _pipeline_map()

    pipeline_names = [p.strip() for p in pipelines.split(",")] if pipelines != "all" else None
    if pipeline_names:
        for name in pipeline_names:
            if name not in registered:
                console.print(f"[red]Pipeline '{name}' not found[/]")
                raise typer.Exit(1)
    else:
        pipeline_names = sorted(registered.keys())

    console.print(f"[bold cyan]Watch mode[/] — polling every {interval}s")
    console.print(f"Pipelines: {', '.join(pipeline_names)}")
    if once:
        console.print("[dim]Single run mode[/]")

    run_count = 0
    while True:
        run_count += 1
        now = datetime.now(UTC).strftime("%H:%M:%S")
        console.print(f"\n[bold]=== Run {run_count} — {now} ===[/]")
        context = _build_context(config)

        try:
            for name in pipeline_names:
                pipe = registered[name]
                try:
                    result = pipe.run(context)
                    if result.records_processed > 0:
                        console.print(f"  [green]{name}:[/] {result.records_processed} records, {result.errors} errors")
                    else:
                        console.print(f"  [dim]{name}:[/] no new records")
                except Exception as exc:
                    console.print(f"  [red]{name}:[/] {exc}")
        finally:
            context.graph.close()

        if once:
            break

        console.print(f"[dim]Next poll in {interval}s... Ctrl+C to stop[/]")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Watch mode stopped[/]")
            break
