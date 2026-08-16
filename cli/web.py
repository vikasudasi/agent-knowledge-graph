"""Web server CLI commands."""

from __future__ import annotations

from typing import Annotated

import typer
import uvicorn
from rich.console import Console

web_app = typer.Typer(help="Web dashboard and multi-user MCP API")


@web_app.command("serve")
def serve(
    host: Annotated[str, typer.Option(help="Bind host")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port")] = 8000,
) -> None:
    """Start the FastAPI web dashboard and authenticated MCP endpoint."""
    try:
        from web.app import create_app
    except ImportError as exc:
        Console().print('[red]Web dependencies not installed. Run: pip install -e ".[web]"[/]')
        raise typer.Exit(1) from exc

    app = create_app()
    Console().print(f"[green]Starting web server on http://{host}:{port}[/]")
    Console().print(f"[dim]MCP endpoint: http://{host}:{port}/mcp[/]")
    uvicorn.run(app, host=host, port=port)
