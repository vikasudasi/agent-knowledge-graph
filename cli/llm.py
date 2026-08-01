"""LLM-related CLI commands for testing and debugging."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from core.config import load_config
from core.llm import LLMProviderFactory

app = typer.Typer(help="LLM provider commands")
console = Console()


@app.command()
def ping() -> None:
    """Test LLM connectivity with a simple chat completion."""
    cfg = load_config(auto_create=False)
    try:
        client = LLMProviderFactory.create(cfg)
        response = client.generate(
            "Respond with exactly: OK. Say nothing else.",
            temperature=0.1,
        )
        console.print(Panel(f"[green]{response.strip()}[/]", title="LLM Ping"))
    except Exception as exc:
        console.print(f"[red]LLM ping failed: {exc}[/]")
        raise typer.Exit(1) from exc


@app.command()
def extract() -> None:
    """Test structured extraction with a simple schema."""
    from pydantic import BaseModel, Field

    class TestExtract(BaseModel):
        name: str = Field(..., description="The person's name")
        role: str = Field(..., description="Their role or title")
        confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    cfg = load_config(auto_create=False)
    try:
        client = LLMProviderFactory.create(cfg)
        result = client.extract_structured(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Vik is the architect behind agent-knowledge-graph. He works on AI and cloud architecture."
                    ),
                }
            ],
            schema=TestExtract,
        )
        console.print(Panel(f"[green]{result.model_dump_json(indent=2)}[/]", title="Structured Extraction"))
    except Exception as exc:
        console.print(f"[red]Extraction failed: {exc}[/]")
        raise typer.Exit(1) from exc
