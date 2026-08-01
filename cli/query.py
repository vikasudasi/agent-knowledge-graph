"""Query commands — semantic, traversal, NL→Cypher, explain."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.config import load_config
from core.embedding import EmbeddingProviderFactory
from core.graph import Neo4jClient
from core.llm import LLMProviderFactory
from core.query import QueryEngine

app = typer.Typer(help="Query the knowledge graph")
console = Console()


def _get_engine() -> QueryEngine:
    """Build a QueryEngine from current config."""
    cfg = load_config(auto_create=False)
    graph = Neo4jClient(cfg)
    graph.connect()
    embedder = EmbeddingProviderFactory.create(cfg) if cfg.embedding.provider else None
    llm = LLMProviderFactory.create(cfg) if cfg.llm.api_key else None
    return QueryEngine(graph=graph, embedder=embedder, llm=llm)


@app.command()
def semantic(
    query: str,
    top_k: int = typer.Option(10, "--top", "-k", help="Number of results"),
    type_filter: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by Resource type",
    ),
) -> None:
    """Semantic (vector) search."""
    engine = _get_engine()
    result = engine.semantic(query, top_k=top_k, type_filter=type_filter)

    table = Table(title=f"Semantic Search: '{query}'")
    table.add_column("Score", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Label")
    table.add_column("ID")
    for i, node in enumerate(result.nodes):
        score = f"{result.scores[i]:.4f}" if result.scores else "—"
        table.add_row(score, node.type, node.label[:60], node.id[:40])
    console.print(table)
    console.print(f"[dim]{result.execution_time_ms:.0f}ms | {len(result.nodes)} results[/]")


@app.command()
def traverse(
    start_id: str,
    hops: int = typer.Option(1, "--hops", "-d", help="Traversal depth"),
    direction: str = typer.Option("both", "--dir", help="outgoing|incoming|both"),
) -> None:
    """Graph traversal from a node."""
    engine = _get_engine()
    result = engine.traverse(start_id, hops=hops, direction=direction)

    console.print(f"[bold]Traversal:[/] {start_id} ({hops} hop(s), {direction})")
    console.print(f"[dim]{len(result.nodes)} nodes, {len(result.relationships)} relationships[/]")

    table = Table()
    table.add_column("Node", style="green")
    table.add_column("Type")
    table.add_column("Relationships")
    for node in result.nodes:
        rels = [r for r in result.relationships if r.source_id == node.id or r.target_id == node.id]
        rel_summary = ", ".join(sorted(set(r.type for r in rels)))
        table.add_row(node.id[:40], node.type, rel_summary)
    console.print(table)


@app.command()
def ask(question: str = typer.Argument(..., help="Natural language question")) -> None:
    """Natural language -> Cypher -> results."""
    engine = _get_engine()
    result = engine.nl_query(question)

    panel = Panel(
        f"[bold]Question:[/] {question}\n\n"
        f"[bold]Generated Cypher:[/]\n{result.cypher}\n\n"
        f"[bold]Results:[/] {len(result.results)} rows\n"
        f"[bold]Time:[/] {result.execution_time_ms:.0f}ms\n\n"
        + (f"[red]Error:[/] {result.error}" if result.error else ""),
        title="NL Query",
    )
    console.print(panel)

    if result.results:
        table = Table()
        for key in result.results[0]:
            table.add_column(key, style="cyan")
        for row in result.results[:20]:
            table.add_row(*[str(value)[:50] for value in row.values()])
        console.print(table)


@app.command()
def explain(cypher: str = typer.Argument(..., help="Cypher query to explain")) -> None:
    """Explain a Cypher query in plain English."""
    cfg = load_config(auto_create=False)
    if not cfg.llm.api_key:
        console.print("[red]LLM not configured — set KG_LLM_API_KEY[/]")
        raise typer.Exit(1)

    llm = LLMProviderFactory.create(cfg)
    graph = Neo4jClient(cfg)
    engine = QueryEngine(graph=graph, llm=llm)
    explanation = engine.explain(cypher)
    console.print(Panel(explanation, title="Cypher Explanation"))
