"""kg visualize — ASCII graph visualization."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.tree import Tree

from core.config import load_config
from core.graph import Neo4jClient

app = typer.Typer(help="Visualize the knowledge graph")
console = Console()


@app.command()
def tree(
    root_id: Optional[str] = typer.Argument(None, help="Root node ID to start from"),
    depth: int = typer.Option(2, "--depth", "-d", help="Traversal depth"),
    max_children: int = typer.Option(5, "--max", "-m", help="Max children per node"),
) -> None:
    """Display the graph as a tree rooted at a node or showing summary stats."""
    config = load_config(auto_create=False)
    graph = Neo4jClient(config)
    try:
        graph.connect()
        if root_id:
            tree_root = Tree(f"[bold cyan]{root_id}[/]")
            _build_tree(graph, root_id, tree_root, depth, max_children, set())
            console.print(tree_root)
        else:
            stats = graph.get_stats()
            stats_tree = Tree("[bold]Knowledge Graph Overview[/]")
            stats_tree.add(f"[green]nodes[/]: {stats.node_count}")
            stats_tree.add(f"[green]relationships[/]: {stats.relationship_count}")
            stats_tree.add(
                f"[green]vector index[/]: {'ready' if stats.vector_index_ready else 'missing'}"
            )
            console.print(stats_tree)
    except Exception as exc:
        console.print(f"[red]Visualization failed: {exc}[/]")
        raise typer.Exit(1) from exc
    finally:
        graph.close()


def _build_tree(
    graph: Neo4jClient,
    node_id: str,
    tree_node: Tree,
    depth: int,
    max_children: int,
    visited: set[str],
) -> None:
    """Recursively expand neighbors from a root node."""
    if depth <= 0 or node_id in visited:
        return
    visited.add(node_id)

    try:
        result = graph.traverse(node_id, hops=1)
        count = 0
        for rel in result.relationships:
            if count >= max_children:
                tree_node.add("[dim]...and more[/]")
                break
            neighbor = rel.target_id if rel.source_id == node_id else rel.source_id
            child_label = f"{neighbor} [dim]({rel.type})[/]"
            child_node = tree_node.add(child_label)
            _build_tree(graph, neighbor, child_node, depth - 1, max_children, visited)
            count += 1
    except Exception:
        tree_node.add("[red]tree traversal error[/]")
