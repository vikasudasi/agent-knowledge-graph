# agent-knowledge-graph — Task 7: Session-Ingest Pipeline + Task 8: Query Layer

## What to Build

Two major layers in one go:

**Task 7 (Session-Ingest):** Reads Hermes session DB (SQLite FTS5), extracts entities/relations via LLM, generates embeddings, upserts into Neo4j.

**Task 8 (Query Layer):** Semantic search, graph traversal, hybrid search, and NL→Cypher translation.

---

# PART A: Session-Ingest Pipeline

## Files
- `pipelines/session.py` — main pipeline implementation  
- `core/extraction_schema.py` — Pydantic models for LLM extraction output  
- `tests/test_session_pipeline.py` — tests  

## Implementation

### core/extraction_schema.py

```python
"""Pydantic schemas for LLM entity/relation extraction."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """A single entity extracted from a conversation."""
    name: str = Field(..., description="Canonical entity name")
    type: str = Field(..., description="person|project|tool|concept|file|task|skill|artifact")
    label: str = Field(..., description="Human-readable short label")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)
    context: str = Field(default="", description="Why this entity was extracted")


class ExtractedRelation(BaseModel):
    """A relationship between two entities."""
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    type: str = Field(..., description="mentions|produces|uses|decides|references|blocks|resolves|assigns")
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    context: str = Field(default="", description="Evidence/snippet from conversation")


class ExtractedKnowledge(BaseModel):
    """Complete extraction result for a single session."""
    session_id: str = ""
    summary: str = Field(default="", description="One-paragraph session summary")
    topics: list[str] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    outcome: str = Field(default="in_progress", description="completed|in_progress|failed|unknown")
```

### pipelines/session.py

```python
"""Session-ingest pipeline — Hermes session DB → knowledge graph."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from core.extraction_schema import ExtractedEntity, ExtractedKnowledge, ExtractedRelation
from core.models import PipelineCheckpoint, Resource, Relationship
from pipelines.base import KnowledgePipeline, PipelineContext, PipelineRegistry

logger = logging.getLogger(__name__)


HERMES_DB_PATH = Path.home() / ".hermes" / "state.db"

EXTRACTION_PROMPT = """You are an AI knowledge graph extraction assistant. Analyze this agent conversation session and extract structured knowledge.

SESSION:
Title: {title}
Started: {started_at}
Messages:
{messages}

Extract the following in JSON format:
1. summary: One paragraph summarizing what happened
2. topics: Key topics discussed (3-5 tags)
3. entities: People, projects, tools, concepts, files, tasks mentioned
4. relations: Relationships between entities
5. decisions: Key decisions or conclusions
6. tools_used: Any tools or commands used
7. outcome: Was the goal completed, in progress, or failed?

Be thorough but accurate. Only extract what is clearly present in the text."""  # noqa: E501


class SessionIngestPipeline(KnowledgePipeline):
    """Reads Hermes session DB, extracts knowledge via LLM, writes to Neo4j."""

    def __init__(self) -> None:
        super().__init__(
            name="session-ingest",
            description="Extract entities, relations, and topics from Hermes agent sessions",
            version="1.0",
        )
        self._db_path: Path | None = None

    # ── Extract ─────────────────────────────────────────────────────

    def extract(
        self,
        context: PipelineContext,
        checkpoint: PipelineCheckpoint | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        db_path = self._resolve_db_path(context)
        if not db_path.exists():
            logger.warning(f"Hermes session DB not found at {db_path}")
            return

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if table exists
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sessions', 'messages')"
        ).fetchall()
        table_names = {t["name"] for t in tables}

        if "sessions" not in table_names:
            logger.warning(f"No 'sessions' table in {db_path}")
            conn.close()
            return

        # Query sessions — ordered by created_at for deterministic checkpoints
        query = "SELECT id, title, started_at, created_at FROM sessions"
        params: dict[str, Any] = {}

        if checkpoint and checkpoint.last_processed_id:
            query += " WHERE id > :checkpoint_id"
            params["checkpoint_id"] = checkpoint.last_processed_id

        query += " ORDER BY id ASC"

        for row in cursor.execute(query, params):
            session = dict(row)

            # Fetch messages for this session
            messages = []
            if "messages" in table_names:
                msg_rows = cursor.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                    (session["id"],),
                ).fetchall()
                for msg in msg_rows:
                    role = msg["role"] or "user"
                    content = (msg["content"] or "")[:500]  # Truncate long messages
                    messages.append(f"{role}: {content}")

            session["messages"] = messages
            yield session

        conn.close()

    # ── Resolve ─────────────────────────────────────────────────────

    def resolve(self, context: PipelineContext, record: dict[str, Any]) -> list[Resource]:
        """Extract knowledge from session via LLM, then convert to Resources."""
        # Build prompt
        title = record.get("title", "Untitled Session")
        started_at = record.get("started_at", "unknown")
        messages_text = "\n".join(record.get("messages", ["(no messages)"]))
        session_id = str(record["id"])

        prompt = EXTRACTION_PROMPT.format(
            title=title,
            started_at=started_at,
            messages=messages_text,
        )

        # LLM extraction
        try:
            extracted = context.llm.extract_structured(
                messages=[{"role": "user", "content": prompt}],
                schema=ExtractedKnowledge,
                system_prompt="You are a knowledge graph extraction assistant. Output ONLY valid JSON.",
                model=context.config.llm.extraction_model,
            )
            if hasattr(extracted, "model_dump"):
                extracted_data = extracted.model_dump()
            else:
                extracted_data = json.loads(json.dumps(extracted, default=str))
        except Exception as e:
            logger.warning(f"LLM extraction failed for session {session_id}: {e}")
            extracted_data = {
                "session_id": session_id,
                "summary": title,
                "entities": [],
                "relations": [],
            }

        resources: list[Resource] = []
        ingested_at = datetime.now(timezone.utc)

        # Build session resource
        session_resource = Resource(
            id=f"session:{session_id}",
            type="session",
            label=title[:200],
            properties={
                "session_id": str(session_id),
                "title": title,
                "started_at": str(started_at),
                "summary": extracted_data.get("summary", ""),
                "topics": extracted_data.get("topics", []),
                "decisions": extracted_data.get("decisions", []),
                "tools_used": extracted_data.get("tools_used", []),
                "outcome": extracted_data.get("outcome", "unknown"),
                "message_count": len(record.get("messages", [])),
            },
            ingested_at=ingested_at,
        )
        resources.append(session_resource)

        # Build entity resources
        entities = extracted_data.get("entities", [])
        for ent in entities:
            if isinstance(ent, dict):
                ent_id = ent.get("name", "unknown").lower().replace(" ", "-").replace("/", "-")
                ent_type = ent.get("type", "concept")
                ent_label = ent.get("label", ent.get("name", "Unknown"))
            else:
                ent_id = ent.name.lower().replace(" ", "-").replace("/", "-")
                ent_type = ent.type
                ent_label = ent.label or ent.name

            resource = Resource(
                id=f"entity:{ent_id}",
                type=ent_type,
                label=ent_label[:200],
                properties={
                    "canonical_name": ent_label if isinstance(ent, dict) else ent.name,
                    "aliases": ent.get("aliases", []) if isinstance(ent, dict) else (ent.aliases if hasattr(ent, "aliases") else []),
                    "confidence": ent.get("confidence", 0.8) if isinstance(ent, dict) else (ent.confidence if hasattr(ent, "confidence") else 0.8),
                },
                ingested_at=ingested_at,
            )
            resources.append(resource)

        return resources

    # ── Relationships ────────────────────────────────────────────────

    def get_relationships(
        self,
        context: PipelineContext,
        records: list[dict[str, Any]],
        resources: list[Resource],
    ) -> list[Relationship]:
        """Generate relationships between session and entities."""
        session_resources = [r for r in resources if r.type == "session"]
        entity_resources = [r for r in resources if r.type != "session"]

        if not session_resources:
            return []

        relationships: list[Relationship] = []
        session_id = session_resources[0].id

        # Session mentions entity
        for entity in entity_resources:
            relationships.append(Relationship(
                source_id=session_id,
                target_id=entity.id,
                type="mentions",
                properties={"weight": 1.0},
            ))

        return relationships

    # ── Helpers ──────────────────────────────────────────────────────

    def _resolve_db_path(self, context: PipelineContext) -> Path:
        if self._db_path:
            return self._db_path
        db_path = Path(
            context.metadata.get("hermes_db_path", str(HERMES_DB_PATH))
        )
        self._db_path = db_path
        return db_path

    def set_db_path(self, path: str | Path) -> None:
        """Override Hermes DB path (for testing)."""
        self._db_path = Path(path)


# Register automatically
PipelineRegistry.register(SessionIngestPipeline())
```

---

# PART B: Query Layer

## Files
- `cli/query.py` — `kg query`, `kg semantic`, `kg traverse`, `kg explain` commands
- `core/query.py` — query engine with NL→Cypher, semantic search, traversal
- `tests/test_query.py` — tests

## Implementation

### core/query.py

```python
"""Query engine — semantic, traversal, hybrid, and NL→Cypher."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from core.embedding import EmbeddingProvider
from core.graph import Neo4jClient
from core.llm import LLMClient
from core.models import QueryResult as KGQueryResult
from core.models import Relationship, Resource

logger = logging.getLogger(__name__)


NL_CYPHER_SYSTEM_PROMPT = """You are a Neo4j Cypher query generator for a knowledge graph.

SCHEMA:
- Nodes: (:Resource {id, type, label, properties, embedding})
- Relationships: (:Resource)-[:RELATES {type, weight, context}]->(:Resource)
- Resource types: session, person, project, tool, concept, file, task, skill, artifact
- Relationship types: mentions, produces, uses, decides, references, blocks, resolves, assigns

Generate ONLY the Cypher query. No explanations. Return valid Cypher only.

GUIDELINES:
- Use MATCH with labels for filtering: MATCH (r:Resource {type: 'person'})
- For text search: WHERE r.label CONTAINS 'term' OR r.properties CONTAINS 'term'
- For relationships: MATCH (a:Resource {id: $id})-[:RELATES]->(b:Resource)
- Always RETURN distinct results
- Limit results to 20 unless the user asks for more
- Use parameterized queries ($param) for user input"""  # noqa: E501


@dataclass
class NLQueryResult:
    """Result of a natural language query against the knowledge graph."""
    cypher: str = ""
    explanation: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    execution_time_ms: float = 0.0


class QueryEngine:
    """Main query interface — semantic, traversal, hybrid, NL→Cypher."""

    def __init__(
        self,
        graph: Neo4jClient,
        embedder: EmbeddingProvider | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self._graph = graph
        self._embedder = embedder
        self._llm = llm

    # ── Query methods ───────────────────────────────────────────────

    def semantic(self, query: str, top_k: int = 10, type_filter: str | None = None) -> KGQueryResult:
        """Search by semantic meaning — embed query, vector search."""
        if not self._embedder:
            raise RuntimeError("QueryEngine needs embedder for semantic search")

        t0 = time.monotonic()
        query_vec = self._embedder.embed_query(query)
        result = self._graph.vector_search(query_vec, top_k=top_k, type_filter=type_filter)
        result.execution_time_ms = (time.monotonic() - t0) * 1000
        return result

    def traverse(
        self,
        start_id: str,
        hops: int = 1,
        rel_types: list[str] | None = None,
        direction: str = "both",
    ) -> KGQueryResult:
        """Graph traversal from a node."""
        return self._graph.traverse(start_id, hops=hops, rel_types=rel_types, direction=direction)

    def hybrid(
        self,
        query: str,
        cypher_filter: str = "",
        top_k: int = 10,
    ) -> KGQueryResult:
        """Vector search + optional Cypher pre-filter."""
        if not self._embedder:
            raise RuntimeError("QueryEngine needs embedder for hybrid search")

        t0 = time.monotonic()
        query_vec = self._embedder.embed_query(query)
        result = self._graph.hybrid_search(query_vec, cypher_filter=cypher_filter, top_k=top_k)
        result.execution_time_ms = (time.monotonic() - t0) * 1000
        return result

    def nl_query(self, question: str) -> NLQueryResult:
        """Natural language → Cypher → execution."""
        if not self._llm:
            return NLQueryResult(error="QueryEngine needs LLM for NL→Cypher")

        t0 = time.monotonic()

        # Step 1: NL → Cypher
        try:
            cypher = self._llm.generate(
                prompt=f"Convert this question to a Cypher query:\n\n{question}",
                system_prompt=NL_CYPHER_SYSTEM_PROMPT,
                model=self._llm.config.query_model,
                temperature=0.1,
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return NLQueryResult(error=f"LLM translation failed: {e}", execution_time_ms=elapsed)

        # Clean up the response — strip markdown fences
        cypher = cypher.strip()
        if cypher.startswith("```"):
            cypher = cypher.strip("`")
            if cypher.startswith("cypher"):
                cypher = cypher[6:].strip()
            elif cypher.startswith("sql"):
                cypher = cypher[3:].strip()
            cypher = cypher.strip()

        if not cypher:
            elapsed = (time.monotonic() - t0) * 1000
            return NLQueryResult(error="LLM returned empty Cypher", execution_time_ms=elapsed)

        # Step 2: Execute
        try:
            raw_results = self._graph.run_cypher(cypher)
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return NLQueryResult(
                cypher=cypher,
                error=f"Cypher execution failed: {e}",
                execution_time_ms=elapsed,
            )

        elapsed = (time.monotonic() - t0) * 1000

        return NLQueryResult(
            cypher=cypher,
            explanation=f"Generated query returned {len(raw_results)} results",
            results=[dict(r) for r in raw_results],
            execution_time_ms=elapsed,
        )

    def explain(self, cypher: str) str:
        """Generate a plain-English explanation of a Cypher query."""
        if not self._llm:
            return "LLM not configured"
        try:
            return self._llm.generate(
                prompt=f"Explain this Neo4j Cypher query in plain English:\n\n{cypher}",
                system_prompt="You are a Cypher expert. Provide clear, concise explanations.",
                temperature=0.3,
            )
        except Exception as e:
            return f"Explanation failed: {e}"
```

### cli/query.py

```python
"""Query commands — semantic, traversal, NL→Cypher, explain."""

from __future__ import annotations

import json
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
    type_filter: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by Resource type"),
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
def ask(
    question: str = typer.Argument(..., help="Natural language question"),
) -> None:
    """Natural language → Cypher → results."""
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
        if result.results:
            for key in result.results[0]:
                table.add_column(key, style="cyan")
            for row in result.results[:20]:
                table.add_row(*[str(v)[:50] for v in row.values()])
        console.print(table)


@app.command()
def explain(
    cypher: str = typer.Argument(..., help="Cypher query to explain"),
) -> None:
    """Explain a Cypher query in plain English."""
    from core.llm import LLMProviderFactory
    cfg = load_config(auto_create=False)
    if not cfg.llm.api_key:
        console.print("[red]LLM not configured — set KG_LLM_API_KEY[/]")
        raise typer.Exit(1)
    llm = LLMProviderFactory.create(cfg)
    engine = QueryEngine(graph=MagicMock(), llm=llm)
    explanation = engine.explain(cypher)
    console.print(Panel(explanation, title=f"Cypher Explanation"))
```

### Update cli/main.py — add query subcommand

```python
from cli.query import app as query_app
app.add_typer(query_app, name="query", help="Query the knowledge graph")
```

### Update core/__init__.py

```python
from core.query import NLQueryResult, QueryEngine
```

### tests/test_session_pipeline.py

```python
"""Tests for the session-ingest pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.extraction_schema import ExtractedEntity, ExtractedKnowledge, ExtractedRelation
from pipelines.session import SessionIngestPipeline


class TestExtractionSchema:
    def test_extracted_entity(self):
        e = ExtractedEntity(name="Hermes Agent", type="tool", label="Hermes Agent", confidence=0.95)
        assert e.name == "Hermes Agent"
        assert e.type == "tool"

    def test_extracted_relation(self):
        r = ExtractedRelation(source="Hermes", target="Neo4j", type="uses", weight=0.9)
        assert r.source == "Hermes"

    def test_extracted_knowledge_defaults(self):
        k = ExtractedKnowledge()
        assert k.entities == []
        assert k.relations == []
        assert k.decisions == []


class TestSessionIngestPipeline:
    @pytest.fixture
    def pipeline(self):
        return SessionIngestPipeline()

    def test_init(self, pipeline):
        assert pipeline.name == "session-ingest"
        assert pipeline.description

    def test_resolve_creates_session_resource(self, pipeline):
        """Test that resolve creates a session Resource."""
        mock_context = MagicMock()
        mock_llm = MagicMock()
        mock_llm.extract_structured.return_value = ExtractedKnowledge(
            summary="Test session summary",
            topics=["AI", "testing"],
            entities=[ExtractedEntity(name="Vik", type="person", label="Vik")],
        )
        mock_context.llm = mock_llm
        mock_context.config.llm.extraction_model = "test-model"

        record = {"id": "sess-1", "title": "Test Session", "started_at": "2024-01-01", "messages": ["user: hello"]}
        resources = pipeline.resolve(mock_context, record)

        assert len(resources) == 2  # session + 1 entity
        assert resources[0].type == "session"
        assert resources[0].id == "session:sess-1"
        assert resources[1].type == "person"
        assert resources[1].id == "entity:vik"

    def test_resolve_handles_llm_failure(self, pipeline):
        """Should create a fallback session resource if LLM fails."""
        mock_context = MagicMock()
        mock_llm = MagicMock()
        mock_llm.extract_structured.side_effect = Exception("API error")
        mock_context.llm = mock_llm
        mock_context.config.llm.extraction_model = "test-model"

        record = {"id": "sess-2", "title": "Fallback", "messages": ["user: hi"]}
        resources = pipeline.resolve(mock_context, record)

        assert len(resources) == 1  # just the session resource
        assert resources[0].type == "session"

    def test_get_relationships(self, pipeline):
        """Session should mention entities."""
        mock_context = MagicMock()
        session_resource = MagicMock(type="session", id="session:sess-1")
        entity_resource = MagicMock(type="person", id="entity:vik")
        entity_resource2 = MagicMock(type="tool", id="entity:python")

        rels = pipeline.get_relationships(
            mock_context,
            [],
            [session_resource, entity_resource, entity_resource2],
        )

        assert len(rels) == 2
        assert rels[0].source_id == "session:sess-1"
        assert rels[0].target_id == "entity:vik"
        assert rels[0].type == "mentions"

    def test_extract_with_empty_db(self, pipeline, tmp_path):
        """Should yield nothing if DB doesn't exist."""
        db = tmp_path / "nonexistent.db"
        pipeline.set_db_path(db)
        mock_context = MagicMock()
        mock_context.metadata = {}
        records = list(pipeline.extract(mock_context, None))
        assert records == []
```

### tests/test_query.py

```python
"""Tests for the query engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import QueryResult, Resource
from core.query import NLQueryResult, QueryEngine


@pytest.fixture
def engine():
    return QueryEngine(
        graph=MagicMock(),
        embedder=MagicMock(),
        llm=MagicMock(),
    )


class TestSemanticSearch:
    def test_semantic_returns_results(self, engine):
        engine._embedder.embed_query.return_value = [0.1] * 384
        engine._graph.vector_search.return_value = QueryResult(
            nodes=[Resource(id="r1", type="session", label="Test")],
            scores=[0.95],
        )
        result = engine.semantic("test query", top_k=5)
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "r1"

    def test_semantic_no_embedder(self, engine):
        engine._embedder = None
        with pytest.raises(RuntimeError, match="embedder"):
            engine.semantic("test")

    def test_semantic_with_type_filter(self, engine):
        engine._embedder.embed_query.return_value = [0.1] * 384
        engine._graph.vector_search.return_value = QueryResult()
        engine.semantic("test", type_filter="session")
        engine._graph.vector_search.assert_called_with(
            [0.1] * 384, top_k=10, type_filter="session"
        )


class TestTraverse:
    def test_traverse_delegates(self, engine):
        engine._graph.traverse.return_value = QueryResult()
        engine.traverse("node-1", hops=2, direction="outgoing")
        engine._graph.traverse.assert_called_with(
            "node-1", hops=2, rel_types=None, direction="outgoing"
        )


class TestHybridSearch:
    def test_hybrid(self, engine):
        engine._embedder.embed_query.return_value = [0.1] * 384
        engine._graph.hybrid_search.return_value = QueryResult()
        result = engine.hybrid("test", cypher_filter="node.type='session'", top_k=5)
        assert result is not None
        engine._graph.hybrid_search.assert_called_once()

    def test_hybrid_no_embedder(self, engine):
        engine._embedder = None
        with pytest.raises(RuntimeError):
            engine.hybrid("test")


class TestNLQuery:
    def test_nl_query_generates_and_executes(self, engine):
        engine._llm.generate.return_value = "MATCH (r:Resource) RETURN r LIMIT 5"
        engine._graph.run_cypher.return_value = [{"id": "r1", "label": "Test"}]

        result = engine.nl_query("Show me resources")
        assert result.cypher == "MATCH (r:Resource) RETURN r LIMIT 5"
        assert len(result.results) == 1
        assert result.error == ""

    def test_nl_query_handles_markdown_fences(self, engine):
        engine._llm.generate.return_value = "```cypher\nMATCH (r:Resource) RETURN r\n```"
        engine._graph.run_cypher.return_value = []

        result = engine.nl_query("Show resources")
        assert "```" not in result.cypher

    def test_nl_query_empty_result(self, engine):
        engine._llm.generate.return_value = ""
        result = engine.nl_query("test")
        assert "empty" in result.error

    def test_nl_query_no_llm(self, engine):
        engine._llm = None
        result = engine.nl_query("test")
        assert "LLM" in result.error

    def test_cypher_execution_error(self, engine):
        engine._llm.generate.return_value = "MATCH (r:Resource) RETURN r"
        engine._graph.run_cypher.side_effect = Exception("Syntax error")

        result = engine.nl_query("test")
        assert "Syntax error" in result.error

    def test_explain(self, engine):
        engine._llm.generate.return_value = "This query finds all sessions."
        result = engine.explain("MATCH (s:Resource {type:'session'}) RETURN s")
        assert "sessions" in result


class TestQueryEngineInit:
    def test_init_without_optional(self):
        engine = QueryEngine(graph=MagicMock())
        assert engine._embedder is None
        assert engine._llm is None
```

## Instructions for Cursor CLI

1. Create `core/extraction_schema.py` with `ExtractedEntity`, `ExtractedRelation`, `ExtractedKnowledge`
2. Replace `pipelines/session.py` with `SessionIngestPipeline`
3. Create `core/query.py` with `QueryEngine`
4. Replace `cli/query.py` with semantic/traverse/ask/explain commands
5. Create `tests/test_session_pipeline.py`
6. Create `tests/test_query.py`
7. Update `cli/main.py` to add `kg query` subcommand
8. Update `core/__init__.py` to export `QueryEngine`, `NLQueryResult`
9. Run `uv run python -m pytest tests/test_session_pipeline.py tests/test_query.py -v` and report results
10. Run `uv run python -c "from core.query import QueryEngine, NLQueryResult; from core.extraction_schema import ExtractedKnowledge; from pipelines.session import SessionIngestPipeline; print('All imports OK')"`