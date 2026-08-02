"""Session-ingest pipeline — Hermes session DB -> knowledge graph."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.extraction_schema import ExtractedKnowledge
from core.models import PipelineCheckpoint, Relationship, Resource
from pipelines.base import KnowledgePipeline, PipelineContext, PipelineRegistry

logger = logging.getLogger(__name__)


HERMES_DB_PATH = Path.home() / ".hermes" / "state.db"

EXTRACTION_PROMPT = """\
You are an AI knowledge graph extraction assistant.
Analyze this agent conversation session and extract structured knowledge.

SESSION:
Title: {title}
Started: {started_at}
Messages:
{messages}

Extract the following in JSON format with EXACTLY this structure:

{{
  "summary": "One paragraph summarizing what happened",
  "topics": ["topic1", "topic2", "topic3"],
  "entities": [
    {{"name": "EntityName", "type": "person|project|tool|concept|file|task|skill|artifact", "label": "Short human-readable label", "context": "Why this entity is relevant"}}
  ],
  "relations": [
    {{"source": "EntityA", "target": "EntityB", "type": "mentions|produces|uses|decides|references|blocks|resolves|assigns", "context": "Evidence from conversation"}}
  ],
  "decisions": ["Decision or conclusion"],
  "tools_used": ["tool/command"],
  "outcome": "completed|in_progress|failed|unknown"
}}

CRITICAL RULES:
- "entities" MUST be a JSON array of objects. Each object MUST have "name", "type", and "label" fields.
- "relations" MUST be a JSON array of objects. Each object MUST have "source", "target", and "type" fields.
- Do NOT use strings for entities or relations — use the object format shown above.
- Be thorough but accurate. Only extract what is clearly present in the text."""


class SessionIngestPipeline(KnowledgePipeline[dict[str, Any]]):
    """Reads Hermes session DB, extracts knowledge via LLM, writes to Neo4j."""

    def __init__(self) -> None:
        super().__init__(
            name="session-ingest",
            description="Extract entities, relations, and topics from Hermes agent sessions",
            version="1.0",
        )
        self._db_path: Path | None = None

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
        msg_cursor = conn.cursor()

        try:
            tables = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sessions', 'messages')"
            ).fetchall()
            table_names = {row["name"] for row in tables}

            if "sessions" not in table_names:
                logger.warning(f"No 'sessions' table in {db_path}")
                return

            query = "SELECT id, title, started_at FROM sessions"
            params: dict[str, Any] = {}
            if checkpoint and checkpoint.last_processed_id:
                query += " WHERE started_at > :checkpoint_ts"
                params["checkpoint_ts"] = float(checkpoint.last_processed_id)
            query += " ORDER BY started_at ASC"

            from datetime import datetime, timezone

            session_rows = cursor.execute(query, params).fetchall()
            processed = 0
            for row in session_rows:
                if context.max_records is not None and processed >= context.max_records:
                    break
                processed += 1
                session = dict(row)
                messages: list[str] = []
                # Convert Unix timestamp to ISO date
                ts = session.get("started_at")
                if isinstance(ts, (int, float)):
                    session["started_at"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

                if "messages" in table_names:
                    msg_rows = msg_cursor.execute(
                        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                        (session["id"],),
                    ).fetchall()
                    for msg in msg_rows:
                        role = msg["role"] or "user"
                        content = (msg["content"] or "")[:500]
                        messages.append(f"{role}: {content}")

                session["messages"] = messages
                yield session
        finally:
            conn.close()

    def resolve(self, context: PipelineContext, record: dict[str, Any]) -> list[Resource]:
        """Extract knowledge from session via LLM, then convert to Resource nodes."""
        title = record.get("title", "Untitled Session")
        started_at = record.get("started_at", "unknown")
        messages_text = "\n".join(record.get("messages", ["(no messages)"]))
        session_id = str(record["id"])

        prompt = EXTRACTION_PROMPT.format(
            title=title,
            started_at=started_at,
            messages=messages_text,
        )

        try:
            extracted = context.llm.extract_structured(
                messages=[{"role": "user", "content": prompt}],
                schema=ExtractedKnowledge,
                system_prompt="You are a knowledge graph extraction assistant. Output ONLY valid JSON.",
                model=context.config.llm.extraction_model,
            )
            extracted_data = extracted.model_dump() if hasattr(extracted, "model_dump") else {}
        except Exception as exc:
            logger.warning(f"LLM extraction failed for session {session_id}: {exc}")
            extracted_data = {
                "session_id": session_id,
                "summary": title,
                "entities": [],
                "relations": [],
            }

        resources: list[Resource] = []
        ingested_at = datetime.now(UTC)

        session_resource = Resource(
            id=f"session:{session_id}",
            type="session",
            label=(title or "Untitled Session")[:200],
            properties={
                "session_id": session_id,
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

        entities = extracted_data.get("entities", [])
        for ent in entities:
            ent_data = ent if isinstance(ent, dict) else json.loads(ent.model_dump_json())
            name = ent_data.get("name", "unknown")
            ent_id = name.lower().replace(" ", "-").replace("/", "-")
            ent_type = ent_data.get("type", "concept")
            ent_label = ent_data.get("label", name)
            resource = Resource(
                id=f"entity:{ent_id}",
                type=ent_type,
                label=ent_label[:200],
                properties={
                    "canonical_name": name,
                    "aliases": ent_data.get("aliases", []),
                    "confidence": ent_data.get("confidence", 0.8),
                    "context": ent_data.get("context", ""),
                },
                ingested_at=ingested_at,
            )
            resources.append(resource)

        return resources

    def get_relationships(
        self,
        context: PipelineContext,
        records: list[dict[str, Any]],
        resources: list[Resource],
    ) -> list[Relationship]:
        """Generate session->entity mention links for resolved resources."""
        _ = context
        _ = records
        session_resources = [r for r in resources if r.type == "session"]
        entity_resources = [r for r in resources if r.type != "session"]
        if not session_resources:
            return []

        session_id = session_resources[0].id
        relationships: list[Relationship] = []
        for entity in entity_resources:
            relationships.append(
                Relationship(
                    source_id=session_id,
                    target_id=entity.id,
                    type="mentions",
                    properties={"weight": 1.0},
                )
            )
        return relationships

    def _resolve_db_path(self, context: PipelineContext) -> Path:
        if self._db_path:
            return self._db_path
        db_path = Path(context.metadata.get("hermes_db_path", str(HERMES_DB_PATH)))
        self._db_path = db_path
        return db_path

    def set_db_path(self, path: str | Path) -> None:
        """Override Hermes DB path (for tests)."""
        self._db_path = Path(path)


PipelineRegistry.register(SessionIngestPipeline())
