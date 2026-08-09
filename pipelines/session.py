"""Session-ingest pipeline — Hermes session DB -> knowledge graph."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
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
    {{"name": "EntityName", "type": "person|project|tool|concept|file|task|skill|artifact",
      "label": "Short human-readable label", "context": "Why this entity is relevant"}}
  ],
  "relations": [
    {{"source": "EntityA", "target": "EntityB",
      "type": "mentions|produces|uses|decides|references|blocks|resolves|assigns",
      "context": "Evidence from conversation"}}
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

            # Query sessions that have messages newer than checkpoint
            has_messages_table = "messages" in table_names
            checkpoint_ts: float | None = None
            if checkpoint and checkpoint.last_processed_id:
                checkpoint_ts = float(checkpoint.last_processed_id)

            if has_messages_table:
                query = "SELECT DISTINCT s.id, s.title, s.started_at FROM sessions s"
                params: dict[str, Any] = {}
                if checkpoint_ts is not None:
                    query += " WHERE EXISTS (SELECT 1 FROM messages m"
                    query += " WHERE m.session_id = s.id"
                    query += " AND m.timestamp > :checkpoint_ts)"
                    params["checkpoint_ts"] = checkpoint_ts
                else:
                    query += " WHERE EXISTS ("
                    query += "SELECT 1 FROM messages m WHERE m.session_id = s.id"
                    query += ")"
                query += " ORDER BY s.started_at ASC"
            else:
                query = "SELECT id, title, started_at FROM sessions"
                params = {}
                if checkpoint_ts is not None:
                    query += " WHERE started_at > :checkpoint_ts"
                query += " ORDER BY started_at ASC"

            from datetime import datetime, timezone

            session_rows = cursor.execute(query, params).fetchall()
            processed = 0
            global_max_ts: float = 0.0
            for row in session_rows:
                if context.max_records is not None and processed >= context.max_records:
                    break
                processed += 1
                session = dict(row)
                messages: list[str] = []
                # Convert Unix timestamp to ISO date
                ts = session.get("started_at")
                if isinstance(ts, (int, float)):
                    session["started_at"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()  # noqa: UP017

                if has_messages_table:
                    msg_rows = msg_cursor.execute(
                        "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                        (session["id"],),
                    ).fetchall()
                    for msg in msg_rows:
                        role = msg["role"] or "user"
                        content = (msg["content"] or "")[:500]
                        messages.append(f"{role}: {content}")
                        msg_ts = msg["timestamp"]
                        if isinstance(msg_ts, (int, float)) and msg_ts > global_max_ts:
                            global_max_ts = msg_ts

                session["messages"] = messages
                # Attach checkpoint hint for base.py
                if global_max_ts > 0:
                    session["_checkpoint_ts"] = global_max_ts
                yield session
        finally:
            conn.close()

    def _extract_chunk(
        self,
        context: PipelineContext,
        session_id: str,
        title: str,
        started_at: str,
        messages: list[str],
    ) -> ExtractedKnowledge | None:
        """Run LLM extraction for a single chunk of messages. Returns None on failure."""
        prompt = EXTRACTION_PROMPT.format(
            title=title,
            started_at=started_at,
            messages="\n".join(messages),
        )
        try:
            extracted = context.llm.extract_structured(
                messages=[{"role": "user", "content": prompt}],
                schema=ExtractedKnowledge,
                system_prompt="You are a knowledge graph extraction assistant. Output ONLY valid JSON.",
                model=context.config.llm.extraction_model,
            )
            return extracted if isinstance(extracted, ExtractedKnowledge) else None
        except Exception as exc:
            logger.warning(f"LLM extraction failed for session {session_id}: {exc}")
            return None

    @staticmethod
    def _merge_extractions(results: list[ExtractedKnowledge]) -> ExtractedKnowledge:
        """Merge multiple chunk extraction results, deduplicating entities, relations, and lists."""
        merged = ExtractedKnowledge(session_id=results[0].session_id if results else "")
        entity_map: dict[str, Any] = {}
        relation_keys: set[tuple[str, str, str]] = set()
        topics: list[str] = []
        decisions: list[str] = []
        tools: list[str] = []
        summaries: list[str] = []
        outcomes: list[str] = []

        for r in results:
            if r is None:
                continue
            if r.summary:
                summaries.append(r.summary)
            for t in r.topics:
                if t not in topics:
                    topics.append(t)
            for d in r.decisions:
                if d not in decisions:
                    decisions.append(d)
            for tool in r.tools_used:
                if tool not in tools:
                    tools.append(tool)
            if r.outcome:
                outcomes.append(r.outcome)
            for ent in r.entities:
                key = (ent.name or "").lower()
                if key not in entity_map:
                    entity_map[key] = ent
            for rel in r.relations:
                rel_key = (rel.source, rel.target, rel.type)
                if rel_key not in relation_keys:
                    relation_keys.add(rel_key)
                    merged.relations.append(rel)
            if not merged.session_id:
                merged.session_id = r.session_id

        merged.entities = list(entity_map.values())
        merged.topics = topics
        merged.decisions = decisions
        merged.tools_used = tools
        merged.summary = " ".join(s for s in summaries if s)[:500] if summaries else ""
        merged.outcome = outcomes[0] if outcomes else "in_progress"
        return merged

    def resolve(self, context: PipelineContext, record: dict[str, Any]) -> list[Resource]:
        """Extract knowledge from session via LLM (batched + concurrent), then convert to Resource nodes."""
        title = record.get("title", "Untitled Session")
        started_at = record.get("started_at", "unknown")
        session_id = str(record["id"])
        messages = record.get("messages", []) or ["(no messages)"]

        batch_size = 20
        max_workers = 4
        try:
            batch_size = int(context.config.pipelines.session_ingest_batch_size)
        except Exception:
            pass
        try:
            max_workers = int(context.config.pipelines.session_ingest_max_workers)
        except Exception:
            pass

        # Split the session's messages into chunks of `batch_size`.
        chunks = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]

        if len(chunks) == 1:
            results: list[ExtractedKnowledge | None] = [
                self._extract_chunk(context, session_id, title, started_at, chunks[0])
            ]
        else:
            with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
                results = list(
                    executor.map(
                        lambda c: self._extract_chunk(context, session_id, title, started_at, c),
                        chunks,
                    )
                )

        successful = [r for r in results if r is not None]
        if successful:
            extracted_data = self._merge_extractions(successful).model_dump()
        else:
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
                label=str(ent_label)[:200],
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
