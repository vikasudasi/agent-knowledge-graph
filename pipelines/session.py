"""Session-ingest pipeline — Hermes session DB -> knowledge graph."""

from __future__ import annotations

import json
import logging
import re
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

ENTITY QUALITY RULES — extract ONLY durable, reusable concepts. Skip transient noise:
- EXTRACT: people, projects, tools, libraries, frameworks, architectural patterns,
  concrete files being created/modified, named skills, technical concepts, decisions.
- SKIP:
  • CSS selectors / HTML class names / UI styling (.btn, .container, flex-row)
  • Linter error codes (E402, RUF015, TRY004) and HTTP status codes (200, 404)
  • CLI flags (--verbose, --force) and config key=value pairs
  • Raw Python tracebacks, error messages, or exception strings
  • Numbers, timestamps, hex values in isolation
  • File paths mentioned only in passing (extract only if the file IS the subject)
  • Generic filler words ("bug", "fix", "test", "issue", "thing")

LABEL QUALITY RULES:
- Labels MUST be unique and specific. Never use "Topic", "Company", "Bug", "Feature",
  or "Concept" as a label — use the actual name: "Neo4j" not "Database",
  "CypherSyntaxError" not "Bug", "DeepSeek" not "AI Model".
- Each label should self-identify the entity without needing the type field.
- Be thorough but accurate. Only extract what is clearly present in the text."""


# ---------------------------------------------------------------------------
# Entity noise filter — catches the 4 categories of junk seen in production:
#   CSS selectors (.btn-primary), lint codes (E402, RUF015), CLI flags (--verbose),
#   escape-artifact strings ('nonetype'-object-has-no-attribute)
# ---------------------------------------------------------------------------
_RE_CSS_SELECTOR = re.compile(
    r"^(\.|#)[\w-]+(?:[.:#][\w-]+)*$"  # .class, #id, element.class, .class::pseudo
)
_RE_LINT_CODE = re.compile(r"^[A-Z]{1,6}\d{2,4}$")  # E402, RUF015, TRY004
_RE_CLI_FLAG = re.compile(r"^--?[\w-]+$")  # --verbose, -f, --no-cache
_RE_ERROR_CODE = re.compile(r"^\d{3}[\s-]")  # 402 error, 502-bad-gateway
_RE_ESCAPE_ARTIFACT = re.compile(r"'.+?'-")  # 'nonetype'-object, 'total'-is-undefined
_RE_PURE_DIGITS = re.compile(r"^\d+$")  # 384, 500, 7000
_RE_SHORT_GENERIC = re.compile(r"^(bug|fix|test|issue|thing|stuff|item|todo|misc)$", re.I)
_RE_GENERIC_LABEL = re.compile(r"^(Topic|Company|Bug|Feature|Concept|Tool|File|Project|Task|Skill|Artifact|Person|Error|Exception|Module|Class|Function|Command|Config|Model|Issue|Decision|Process|Event|Format|Type|Category|Tag)$", re.I)


def _is_valid_entity(name: str, ent_type: str, label: str) -> tuple[bool, str]:
    """Return (valid, reason) for an extracted entity. Rejects obvious noise."""
    name_stripped = name.strip()

    # Too short or empty
    if len(name_stripped) < 2:
        return False, "too short"

    # Pure numbers
    if _RE_PURE_DIGITS.match(name_stripped):
        return False, "pure digits"

    # Escape-artifact strings from mangled tracebacks
    if _RE_ESCAPE_ARTIFACT.search(name_stripped):
        return False, "escape artifact"

    # Short generic filler words
    if _RE_SHORT_GENERIC.match(name_stripped):
        return False, "generic filler"

    # Concept-type specific checks (most noise is typed as "concept")
    if ent_type == "concept":
        if _RE_CSS_SELECTOR.match(name_stripped):
            return False, "CSS selector"
        if _RE_LINT_CODE.match(name_stripped):
            return False, "lint code"
        if _RE_CLI_FLAG.match(name_stripped):
            return False, "CLI flag"
        if _RE_ERROR_CODE.match(name_stripped):
            return False, "error code"

    # Generic labels that indicate the LLM gave no real name
    if _RE_GENERIC_LABEL.match(label.strip()):
        return False, "generic label"

    return True, ""


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
        skipped = 0
        for ent in entities:
            ent_data = ent if isinstance(ent, dict) else json.loads(ent.model_dump_json())
            name = ent_data.get("name", "unknown")
            ent_type = ent_data.get("type", "concept")
            ent_label = ent_data.get("label", name)

            valid, reason = _is_valid_entity(name, ent_type, ent_label)
            if not valid:
                skipped += 1
                logger.debug("Filtered entity '%s' (%s): %s", name, ent_type, reason)
                continue

            ent_id = name.lower().replace(" ", "-").replace("/", "-")
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

        if skipped:
            logger.info(
                "Filtered %d/%d noise entities from session %s",
                skipped,
                len(entities),
                session_id,
            )

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
