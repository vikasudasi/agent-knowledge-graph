"""Tests for the session-ingest pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.extraction_schema import ExtractedEntity, ExtractedKnowledge, ExtractedRelation
from pipelines.session import SessionIngestPipeline


class TestExtractionSchema:
    def test_extracted_entity(self) -> None:
        entity = ExtractedEntity(
            name="Hermes Agent",
            type="tool",
            label="Hermes Agent",
            confidence=0.95,
        )
        assert entity.name == "Hermes Agent"
        assert entity.type == "tool"

    def test_extracted_relation(self) -> None:
        relation = ExtractedRelation(source="Hermes", target="Neo4j", type="uses", weight=0.9)
        assert relation.source == "Hermes"

    def test_extracted_knowledge_defaults(self) -> None:
        knowledge = ExtractedKnowledge()
        assert knowledge.entities == []
        assert knowledge.relations == []
        assert knowledge.decisions == []


class TestSessionIngestPipeline:
    @pytest.fixture
    def pipeline(self) -> SessionIngestPipeline:
        return SessionIngestPipeline()

    def test_init(self, pipeline: SessionIngestPipeline) -> None:
        assert pipeline.name == "session-ingest"
        assert pipeline.description

    def test_resolve_creates_session_resource(self, pipeline: SessionIngestPipeline) -> None:
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

        record = {
            "id": "sess-1",
            "title": "Test Session",
            "started_at": "2024-01-01",
            "messages": ["user: hello"],
        }
        resources = pipeline.resolve(mock_context, record)

        assert len(resources) == 2  # session + 1 entity
        assert resources[0].type == "session"
        assert resources[0].id == "session:sess-1"
        assert resources[1].type == "person"
        assert resources[1].id == "entity:vik"

    def test_resolve_handles_llm_failure(self, pipeline: SessionIngestPipeline) -> None:
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

    def test_get_relationships(self, pipeline: SessionIngestPipeline) -> None:
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

    def test_extract_with_empty_db(self, pipeline: SessionIngestPipeline, tmp_path) -> None:
        """Should yield nothing if DB doesn't exist."""
        db = tmp_path / "nonexistent.db"
        pipeline.set_db_path(db)
        mock_context = MagicMock()
        mock_context.metadata = {}
        records = list(pipeline.extract(mock_context, None))
        assert records == []
