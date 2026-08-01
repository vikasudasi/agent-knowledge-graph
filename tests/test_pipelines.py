"""Tests for the pipeline framework."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.models import PipelineCheckpoint, Relationship, Resource
from pipelines.base import KnowledgePipeline, PipelineContext, PipelineRegistry, PipelineResult


class SimplePipeline(KnowledgePipeline):
    """Test pipeline that yields static records."""

    def extract(self, context, checkpoint=None):
        for i in range(3):
            yield {"id": f"rec-{i}", "content": f"Record {i}"}

    def resolve(self, context, record):
        return [
            Resource(
                id=record["id"],
                type="test",
                label=record["content"],
                properties={"index": int(record["id"].split("-")[1])},
            )
        ]

    def get_relationships(self, context, records, resources):
        if len(resources) >= 2:
            return [
                Relationship(
                    source_id=resources[0].id,
                    target_id=resources[1].id,
                    type="relates_to",
                    properties={"order": 1},
                )
            ]
        return []


class EmptyPipeline(KnowledgePipeline):
    """Pipeline that yields nothing."""

    def extract(self, context, checkpoint=None):
        return iter([])
        yield  # make it a generator

    def resolve(self, context, record):
        return []


class FailingPipeline(KnowledgePipeline):
    """Pipeline that fails during resolve."""

    def extract(self, context, checkpoint=None):
        yield {"id": "fail-1"}

    def resolve(self, context, record):
        raise ValueError("Intentional failure")


class TestPipelineContext:
    def test_create_context_wires_providers(self):
        from core.config import KGConfig

        cfg = KGConfig()
        cfg.llm.api_key = "test-key"
        cfg.embedding.provider = "local"

        with patch("core.embedding.LocalEmbeddingProvider"):
            with patch.object(cfg.llm, "api_key", "test"):
                with patch("pipelines.base.LLMProviderFactory.create") as mock_llm:
                    with patch("pipelines.base.EmbeddingProviderFactory.create") as mock_emb:
                        with patch("pipelines.base.Neo4jClient") as mock_graph:
                            mock_graph.return_value = MagicMock()
                            ctx = PipelineRegistry.create_context(cfg)
                            assert isinstance(ctx, PipelineContext)
                            assert ctx.config == cfg


class TestKnowledgePipeline:
    def test_run_extract_resolve_embed_write(self):
        pipeline = SimplePipeline(name="test-simple", description="Simple test pipeline")
        mock_graph = MagicMock()
        mock_graph.get_checkpoint.return_value = None
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
        )

        result = pipeline.run(ctx)
        assert isinstance(result, PipelineResult)
        assert result.pipeline_name == "test-simple"
        assert result.records_processed == 3
        assert result.resources_created == 3
        assert result.relationships_created == 1

    def test_run_empty_pipeline(self):
        pipeline = EmptyPipeline(name="empty")
        mock_graph = MagicMock()
        mock_graph.get_checkpoint.return_value = None

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=MagicMock(),
            graph=mock_graph,
        )

        result = pipeline.run(ctx)
        assert result.records_processed == 0
        assert result.resources_created == 0

    def test_run_with_checkpoint_resume(self):
        pipeline = SimplePipeline(name="test-checkpoint")
        mock_graph = MagicMock()
        mock_graph.get_checkpoint.return_value = PipelineCheckpoint(
            pipeline_name="test-checkpoint",
            last_processed_id="rec-1",
            total_processed=2,
        )
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
        )

        result = pipeline.run(ctx)
        assert result.records_processed == 3

    def test_run_full_rebuild_ignores_checkpoint(self):
        pipeline = SimplePipeline(name="test-rebuild")
        mock_graph = MagicMock()
        mock_graph.get_checkpoint.return_value = PipelineCheckpoint(
            pipeline_name="test-rebuild", last_processed_id="rec-5", total_processed=10
        )
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
            full_rebuild=True,
        )

        result = pipeline.run(ctx)
        assert result.records_processed == 3  # Not 10 + 3, full rebuild

    def test_run_dry_mode(self):
        pipeline = SimplePipeline(name="test-dry")
        mock_graph = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
            dry_run=True,
        )

        result = pipeline.run(ctx)
        assert result.resources_created == 3
        mock_graph.upsert_resource.assert_not_called()

    def test_error_isolation(self):
        pipeline = FailingPipeline(name="test-fail")
        mock_graph = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
        )

        result = pipeline.run(ctx)
        assert result.errors == 1
        assert result.records_processed == 1
        assert result.resources_created == 0

    def test_progress_callback(self):
        pipeline = SimplePipeline(name="test-progress")
        mock_graph = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        calls = []

        def progress(current, total, phase):
            calls.append((current, total, phase))

        ctx = PipelineContext(
            config=MagicMock(),
            llm=MagicMock(),
            embedder=mock_embedder,
            graph=mock_graph,
        )

        pipeline.run(ctx, progress_callback=progress)
        assert len(calls) > 0


class TestPipelineRegistry:
    def test_register_and_get(self):
        pipeline = SimplePipeline(name="reg-test")
        PipelineRegistry.register(pipeline)
        assert PipelineRegistry.get("reg-test") is pipeline

    def test_list_pipelines(self):
        PipelineRegistry._pipelines.clear()
        PipelineRegistry.register(SimplePipeline(name="p1"))
        PipelineRegistry.register(SimplePipeline(name="p2"))
        listed = PipelineRegistry.list_pipelines()
        assert len(listed) == 2
        names = [p["name"] for p in listed]
        assert "p1" in names
        assert "p2" in names

    def test_get_nonexistent(self):
        assert PipelineRegistry.get("nope") is None
