"""Tests for the embedding layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.embedding import (
    EmbeddingDimensionMismatchError,
    EmbeddingProvider,
    EmbeddingProviderFactory,
    LocalEmbeddingProvider,
    RemoteEmbeddingProvider,
)


class TestEmbeddingProviderFactory:
    def test_create_local(self) -> None:
        from core.config import KGConfig

        cfg = KGConfig()
        cfg.embedding.provider = "local"
        provider = EmbeddingProviderFactory.create(cfg)
        assert isinstance(provider, LocalEmbeddingProvider)

    def test_create_openai(self) -> None:
        from core.config import KGConfig

        cfg = KGConfig()
        cfg.embedding.provider = "openai"
        cfg.embedding.api_key = "test-key"
        with patch("httpx.Client"):
            provider = EmbeddingProviderFactory.create(cfg)
            assert isinstance(provider, RemoteEmbeddingProvider)

    def test_create_unknown(self) -> None:
        from core.config import KGConfig

        cfg = KGConfig()
        cfg.embedding.provider = "nonexistent"
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            EmbeddingProviderFactory.create(cfg)

    def test_register_custom(self) -> None:
        class FakeProvider(EmbeddingProvider):
            def embed(self, text: str) -> list[float]:
                return [0.0] * self.config.dimension

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * self.config.dimension for _ in texts]

        from core.config import KGConfig

        EmbeddingProviderFactory.register("fake", FakeProvider)
        cfg = KGConfig()
        cfg.embedding.provider = "fake"
        provider = EmbeddingProviderFactory.create(cfg)
        assert isinstance(provider, FakeProvider)


class TestLocalEmbeddingProvider:
    @pytest.fixture
    def config(self):  # noqa: ANN201
        from core.config import EmbeddingConfig

        return EmbeddingConfig(provider="local", local_model="all-MiniLM-L6-v2", dimension=384)

    def test_init(self, config) -> None:  # noqa: ANN001
        provider = LocalEmbeddingProvider(config)
        assert provider.config == config

    def test_embed_returns_correct_dimension(self, config) -> None:  # noqa: ANN001
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1] * 384

        provider = LocalEmbeddingProvider(config)
        provider._model = mock_model

        result = provider.embed("hello world")
        assert len(result) == 384
        mock_model.encode.assert_called_once_with("hello world", normalize_embeddings=True)

    def test_embed_batch_returns_correct_dimensions(self, config) -> None:  # noqa: ANN001
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            MagicMock(tolist=lambda: [0.1] * 384),
            MagicMock(tolist=lambda: [0.2] * 384),
        ]

        provider = LocalEmbeddingProvider(config)
        provider._model = mock_model

        results = provider.embed_batch(["hello", "world"])
        assert len(results) == 2
        assert len(results[0]) == 384
        assert len(results[1]) == 384

    def test_embed_validates_dimension(self, config) -> None:  # noqa: ANN001
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1] * 512

        provider = LocalEmbeddingProvider(config)
        provider._model = mock_model

        with pytest.raises(EmbeddingDimensionMismatchError):
            provider.embed("test")

    def test_uses_lru_cache(self, config) -> None:  # noqa: ANN001
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1] * 384

        provider = LocalEmbeddingProvider(config)
        provider._model = mock_model

        provider.embed("hello")
        provider.embed("hello")
        assert mock_model.encode.call_count == 1

    def test_batch_uses_config_batch_size(self, config) -> None:  # noqa: ANN001
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            MagicMock(tolist=lambda: [0.1] * 384),
            MagicMock(tolist=lambda: [0.2] * 384),
        ]

        provider = LocalEmbeddingProvider(config)
        provider._model = mock_model

        provider.embed_batch(["a", "b"])
        mock_model.encode.assert_called_once_with(
            ["a", "b"], batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )

    def test_embed_query_same_as_embed(self, config) -> None:  # noqa: ANN001
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1] * 384

        provider = LocalEmbeddingProvider(config)
        provider._model = mock_model

        result = provider.embed_query("test query")
        assert len(result) == 384

    def test_auto_download_on_first_call(self, config) -> None:  # noqa: ANN001
        with patch("core.embedding.LocalEmbeddingProvider._detect_device", return_value="cpu"):
            with patch("core.embedding.LocalEmbeddingProvider._load_model") as mock_load:
                mock_model = MagicMock()
                mock_model.encode.return_value = MagicMock()
                mock_model.encode.return_value.tolist.return_value = [0.1] * 384
                mock_load.return_value = mock_model

                provider = LocalEmbeddingProvider(config)
                assert provider._model is None

                provider.embed("lazy load test")
                mock_load.assert_called_once()


class TestRemoteEmbeddingProvider:
    @pytest.fixture
    def config(self):  # noqa: ANN201
        from core.config import EmbeddingConfig

        return EmbeddingConfig(provider="openai", api_key="test-key", model="text-embedding-3-small", dimension=1536)

    def test_embed(self, config) -> None:  # noqa: ANN001
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"index": 0, "embedding": [0.1] * 1536}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            }
            mock_client.post.return_value = mock_response

            provider = RemoteEmbeddingProvider(config)
            result = provider.embed("hello")
            assert len(result) == 1536

    def test_embed_batch_maintains_order(self, config) -> None:  # noqa: ANN001
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"index": 1, "embedding": [0.2] * 1536},
                    {"index": 0, "embedding": [0.1] * 1536},
                ],
                "model": "text-embedding-3-small",
            }
            mock_client.post.return_value = mock_response

            provider = RemoteEmbeddingProvider(config)
            results = provider.embed_batch(["first", "second"])
            assert len(results) == 2
            assert results[0][0] == 0.1

    def test_dimension_mismatch(self, config) -> None:  # noqa: ANN001
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"index": 0, "embedding": [0.1] * 512}],
            }
            mock_client.post.return_value = mock_response

            provider = RemoteEmbeddingProvider(config)
            with pytest.raises(EmbeddingDimensionMismatchError):
                provider.embed("test")
