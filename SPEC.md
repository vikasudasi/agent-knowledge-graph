# agent-knowledge-graph — Embedding Layer (Task 5)

## What to Build

Implement the embedding provider abstraction that pipelines use to generate vector embeddings. Default: local `sentence-transformers` with optional remote (OpenRouter/OpenAI) provider swap.

## Files to Create/Modify

- `core/embedding.py` — replace with full implementation
- `tests/test_embedding.py` — create comprehensive test suite
- `core/__init__.py` — update exports

## Requirements

1. **`EmbeddingProvider`** — abstract base: `embed(text)`, `embed_batch(texts)`, `embed_query(text)` (optimized for asymmetric search)
2. **`LocalEmbeddingProvider`** — sentence-transformers, default `all-MiniLM-L6-v2` (384d), auto-download, GPU auto-detection, batch processing
3. **`RemoteEmbeddingProvider`** — OpenAI-compatible embedding API (OpenRouter/OpenAI)
4. **`EmbeddingProviderFactory`** — creates from `config.embedding.provider`
5. **Dimension check** — embedding vector must match `config.embedding.dimension`
6. **LRU caching** — `functools.lru_cache` on `embed()` for dedup
7. **Thread-safe** — provider instances are independent per-thread

## Implementation

### core/embedding.py — replacement

```python
"""Embedding providers — local (sentence-transformers) and remote (OpenRouter/OpenAI)."""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from typing import Any

from core.config import EmbeddingConfig, KGConfig

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base embedding error."""

class EmbeddingDimensionMismatchError(EmbeddingError):
    """Embedding vector dimension doesn't match config."""

class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text into a fixed-size float vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a query text (may use different pooling for asymmetric models)."""
        return self.embed(text)

    def _validate_dimension(self, vector: list[float]) -> None:
        """Check vector dimension matches config."""
        expected = self._config.dimension
        if len(vector) != expected:
            raise EmbeddingDimensionMismatchError(
                f"Expected dimension {expected}, got {len(vector)}. "
                f"Check config.embedding.dimension or the model being used."
            )


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers local inference — zero API key needed."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._model = None
        self._device: str | None = None

    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA detected, using GPU for embeddings")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info(f"MPS detected (Apple Silicon), using GPU for embeddings")
        else:
            device = "cpu"
            logger.info(f"No GPU detected, using CPU for embeddings")
        return device

    def _load_model(self) -> Any:
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self._device = self._detect_device()
        logger.info(f"Loading embedding model: {self._config.local_model}")
        self._model = SentenceTransformer(self._config.local_model, device=self._device)
        return self._model

    @functools.lru_cache(maxsize=4096)
    def embed(self, text: str) -> list[float]:
        model = self._load_model()
        vector = model.encode(text, normalize_embeddings=True).tolist()
        self._validate_dimension(vector)
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        batch_size = self._config.batch_size
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        results = [v.tolist() for v in vectors]
        for v in results:
            self._validate_dimension(v)
        return results

    def embed_query(self, text: str) -> list[float]:
        """For symmetric models (all-MiniLM-L6-v2), same as embed()."""
        return self.embed(text)


class RemoteEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding API (OpenRouter, OpenAI, etc.)."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        import httpx
        # Determine base URL and API key
        self._api_base = "https://openrouter.ai/api/v1"
        self._api_key = config.api_key

        # If config has a remote model specified, use it
        self._model_name = config.model or "text-embedding-3-small"

        self._client = httpx.Client(
            base_url=self._api_base,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(
            "embeddings",
            json={
                "model": self._model_name,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()
        # Sort by index to preserve order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        results = [item["embedding"] for item in sorted_data]
        for v in results:
            self._validate_dimension(v)
        return results

    def embed_query(self, text: str) -> list[float]:
        """For OpenAI text-embedding-3-small, use the short-text model."""
        return self.embed(text)


class EmbeddingProviderFactory:
    """Creates the right embedding provider from config."""

    _providers: dict[str, type[EmbeddingProvider]] = {
        "local": LocalEmbeddingProvider,
        "openai": RemoteEmbeddingProvider,
        "openrouter": RemoteEmbeddingProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: type[EmbeddingProvider]) -> None:
        cls._providers[name] = provider_cls

    @classmethod
    def create(cls, config: KGConfig) -> EmbeddingProvider:
        provider_name = config.embedding.provider
        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown embedding provider: {provider_name}. "
                f"Available: {', '.join(cls._providers)}"
            )
        return cls._providers[provider_name](config.embedding)
```

### core/__init__.py update

Add to exports:
```python
from core.embedding import (
    EmbeddingProvider,
    EmbeddingProviderFactory,
    LocalEmbeddingProvider,
    RemoteEmbeddingProvider,
)
```

### tests/test_embedding.py

```python
"""Tests for the embedding layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.embedding import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingProviderFactory,
    LocalEmbeddingProvider,
    RemoteEmbeddingProvider,
)


class TestEmbeddingProviderFactory:
    def test_create_local(self):
        from core.config import KGConfig
        cfg = KGConfig()
        cfg.embedding.provider = "local"
        provider = EmbeddingProviderFactory.create(cfg)
        assert isinstance(provider, LocalEmbeddingProvider)

    def test_create_openai(self):
        from core.config import KGConfig
        cfg = KGConfig()
        cfg.embedding.provider = "openai"
        cfg.embedding.api_key = "test-key"
        with patch("httpx.Client"):
            provider = EmbeddingProviderFactory.create(cfg)
            assert isinstance(provider, RemoteEmbeddingProvider)

    def test_create_unknown(self):
        from core.config import KGConfig
        cfg = KGConfig()
        cfg.embedding.provider = "nonexistent"
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            EmbeddingProviderFactory.create(cfg)

    def test_register_custom(self):
        class FakeProvider(EmbeddingProvider):
            def embed(self, text):
                return [0.0] * self._config.dimension
            def embed_batch(self, texts):
                return [[0.0] * self._config.dimension for _ in texts]

        from core.config import EmbeddingConfig
        EmbeddingProviderFactory.register("fake", FakeProvider)

        from core.config import KGConfig
        cfg = KGConfig()
        cfg.embedding.provider = "fake"
        provider = EmbeddingProviderFactory.create(cfg)
        assert isinstance(provider, FakeProvider)


class TestLocalEmbeddingProvider:
    @pytest.fixture
    def config(self):
        from core.config import EmbeddingConfig
        return EmbeddingConfig(provider="local", local_model="all-MiniLM-L6-v2", dimension=384)

    def test_init(self, config):
        provider = LocalEmbeddingProvider(config)
        assert provider.config == config

    def test_embed_returns_correct_dimension(self, config):
        """Integration-light: mock SentenceTransformer.encode to return a 384-d vector."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1] * 384

        provider = LocalEmbeddingProvider(config)
        provider._model = mock_model

        result = provider.embed("hello world")
        assert len(result) == 384
        mock_model.encode.assert_called_once_with("hello world", normalize_embeddings=True)

    def test_embed_batch_returns_correct_dimensions(self, config):
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

    def test_embed_validates_dimension(self, config):
        """Should raise if dimension differs from config."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1] * 512  # wrong dim

        provider = LocalEmbeddingProvider(config)
        provider._model = mock_model

        with pytest.raises(EmbeddingDimensionMismatchError):
            provider.embed("test")

    def test_uses_lru_cache(self, config):
        """Same text should hit cache and not re-encode."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1] * 384

        provider = LocalEmbeddingProvider(config)
        provider._model = mock_model

        provider.embed("hello")
        provider.embed("hello")
        assert mock_model.encode.call_count == 1

    def test_batch_uses_config_batch_size(self, config):
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

    def test_embed_query_same_as_embed(self, config):
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1] * 384

        provider = LocalEmbeddingProvider(config)
        provider._model = mock_model

        result = provider.embed_query("test query")
        assert len(result) == 384

    def test_auto_download_on_first_call(self, config):
        """Lazy-load: model shouldn't be loaded until first embed call."""
        with patch("core.embedding.LocalEmbeddingProvider._detect_device", return_value="cpu"):
            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                mock_instance = MagicMock()
                mock_instance.encode.return_value = MagicMock()
                mock_instance.encode.return_value.tolist.return_value = [0.1] * 384
                mock_st.return_value = mock_instance

                provider = LocalEmbeddingProvider(config)
                assert provider._model is None  # Not loaded yet

                provider.embed("lazy load test")
                mock_st.assert_called_once_with("all-MiniLM-L6-v2", device="cpu")
                assert provider._model is not None


class TestRemoteEmbeddingProvider:
    @pytest.fixture
    def config(self):
        from core.config import EmbeddingConfig
        return EmbeddingConfig(
            provider="openai", api_key="test-key", model="text-embedding-3-small", dimension=1536
        )

    def test_embed(self, config):
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"index": 0, "embedding": [0.1] * 1536},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            }
            mock_client.post.return_value = mock_response

            provider = RemoteEmbeddingProvider(config)
            result = provider.embed("hello")
            assert len(result) == 1536

    def test_embed_batch_maintains_order(self, config):
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
            # Should be re-sorted by index
            assert results[0][0] == 0.1  # index 0

    def test_dimension_mismatch(self, config):
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
```

## Instructions for Cursor CLI

1. Replace `core/embedding.py` with the full implementation above
2. Create `tests/test_embedding.py` with the test suite above
3. Update `core/__init__.py` to export new embedding classes
4. Run `uv run python -m pytest tests/test_embedding.py -v` and report results
5. Run `uv run python -c "from core.embedding import EmbeddingProvider, LocalEmbeddingProvider, RemoteEmbeddingProvider, EmbeddingProviderFactory; print('Imports OK')"`