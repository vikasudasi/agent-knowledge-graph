"""Embedding providers — local (sentence-transformers) and remote (OpenRouter/OpenAI)."""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

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
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        """Embed a query text (may use different pooling for asymmetric models)."""
        return self.embed(text)

    def _validate_dimension(self, vector: list[float]) -> None:
        """Check vector dimension matches config."""
        expected = self._config.dimension
        if len(vector) != expected:
            raise EmbeddingDimensionMismatchError(
                f"Expected dimension {expected}, got {len(vector)}. "
                "Check config.embedding.dimension or the model being used."
            )


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers local inference — zero API key needed."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._model: Any | None = None
        self._device: str | None = None

    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA detected, using GPU for embeddings")
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS detected (Apple Silicon), using GPU for embeddings")
            return "mps"

        logger.info("No GPU detected, using CPU for embeddings")
        return "cpu"

    def _load_model(self) -> Any:
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            ) from exc

        self._device = self._detect_device()
        logger.info(f"Loading embedding model: {self._config.local_model}")
        self._model = SentenceTransformer(self._config.local_model, device=self._device)
        return self._model

    @functools.lru_cache(maxsize=4096)
    def embed(self, text: str) -> list[float]:
        model = self._load_model()
        vector = model.encode(text, normalize_embeddings=True).tolist()
        self._validate_dimension(vector)
        return vector  # type: ignore[no-any-return]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        results = [vector.tolist() for vector in vectors]
        for vector in results:
            self._validate_dimension(vector)
        return results

    def embed_query(self, text: str) -> list[float]:
        """For symmetric models (all-MiniLM-L6-v2), same as embed()."""
        return self.embed(text)


class RemoteEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding API (OpenRouter, OpenAI, etc.)."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._api_base = "https://openrouter.ai/api/v1"
        self._api_key = config.api_key
        self._model_name = config.model or "text-embedding-3-small"
        self._client = httpx.Client(
            base_url=self._api_base.rstrip("/") + "/",
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
        sorted_data = sorted(data["data"], key=lambda item: item["index"])
        results = [item["embedding"] for item in sorted_data]
        for vector in results:
            self._validate_dimension(vector)
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
        """Register a custom embedding provider."""
        cls._providers[name] = provider_cls

    @classmethod
    def create(cls, config: KGConfig) -> EmbeddingProvider:
        """Create an embedding provider from root config."""
        provider_name = config.embedding.provider
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown embedding provider: {provider_name}. Available: {', '.join(cls._providers)}")
        return cls._providers[provider_name](config.embedding)
