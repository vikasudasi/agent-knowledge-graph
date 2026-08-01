# agent-knowledge-graph — LLM Abstraction Layer (Task 4)

## What to Build

Implement a provider-agnostic LLM abstraction layer with OpenRouter as the default provider. Pipelines use this for entity/relation extraction and the query layer uses it for NL→Cypher translation.

## Files to Create/Modify

- `core/llm.py` — full replacement with `LLMClient`, `OpenRouterProvider`, `LLMProviderFactory`
- `cli/llm.py` — `kg llm` test commands (ping a model, test structured extraction)
- `tests/test_llm.py` — comprehensive test suite
- `core/__init__.py` — update exports

## Requirements

1. **`LLMClient`** — abstract base with `chat()`, `extract_structured()`, `generate()`
2. **`OpenRouterProvider`** — OpenAI-compatible client hitting `https://openrouter.ai/api/v1`
3. **Structured extraction** — uses `response_format={"type": "json_object"}` with Pydantic schema validation + fallback parsing
4. **Separate models** — `default_model`, `extraction_model`, `query_model` from config
5. **Retry** — 3 attempts with exponential backoff (2s, 4s, 8s)
6. **Token tracking** — approximate counting via `tiktoken` for logging
7. **Error handling** — rate limits, timeouts, invalid JSON — all with descriptive messages
8. **`LLMProviderFactory`** — creates the right provider from `config.llm.provider`
9. **Extensible** — adding Anthropic or OpenAI is one class + a config entry

## Implementation Details

### core/llm.py

```python
"""Provider-agnostic LLM abstraction with OpenRouter default."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from core.config import KGConfig, LLMConfig

logger = logging.getLogger(__name__)

# ── Exceptions ──────────────────────────────────────────────────────

class LLMError(Exception):
    """Base LLM error."""

class LLMRateLimitError(LLMError):
    """Rate limited."""

class LLMTimeoutError(LLMError):
    """Request timed out."""

class LLMInvalidResponseError(LLMError):
    """Response could not be parsed."""

class LLMAuthenticationError(LLMError):
    """Invalid API key."""

# ── Schema helpers ──────────────────────────────────────────────────

def _ensure_system_prompt(messages: list[dict[str, str]], default_system: str) -> list[dict[str, str]]:
    """Prepend a system message if not present."""
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": default_system}] + messages
    return messages


def _count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Approximate token count using tiktoken. Falls back to rough char/4 estimate."""
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def _parse_json_response(text: str) -> dict[str, Any] | list[Any]:
    """Parse JSON from response text. Handles markdown code fences."""
    text = text.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


# ── Abstract Base ───────────────────────────────────────────────────

class LLMClient(ABC):
    """Abstract LLM client that all providers implement."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def config(self) -> LLMConfig:
        return self._config

    # ── Core methods ────────────────────────────────────────────────

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the text response."""
        ...

    def extract_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Extract structured data matching a Pydantic schema.

        Uses response_format={'type': 'json_object'} for reliable JSON output,
        then validates against the provided Pydantic model.
        Falls back to JSON parsing + validation if JSON mode is unavailable.
        """
        ...

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> str:
        """Simple text generation from a string prompt."""
        ...

    # ── Lifecycle ───────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Release any resources. Override in providers with async clients."""
        pass


# ── OpenRouter Provider ─────────────────────────────────────────────

class OpenRouterProvider(LLMClient):
    """OpenRouter-backed LLM client using OpenAI-compatible REST API."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            logger.warning("No LLM API key configured — set KG_LLM_API_KEY or llm.api_key in config")
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout,
        )
        self._async_client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout,
        )

    def _get_model(self, model: str | None, task: str = "default") -> str:
        """Resolve a model name with config fallback."""
        if model:
            return model
        return {
            "default": self._config.default_model,
            "extraction": self._config.extraction_model,
            "query": self._config.query_model,
        }.get(task, self._config.default_model)

    def _request(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Send a chat completion request with retry logic."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if response_format:
            body["response_format"] = response_format

        last_error: Exception | None = None
        max_retries = self._config.max_retries

        for attempt in range(max_retries + 1):
            try:
                response = self._client.post("chat/completions", json=body)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(f"Rate limited, retrying in {retry_after}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_after)
                    continue
                if response.status_code == 401:
                    raise LLMAuthenticationError("Invalid API key. Set KG_LLM_API_KEY in your environment.")
                if response.status_code == 402:
                    raise LLMError("Insufficient credits/balance. Top up your OpenRouter account.")
                response.raise_for_status()
                return response

            except httpx.TimeoutException as e:
                if attempt < max_retries:
                    delay = 2 ** (attempt + 1)
                    logger.warning(f"Timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise LLMTimeoutError(f"Request timed out after {max_retries + 1} attempts") from e
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                if attempt < max_retries:
                    delay = 2 ** (attempt + 1)
                    logger.warning(f"HTTP error, retrying in {delay}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(delay)
                else:
                    raise LLMError(f"Request failed after {max_retries + 1} attempts: {e}") from e

        raise LLMError("Request failed (exhausted retries)")

    # ── Public API ──────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        model = self._get_model(model)
        response = self._request(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if content is None:
                raise LLMInvalidResponseError("LLM returned empty content (possible content filter)")
            # Log token usage
            usage = data.get("usage", {})
            if usage:
                logger.debug(f"LLM tokens: {usage.get('prompt_tokens', '?')} in, {usage.get('completion_tokens', '?')} out")
            return content.strip()
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMInvalidResponseError(f"Unexpected API response format: {e}")

    def extract_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> BaseModel:
        model = self._get_model(model, task="extraction")
        system_prompt = system_prompt or "You are a structured data extraction assistant. Always respond with valid JSON."
        messages = _ensure_system_prompt(messages, system_prompt)

        response = self._request(
            messages,
            model=model,
            temperature=0.1,  # Low temp for reliable extraction
            response_format={"type": "json_object"},
        )

        try:
            data = response.json()
            raw_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMInvalidResponseError(f"Unexpected API response format: {e}")

        # Parse and validate against schema
        try:
            parsed = _parse_json_response(raw_text)
        except json.JSONDecodeError as e:
            raise LLMInvalidResponseError(f"LLM returned invalid JSON: {e}\nRaw: {raw_text[:500]}")

        try:
            if isinstance(parsed, list):
                # Handle list responses — wrap in schema validation per item
                if hasattr(schema, "model_validate"):
                    return [schema.model_validate(item) for item in parsed]
            return schema.model_validate(parsed)
        except ValidationError as e:
            raise LLMInvalidResponseError(f"LLM output failed schema validation: {e}\nParsed: {json.dumps(parsed, indent=2)[:500]}")

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, model=model, temperature=temperature)


# ── Factory ─────────────────────────────────────────────────────────

class LLMProviderFactory:
    """Creates the right LLM provider based on config."""

    _providers: dict[str, type[LLMClient]] = {
        "openrouter": OpenRouterProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: type[LLMClient]) -> None:
        """Register a custom provider."""
        cls._providers[name] = provider_cls

    @classmethod
    def create(cls, config: KGConfig) -> LLMClient:
        """Create an LLM client from the root config."""
        provider_name = config.llm.provider
        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown LLM provider: {provider_name}. "
                f"Available: {', '.join(cls._providers)}"
            )
        return cls._providers[provider_name](config.llm)
```

### cli/llm.py

```python
"""LLM-related CLI commands for testing and debugging."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from core.config import load_config
from core.llm import LLMProviderFactory

app = typer.Typer(help="LLM provider commands")
console = Console()


@app.command()
def ping() -> None:
    """Test LLM connectivity with a simple chat completion."""
    cfg = load_config(auto_create=False)
    try:
        client = LLMProviderFactory.create(cfg)
        response = client.generate(
            "Respond with exactly: OK. Say nothing else.",
            temperature=0.1,
        )
        console.print(Panel(f"[green]{response.strip()}[/]", title="LLM Ping"))
    except Exception as e:
        console.print(f"[red]LLM ping failed: {e}[/]")
        raise typer.Exit(1)


@app.command()
def extract() -> None:
    """Test structured extraction with a simple schema."""
    from pydantic import BaseModel, Field

    class TestExtract(BaseModel):
        name: str = Field(..., description="The person's name")
        role: str = Field(..., description="Their role or title")
        confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    cfg = load_config(auto_create=False)
    try:
        client = LLMProviderFactory.create(cfg)
        result = client.extract_structured(
            messages=[
                {"role": "user", "content": "Vik is the architect behind agent-knowledge-graph."
                 " He works on AI and cloud architecture."}
            ],
            schema=TestExtract,
        )
        console.print(Panel(f"[green]{result.model_dump_json(indent=2)}[/]", title="Structured Extraction"))
    except Exception as e:
        console.print(f"[red]Extraction failed: {e}[/]")
        raise typer.Exit(1)
```

### Wire into main app

In `cli/main.py`, add the LLM subcommand group:

```python
from cli.llm import app as llm_app

app.add_typer(llm_app, name="llm", help="LLM provider commands (ping, extract)")
```

### Update core/__init__.py

```python
from core.config import EmbeddingConfig, KGConfig, LLMConfig, Neo4jConfig, load_config
from core.graph import Neo4jClient
from core.llm import LLMClient, LLMProviderFactory, OpenRouterProvider
from core.models import GraphStats, PipelineCheckpoint, QueryResult, Resource, Relationship

__all__ = [
    "KGConfig", "LLMConfig", "EmbeddingConfig", "Neo4jConfig", "load_config",
    "Neo4jClient",
    "LLMClient", "OpenRouterProvider", "LLMProviderFactory",
    "Resource", "Relationship", "GraphStats", "PipelineCheckpoint", "QueryResult",
]
```

### Tests (tests/test_llm.py) — mocked, no real API calls

```python
"""Tests for the LLM abstraction layer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, Field

from core.llm import (
    LLMClient,
    LLMAuthenticationError,
    LLMError,
    LLMInvalidResponseError,
    LLMProviderFactory,
    LLMRateLimitError,
    LLMTimeoutError,
    OpenRouterProvider,
    _parse_json_response,
)


# ── Test JSON parser ────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_plain_json(self):
        assert _parse_json_response('{"key": "value"}') == {"key": "value"}

    def test_markdown_fenced_json(self):
        result = _parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_list_json(self):
        assert _parse_json_response('[1, 2, 3]') == [1, 2, 3]

    def test_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("{invalid}")


# ── Test OpenRouter Provider (mocked) ───────────────────────────────

@pytest.fixture
def mock_config():
    from core.config import LLMConfig
    return LLMConfig(api_key="test-key", base_url="https://test.openrouter.ai/api/v1")


@pytest.fixture
def provider(mock_config):
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        provider = OpenRouterProvider(mock_config)
        provider._client = mock_client
        yield provider


class TestOpenRouterProvider:
    def test_init_stores_config(self, mock_config):
        with patch("httpx.Client"):
            provider = OpenRouterProvider(mock_config)
            assert provider.config == mock_config

    def test_init_warns_no_key(self, mock_config, caplog):
        mock_config.api_key = ""
        with patch("httpx.Client"):
            with caplog.at_level("WARNING"):
                OpenRouterProvider(mock_config)
                assert "No LLM API key" in caplog.text

    def test_chat_basic(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        provider._client.post.return_value = mock_response

        result = provider.chat([{"role": "user", "content": "Say hello"}])
        assert result == "Hello!"

    def test_chat_model_override(self, provider):
        """Should use the specified model, not the default."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
        }
        provider._client.post.return_value = mock_response

        provider.chat([{"role": "user", "content": "hi"}], model="custom-model")
        call_body = provider._client.post.call_args[1]["json"]
        assert call_body["model"] == "custom-model"

    def test_extract_structured(self, provider):
        class Person(BaseModel):
            name: str = Field(..., description="Name")
            age: int = Field(..., description="Age")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"name": "Alice", "age": 30}'}}],
        }
        provider._client.post.return_value = mock_response

        result = provider.extract_structured(
            [{"role": "user", "content": "Alice is 30"}],
            schema=Person,
        )
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_extract_structured_invalid_json(self, provider):
        class Person(BaseModel):
            name: str

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "not json"}}],
        }
        provider._client.post.return_value = mock_response

        with pytest.raises(LLMInvalidResponseError):
            provider.extract_structured([{"role": "user", "content": "test"}], schema=Person)

    def test_extract_structured_schema_validation(self, provider):
        class Person(BaseModel):
            name: str

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"age": 30}'}}],
        }
        provider._client.post.return_value = mock_response

        with pytest.raises(LLMInvalidResponseError):
            provider.extract_structured([{"role": "user", "content": "test"}], schema=Person)

    def test_generate(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated text"}}],
        }
        provider._client.post.return_value = mock_response

        result = provider.generate("Write something")
        assert result == "Generated text"

    def test_rate_limit_retry(self, provider):
        """Should retry on 429 and eventually succeed."""
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {}

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": "Success after retry"}}],
        }

        provider._client.post.side_effect = [rate_limit_response, success_response]

        with patch("time.sleep") as mock_sleep:
            result = provider.chat([{"role": "user", "content": "test"}])
            assert result == "Success after retry"
            assert mock_sleep.call_count == 1

    def test_authentication_error(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 401
        provider._client.post.return_value = mock_response

        with pytest.raises(LLMAuthenticationError):
            provider.chat([{"role": "user", "content": "test"}])

    def test_timeout_retry(self, provider):
        """Should retry on timeout and raise after exhausting retries."""
        provider._client.post.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(LLMTimeoutError), patch("time.sleep"):
            provider.chat([{"role": "user", "content": "test"}])

    def test_empty_content_raises(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": None}}],
        }
        provider._client.post.return_value = mock_response

        with pytest.raises(LLMInvalidResponseError, match="empty content"):
            provider.chat([{"role": "user", "content": "test"}])

    def test_generate_with_system_prompt(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
        }
        provider._client.post.return_value = mock_response

        provider.generate("prompt", system_prompt="You are helpful")
        call_body = provider._client.post.call_args[1]["json"]
        assert call_body["messages"][0]["role"] == "system"
        assert call_body["messages"][0]["content"] == "You are helpful"


# ── Test Factory ────────────────────────────────────────────────────

class TestLLMProviderFactory:
    def test_create_openrouter(self):
        from core.config import KGConfig
        cfg = KGConfig()
        cfg.llm.api_key = "test-key"
        provider = LLMProviderFactory.create(cfg)
        assert isinstance(provider, OpenRouterProvider)

    def test_create_unknown_provider(self):
        from core.config import KGConfig
        cfg = KGConfig()
        cfg.llm.provider = "nonexistent"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMProviderFactory.create(cfg)

    def test_register_custom_provider(self):
        class FakeProvider(LLMClient):
            def chat(self, messages, model=None, temperature=0.7, max_tokens=None):
                return "fake"

            def generate(self, prompt, model=None, temperature=0.7, system_prompt=None):
                return "fake"

            def extract_structured(self, messages, schema, model=None, system_prompt=None):
                return schema()

        from core.config import LLMConfig
        LLMProviderFactory.register("fake", FakeProvider)

        from core.config import KGConfig
        cfg = KGConfig()
        cfg.llm.provider = "fake"
        cfg.llm.api_key = "test"
        provider = LLMProviderFactory.create(cfg)
        assert isinstance(provider, FakeProvider)
```

## Instructions for Cursor CLI

1. Replace `core/llm.py` with the full implementation above
2. Create `cli/llm.py` with `ping` and `extract` subcommands
3. Create `tests/test_llm.py` with the complete test suite
4. Update `cli/main.py` to add the `llm` subcommand group
5. Update `core/__init__.py` to export new classes
6. Run `uv run python -m pytest tests/test_llm.py -v` and report results
7. Run `uv run python -c "from core.llm import LLMClient, OpenRouterProvider, LLMProviderFactory; print('Imports OK')"`