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
    """Parse JSON from response text. Handles markdown code fences and double-wrapped JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        # Handle double-wrapped JSON: {{...}} — strip one layer
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            # Try extracting the inner content
            inner = stripped[1:-1].strip()
            if inner.startswith("{"):
                try:
                    return json.loads(inner)  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    pass
            # Some LLMs wrap in {"response": {...}} or {"content": {...}}
            try:
                outer = json.loads(stripped)
                for key in ("response", "content", "result", "data", "output"):
                    val = outer.get(key)
                    if val and isinstance(val, (dict, list)):
                        return val
            except json.JSONDecodeError:
                pass
        raise  # re-raise original error if nothing worked


class LLMClient(ABC):
    """Abstract LLM client that all providers implement."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def config(self) -> LLMConfig:
        return self._config

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the text response."""
        raise NotImplementedError

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
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> str:
        """Simple text generation from a string prompt."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any resources. Override in providers with async clients."""
        return None


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

        max_retries = self._config.max_retries

        for attempt in range(max_retries + 1):
            try:
                response = self._client.post("chat/completions", json=body)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2**attempt))
                    logger.warning(f"Rate limited, retrying in {retry_after}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_after)
                    continue
                if response.status_code == 401:
                    raise LLMAuthenticationError("Invalid API key. Set KG_LLM_API_KEY in your environment.")
                if response.status_code == 402:
                    raise LLMError("Insufficient credits/balance. Top up your OpenRouter account.")
                response.raise_for_status()
                return response

            except httpx.TimeoutException as exc:
                if attempt < max_retries:
                    delay = 2 ** (attempt + 1)
                    logger.warning(f"Timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise LLMTimeoutError(f"Request timed out after {max_retries + 1} attempts") from exc
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                if attempt < max_retries:
                    delay = 2 ** (attempt + 1)
                    logger.warning(f"HTTP error, retrying in {delay}s (attempt {attempt + 1}/{max_retries}): {exc}")
                    time.sleep(delay)
                else:
                    raise LLMError(f"Request failed after {max_retries + 1} attempts: {exc}") from exc

        raise LLMError("Request failed (exhausted retries)")

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

            usage = data.get("usage", {})
            if usage:
                logger.debug(
                    f"LLM tokens: {usage.get('prompt_tokens', '?')} in, {usage.get('completion_tokens', '?')} out"
                )
            else:
                prompt_tokens = _count_tokens(json.dumps(messages), model=model)
                completion_tokens = _count_tokens(str(content), model=model)
                logger.debug(f"LLM tokens (estimated): {prompt_tokens} in, {completion_tokens} out")
            return content.strip()  # type: ignore[no-any-return]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMInvalidResponseError(f"Unexpected API response format: {exc}") from exc

    def extract_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> BaseModel:
        model = self._get_model(model, task="extraction")
        system_prompt = (
            system_prompt or "You are a structured data extraction assistant. Always respond with valid JSON."
        )
        messages = _ensure_system_prompt(messages, system_prompt)

        response = self._request(
            messages,
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        try:
            data = response.json()
            raw_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMInvalidResponseError(f"Unexpected API response format: {exc}") from exc

        try:
            parsed = _parse_json_response(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMInvalidResponseError(f"LLM returned invalid JSON: {exc}\nRaw: {raw_text[:500]}") from exc

        try:
            if isinstance(parsed, list):
                if hasattr(schema, "model_validate"):
                    return [schema.model_validate(item) for item in parsed]  # type: ignore[return-value]
            return schema.model_validate(parsed)
        except ValidationError as exc:
            raise LLMInvalidResponseError(
                f"LLM output failed schema validation: {exc}\nParsed: {json.dumps(parsed, indent=2)[:500]}"
            ) from exc

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, model=model, temperature=temperature)


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
            raise ValueError(f"Unknown LLM provider: {provider_name}. Available: {', '.join(cls._providers)}")
        return cls._providers[provider_name](config.llm)
