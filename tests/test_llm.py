"""Tests for the LLM abstraction layer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, Field

from core.llm import (
    LLMAuthenticationError,
    LLMClient,
    LLMError,
    LLMInvalidResponseError,
    LLMProviderFactory,
    LLMRateLimitError,
    LLMTimeoutError,
    OpenRouterProvider,
    _parse_json_response,
)


class TestParseJsonResponse:
    def test_plain_json(self) -> None:
        assert _parse_json_response('{"key": "value"}') == {"key": "value"}

    def test_markdown_fenced_json(self) -> None:
        result = _parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_list_json(self) -> None:
        assert _parse_json_response("[1, 2, 3]") == [1, 2, 3]

    def test_invalid_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("{invalid}")


@pytest.fixture
def mock_config():
    from core.config import LLMConfig

    return LLMConfig(api_key="test-key", base_url="https://test.openrouter.ai/api/v1")


@pytest.fixture
def provider(mock_config):
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        openrouter_provider = OpenRouterProvider(mock_config)
        openrouter_provider._client = mock_client
        yield openrouter_provider


class TestOpenRouterProvider:
    def test_init_stores_config(self, mock_config) -> None:
        with patch("httpx.Client"):
            provider = OpenRouterProvider(mock_config)
            assert provider.config == mock_config

    def test_init_warns_no_key(self, mock_config, caplog) -> None:
        mock_config.api_key = ""
        with patch("httpx.Client"):
            with caplog.at_level("WARNING"):
                OpenRouterProvider(mock_config)
                assert "No LLM API key" in caplog.text

    def test_chat_basic(self, provider) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        provider._client.post.return_value = mock_response

        result = provider.chat([{"role": "user", "content": "Say hello"}])
        assert result == "Hello!"

    def test_chat_model_override(self, provider) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
        }
        provider._client.post.return_value = mock_response

        provider.chat([{"role": "user", "content": "hi"}], model="custom-model")
        call_body = provider._client.post.call_args[1]["json"]
        assert call_body["model"] == "custom-model"

    def test_extract_structured(self, provider) -> None:
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

    def test_extract_structured_invalid_json(self, provider) -> None:
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

    def test_extract_structured_schema_validation(self, provider) -> None:
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

    def test_generate(self, provider) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated text"}}],
        }
        provider._client.post.return_value = mock_response

        result = provider.generate("Write something")
        assert result == "Generated text"

    def test_rate_limit_retry(self, provider) -> None:
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

    def test_authentication_error(self, provider) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 401
        provider._client.post.return_value = mock_response

        with pytest.raises(LLMAuthenticationError):
            provider.chat([{"role": "user", "content": "test"}])

    def test_timeout_retry(self, provider) -> None:
        provider._client.post.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(LLMTimeoutError), patch("time.sleep"):
            provider.chat([{"role": "user", "content": "test"}])

    def test_empty_content_raises(self, provider) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": None}}],
        }
        provider._client.post.return_value = mock_response

        with pytest.raises(LLMInvalidResponseError, match="empty content"):
            provider.chat([{"role": "user", "content": "test"}])

    def test_generate_with_system_prompt(self, provider) -> None:
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


class TestLLMProviderFactory:
    def test_create_openrouter(self) -> None:
        from core.config import KGConfig

        cfg = KGConfig()
        cfg.llm.api_key = "test-key"
        provider = LLMProviderFactory.create(cfg)
        assert isinstance(provider, OpenRouterProvider)

    def test_create_unknown_provider(self) -> None:
        from core.config import KGConfig

        cfg = KGConfig()
        cfg.llm.provider = "nonexistent"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMProviderFactory.create(cfg)

    def test_register_custom_provider(self) -> None:
        class FakeProvider(LLMClient):
            def chat(self, messages, model=None, temperature=0.7, max_tokens=None):  # noqa: ANN001
                return "fake"

            def generate(self, prompt, model=None, temperature=0.7, system_prompt=None):  # noqa: ANN001
                return "fake"

            def extract_structured(self, messages, schema, model=None, system_prompt=None):  # noqa: ANN001
                return schema()

        LLMProviderFactory.register("fake", FakeProvider)

        from core.config import KGConfig

        cfg = KGConfig()
        cfg.llm.provider = "fake"
        cfg.llm.api_key = "test"
        provider = LLMProviderFactory.create(cfg)
        assert isinstance(provider, FakeProvider)


def test_llm_error_classes_exist() -> None:
    assert issubclass(LLMRateLimitError, LLMError)
