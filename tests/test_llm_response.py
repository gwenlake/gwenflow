"""Tests for ChatAzureOpenAI — Azure path through the shared ModelResponse parser.

ChatAzureOpenAI extends ChatOpenAI, so the response parsing is the same code
path. These tests assert against ModelResponse (the current shape) and verify
the Azure-specific client wiring.

The live VCR test stays skipped until a cassette is recorded — run with
`--record-mode=once` after configuring AZURE_OPENAI_* env vars to capture one.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import vcr

from gwenflow.llms.azure import ChatAzureOpenAI
from gwenflow.types import Message

CASSETTE_DIR = Path(__file__).parent / "__cassettes__"
my_vcr = vcr.VCR(
    record_mode="once",
    cassette_library_dir=str(CASSETTE_DIR),
    filter_headers=["api-key", "authorization"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_azure_response(content="hello from azure", tool_calls=None, finish_reason="stop"):
    """Build an OpenAI SDK-shaped completion (Azure mirrors the same shape)."""
    msg = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=7,
        total_tokens=19,
        completion_tokens_details=None,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def chat():
    return ChatAzureOpenAI(
        api_key="test",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        api_version="2024-08-01-preview",
    )


@pytest.fixture
def sample_user_message():
    return [Message(role="user", content="Get some recent news about Argentina.")]


# ---------------------------------------------------------------------------
# Client wiring
# ---------------------------------------------------------------------------


def test_get_client_uses_azure_specific_params(monkeypatch):
    """Azure client must receive the Azure-specific kwargs, not vanilla OpenAI ones."""
    captured = {}

    def fake_azure_client(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("gwenflow.llms.azure.AzureOpenAI", fake_azure_client)

    llm = ChatAzureOpenAI(
        api_key="key123",
        azure_endpoint="https://acme.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        api_version="2024-08-01-preview",
    )
    llm.get_client()

    assert captured["api_key"] == "key123"
    assert captured["azure_endpoint"] == "https://acme.openai.azure.com"
    assert captured["azure_deployment"] == "gpt-4o-mini"
    assert captured["api_version"] == "2024-08-01-preview"


def test_get_client_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        ChatAzureOpenAI()._get_client_params()


# ---------------------------------------------------------------------------
# Mocked invoke — returns ModelResponse
# ---------------------------------------------------------------------------


def test_invoke_returns_model_response_content(chat, monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_azure_response("mocked azure response")
    chat.client = fake_client

    result = chat.invoke("hi")

    assert result.content == "mocked azure response"
    assert result.text == "mocked azure response"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 7


def test_invoke_accepts_message_list(chat, sample_user_message):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_azure_response("ok")
    chat.client = fake_client

    result = chat.invoke(sample_user_message)
    assert result.content == "ok"

    _, kwargs = fake_client.chat.completions.create.call_args
    sent = kwargs["messages"]
    assert sent[0]["role"] == "user"
    assert sent[0]["content"] == "Get some recent news about Argentina."


def test_invoke_extracts_tool_calls(chat):
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments='{"city": "Paris"}'),
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_azure_response(
        content=None, tool_calls=[tool_call], finish_reason="tool_calls"
    )
    chat.client = fake_client

    result = chat.invoke("weather?")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == '{"city": "Paris"}'


# ---------------------------------------------------------------------------
# Live VCR test (kept skipped until a cassette is recorded)
# ---------------------------------------------------------------------------

CASSETTE = CASSETTE_DIR / "chat_azure_invoke.yaml"


@pytest.mark.skipif(
    not CASSETTE.exists(),
    reason="No VCR cassette recorded yet. Run with AZURE_OPENAI_* env vars and "
    "`pytest --record-mode=once` to capture, then commit the cassette.",
)
@my_vcr.use_cassette("chat_azure_invoke.yaml")
def test_invoke_live_via_vcr(chat, sample_user_message):
    """Integration test against Azure OpenAI, played back from a VCR cassette."""
    result = chat.invoke(sample_user_message)
    assert result.content is not None
    assert result.usage.input_tokens > 0
