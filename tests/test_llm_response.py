import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import vcr

from gwenflow.llms.azure.chat import ChatAzureOpenAI

TEST_ENV = os.getenv("TEST_ENV", "local")

# Setup VCR for integration tests (record once then replay)
my_vcr = vcr.VCR(record_mode="once", cassette_library_dir="tests/__cassettes__")

class FakeCompletion:
    """Fake completion object mimicking the Azure SDK/OpenAI response."""
    def __init__(self, content):
        self._content = content
        self.choices = [
            type("Choice", (), {
                "message": type("Message", (), {
                    "content": content,
                    "tool_calls": None
                })()
            })()
        ]

    def model_dump(self):
        return {
            "id": "test-id",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"content": self._content, "tool_calls": None}
                }
            ],
        }

class PatchedChatAzureOpenAI(ChatAzureOpenAI):
    def input_to_message_list(self, input):
        return cast_messages(input)

@pytest.fixture
def chat():
    return PatchedChatAzureOpenAI()


@pytest.fixture
def sample_user_message():
    """Sample user message input for testing the chat invoke."""
    return [{
        "role": "user",
        "content": "Get some recent news about Argentina.",
        # name, tool_call_id, tool_calls will be set to None by cast_messages if missing
    }]

def cast_messages(message):
    """Helper to convert message dicts to namespace objects (mimics internal casting)."""
    keys = ["name", "tool_call_id", "tool_calls"]
    msgs = [message] if isinstance(message, dict) else message
    result = []
    for m in msgs:
        if isinstance(m, dict):
            # Ensure all expected keys exist, defaulting to None
            for key in keys:
                m.setdefault(key, None)
            result.append(SimpleNamespace(**m))
        else:
            # If already a SimpleNamespace or other type, use as-is
            result.append(m)
    return result

@pytest.mark.skipif(TEST_ENV != "local", reason="Requires local environment (VCR cassette)")
@my_vcr.use_cassette("chat_azure_invoke.yaml")
def test_invoke_returns_content_with_real_api(chat, sample_user_message, snapshot):
    """
    Integration test: ChatAzureOpenAI.invoke should return a response containing content
    when calling the real Azure OpenAI service (captured via VCR).
    """
    result = chat.invoke(sample_user_message)
    assert result.choices[0].message.content is not None
    snapshot.assert_match(result.model_dump())

@pytest.mark.skipif(TEST_ENV == "local", reason="Uses mocked Azure client instead of real API")
def test_invoke_returns_content_with_mock(monkeypatch, chat, sample_user_message):
    """
    Unit test: ChatAzureOpenAI.invoke should return expected content when Azure client is mocked.
    """
    fake_completion = FakeCompletion(content="mocked azure response")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_completion
    monkeypatch.setattr(ChatAzureOpenAI, "get_client", lambda self: fake_client)
    result = chat.invoke(sample_user_message)
    assert result.choices[0].message.content == "mocked azure response"