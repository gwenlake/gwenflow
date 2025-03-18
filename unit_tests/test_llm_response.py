import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import vcr
from dotenv import load_dotenv

# Import your Azure OpenAI–derived class.
from gwenflow.llms.azure_openai import ChatAzureOpenAI

# Environment variable determining test behavior.
TEST_ENV = os.environ.get("TEST_ENV", "local")
load_dotenv(".env.test")

# Define a fake completion object that simulates the response from the Azure OpenAI client.
class FakeCompletion:
    def __init__(self, content):
        self._content = content

    def model_dump(self):
        return {
            "id": "test-id",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "content": self._content,
                    "tool_calls": None  # Simulate no tool calls.
                }
            }]
        }

# Fixture for the ChatAzureOpenAI instance.
@pytest.fixture
def chat():
    return ChatAzureOpenAI()

# Fixture for a list of messages.
# Includes keys: "name", "tool_call_id", and "tool_calls" with default None.
@pytest.fixture
def dummy_message():
    messages = [
        {
            "role": "user",
            "content": "Get some recent news about Argentina.",
            "name": None,
            "tool_call_id": None,
            "tool_calls": None,
        }
    ]
    return messages

# Helper function to cast messages so that they have attribute access.
def cast_messages(m):
    keys = ["name", "tool_call_id", "tool_calls"]
    if isinstance(m, dict):
        for key in keys:
            m.setdefault(key, None)
        return [SimpleNamespace(**m)]
    elif isinstance(m, list):
        new_list = []
        for msg in m:
            if isinstance(msg, dict):
                for key in keys:
                    msg.setdefault(key, None)
                new_list.append(SimpleNamespace(**msg))
            else:
                new_list.append(msg)
        return new_list
    return m

if TEST_ENV == "local":
    # In local mode, record the cassette and (optionally) capture a snapshot.
    my_vcr = vcr.VCR(
        record_mode="once",
        cassette_library_dir="unit_tests/__cassettes__",
    )

    @pytest.mark.vcr()
    @my_vcr.use_cassette("chat_azure_invoke.yaml")
    def test_invoke_local(chat, dummy_message, snapshot):
        # Use the cast_messages helper to ensure the messages have the correct attributes.
        chat._cast_messages = cast_messages

        # Make a real API call (recorded by VCR)
        result = chat.invoke(dummy_message)
        assert result.choices[0].message.content is not None

        snapshot.assert_match(result.model_dump())
else:
    # In non-local (e.g. GitHub) environments, mock the API response.
    def test_invoke_github(monkeypatch, chat, dummy_message):
        chat._cast_messages = cast_messages

        # Create a fake client that returns a predetermined response.
        fake_completion = FakeCompletion("mocked azure response")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_completion

        # Monkeypatch get_client to use the fake client.
        monkeypatch.setattr(ChatAzureOpenAI, "get_client", lambda self: fake_client)
        result = chat.invoke(dummy_message)
        assert result.choices[0].message.content == "mocked azure response"