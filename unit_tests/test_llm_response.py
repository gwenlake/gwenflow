import json
from typing import Optional

import pytest
from dotenv import load_dotenv
from syrupy.matchers import path_type

from gwenflow import ChatAzureOpenAI
from gwenflow.tools import WebsiteReaderTool, DuckDuckGoNewsTool

# Load environment variables
load_dotenv(override=True, dotenv_path=".env.test")

@pytest.fixture
def chat_model_tools():
    """Fixture to initialize the ChatAzureOpenAI model."""
    return ChatAzureOpenAI(model="gpt-4o-mini", tools=[DuckDuckGoNewsTool(), WebsiteReaderTool()], temperature=0)

@pytest.fixture
def chat_model():
    """Fixture to initialize the ChatAzureOpenAI model."""
    return ChatAzureOpenAI(model="gpt-4o-mini", temperature=0)

@pytest.mark.vcr()
def test_response_content(chat_model, snapshot_json):
    """Test if the response stream returns valid data."""

    load_dotenv(override=True)

    messages = [
        {
            "role": "user",
            "content": "How to make cookies?",
        }
    ]
    response = chat_model.response(messages=messages)

    # Convert JSON string to dict so that matchers work properly
    actual = json.loads(response.model_dump_json())

    matcher = path_type({
        "id": (str,),
        "created_at" : (int,),
        "finish_reason": (Optional[str],)
    })

    assert actual == snapshot_json(matcher=matcher)