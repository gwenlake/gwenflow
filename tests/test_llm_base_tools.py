"""Every provider accepts any BaseTool, not only Tool.

The coding bundle (ReadFileTool, ShellTool...), MCP tools and handoffs inherit
BaseTool. ChatOpenAI used to crash on them (`isinstance(t, Tool)` in the text
tool-call recovery) as soon as the model answered in plain text; this pins the
behaviour for every provider: tool schemas are built, and a plain text answer
is parsed, with such tools on offer.
"""

from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from gwenflow.llms.anthropic import ChatAnthropic
from gwenflow.llms.azure import ChatAzureOpenAI
from gwenflow.llms.deepseek import ChatDeepSeek
from gwenflow.llms.google import ChatGoogle
from gwenflow.llms.gwenlake import ChatGwenlake
from gwenflow.llms.mistral import ChatMistral
from gwenflow.llms.ollama import ChatOllama
from gwenflow.llms.openai import ChatOpenAI
from gwenflow.llms.openai_response import ResponseOpenAI
from gwenflow.tools import ReadFileTool, ShellTool, Tool

ANSWER = "Done, the report is written."


def lookup(path: str) -> str:
    """Look a path up.

    Args:
        path: The path.
    """
    return path


def tools():
    return [ReadFileTool(base_dir="."), ShellTool(base_dir="."), Tool(lookup)]


def openai_completion():
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ANSWER}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def anthropic_response():
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=ANSWER)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def mistral_response():
    msg = SimpleNamespace(role="assistant", content=ANSWER, tool_calls=None)
    choice = SimpleNamespace(index=0, message=msg, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(id="r1", choices=[choice], usage=usage, object=None, created=None, model="m")


def google_response():
    part = SimpleNamespace(text=ANSWER, thought=False, thought_signature=None, function_call=None, inline_data=None)
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part], role="model"), finish_reason="STOP")
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=5,
        cached_content_token_count=0,
        thoughts_token_count=0,
        total_token_count=15,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def responses_response():
    message = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=ANSWER)])
    return SimpleNamespace(output=[message], usage=None)


# (llm class, schema converter, native text response)
PROVIDERS = [
    (ChatOpenAI, "_tool_to_openai", openai_completion),
    (ChatGwenlake, "_tool_to_openai", openai_completion),
    (ChatDeepSeek, "_tool_to_openai", openai_completion),
    (ChatOllama, "_tool_to_openai", openai_completion),
    (ChatAzureOpenAI, "_tool_to_openai", openai_completion),
    (ResponseOpenAI, "_tool_to_responses", responses_response),
    (ChatAnthropic, "_tool_to_anthropic", anthropic_response),
    (ChatMistral, "_tool_to_mistral", mistral_response),
    (ChatGoogle, "_tool_to_google", google_response),
]


@pytest.mark.parametrize("cls, converter, response", PROVIDERS, ids=[p[0].__name__ for p in PROVIDERS])
def test_tool_schemas_accept_base_tools(cls, converter, response):
    llm = cls(model="m", api_key="x")
    schemas = [getattr(llm, converter)(tool) for tool in tools()]
    rendered = str(schemas)
    for name in ("ReadFileTool", "ShellTool", "lookup"):
        assert name in rendered


@pytest.mark.parametrize("cls, converter, response", PROVIDERS, ids=[p[0].__name__ for p in PROVIDERS])
def test_plain_text_answer_with_base_tools_on_offer(cls, converter, response):
    llm = cls(model="m", api_key="x")
    llm.tools = tools()
    parsed = llm._parse_response(response())
    assert parsed.text == ANSWER
    assert parsed.tool_calls == []
