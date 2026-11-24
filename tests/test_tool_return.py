"""`ToolReturn`: a tool result with metadata for the application next to what the model reads.

The model gets `return_value` only; `metadata` reaches the application on
`ToolResponse.metadata`, the tool `Message.extra["tool_metadata"]` and
`AgentEventToolCompleted.tool_metadata`. A tool that returns anything else
behaves as before.
"""

import asyncio
import threading
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from gwenflow import Agent, Tool, ToolReturn
from gwenflow.llms.anthropic import ChatAnthropic
from gwenflow.llms.base import ChatBase
from gwenflow.tools import BaseTool
from gwenflow.types import (
    TOOL_METADATA_KEY,
    AgentResponse,
    Message,
    ModelResponse,
    RequestUsage,
    TextContent,
    ToolCall,
    ToolResponse,
)

SOURCES = [{"title": "Rapport annuel 2025", "url": "https://example.org/rapport.pdf", "page": 12}]


@dataclass(kw_only=True)
class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Searches the knowledge base"

    def _run(self, query: str) -> ToolReturn:
        return ToolReturn(return_value={"passages": ["Le chiffre d'affaires a progressé de 12 %."]}, metadata=SOURCES)


@dataclass(kw_only=True)
class PlainTool(BaseTool):
    name: str = "plain"
    description: str = "Returns a dict"

    def _run(self, query: str) -> dict:
        return {"answer": query}


# ---------------------------------------------------------------------------
# BaseTool / Tool
# ---------------------------------------------------------------------------


def test_run_keeps_metadata_and_casts_the_value():
    result = SearchTool().run(query="ca")
    assert isinstance(result, ToolReturn)
    assert result.return_value == '{"passages": ["Le chiffre d\'affaires a progressé de 12 %."]}'
    assert result.metadata == SOURCES


def test_arun_keeps_metadata():
    result = asyncio.run(SearchTool().arun(query="ca"))
    assert isinstance(result, ToolReturn)
    assert result.metadata == SOURCES


def test_plain_result_is_still_text():
    assert PlainTool().run(query="x") == '{"answer": "x"}'
    assert asyncio.run(PlainTool().arun(query="x")) == '{"answer": "x"}'


def test_function_tool_can_return_tool_return():
    def lookup(code: str) -> ToolReturn:
        """Looks a code up."""
        return ToolReturn(return_value=f"code {code}: ok", metadata={"code": code})

    result = Tool(lookup).run(code="A1")
    assert result == ToolReturn(return_value="code A1: ok", metadata={"code": "A1"})


def test_tool_response_puts_metadata_on_the_message_only_when_set():
    with_metadata = ToolResponse(tool_name="search", tool_call_id="c1", content="text", metadata=SOURCES).to_message()
    assert with_metadata.extra == {TOOL_METADATA_KEY: SOURCES}
    without = ToolResponse(tool_name="search", tool_call_id="c1", content="text").to_message()
    assert without.extra == {}


def test_metadata_never_reaches_the_provider():
    message = ToolResponse(tool_name="search", tool_call_id="c1", content="text", metadata=SOURCES).to_message()
    assert message.to_openai() == {"role": "tool", "content": "text", "tool_call_id": "c1"}

    formatted = ChatAnthropic(api_key="test")._format_messages(
        [
            Message(role="user", content="q"),
            Message(
                role="assistant",
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
            ),
            message,
        ]
    )
    assert "Rapport annuel" not in repr(formatted)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def _turn(content=None, tool_calls=None):
    parts = [TextContent(content=content)] if content else []
    parts += tool_calls or []
    return ModelResponse(
        parts=parts,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=RequestUsage(input_tokens=10, output_tokens=5),
    )


def _scripted_llm(turns, seen):
    """Plays `turns` in order on every path; records the messages of each request in `seen`."""
    script = iter(turns)

    def invoke(input=None, **kwargs):
        seen.append(input)
        return next(script)

    async def ainvoke(input=None, **kwargs):
        return invoke(input)

    def stream(input=None, **kwargs):
        turn = invoke(input)
        if turn.content:
            yield ModelResponse(parts=[TextContent(content=turn.content)])
        yield ModelResponse(parts=list(turn.tool_calls), usage=turn.usage)

    async def astream(input=None, **kwargs):
        for chunk in stream(input):
            yield chunk

    llm = MagicMock()
    llm.get_context_size.return_value = 96000
    llm.response_format = None
    llm.tools = []
    llm.tool_choice = None
    llm.invoke.side_effect = invoke
    llm.ainvoke.side_effect = ainvoke
    llm.stream.side_effect = stream
    llm.astream.side_effect = astream
    llm.input_to_message_list.side_effect = ChatBase.input_to_message_list.__get__(llm)
    llm.get_thinking_parts.return_value = None
    return llm


def _run(agent, path, task="Quel est le CA ?"):
    if path == "run":
        return agent.run(task)
    if path == "arun":
        return asyncio.run(agent.arun(task))
    if path == "run_stream":
        return [e for e in agent.run_stream(task) if isinstance(e, AgentResponse)][-1]

    async def consume():
        return [e async for e in agent.arun_stream(task) if isinstance(e, AgentResponse)][-1]

    return asyncio.run(consume())


PATHS = ["run", "arun", "run_stream", "arun_stream"]


def _search_run(path, **agent_kwargs):
    seen = []
    call = ToolCall(id="c1", name="search", arguments='{"query": "CA 2025"}')
    llm = _scripted_llm([_turn(tool_calls=[call]), _turn("Le CA a progressé de 12 %.")], seen)
    agent = Agent(llm=llm, tools=[SearchTool()], **agent_kwargs)
    return agent, _run(agent, path), seen


@pytest.mark.parametrize("path", PATHS)
def test_agent_hands_metadata_to_the_application(path):
    agent, response, _ = _search_run(path)

    tool_message = next(m for m in response.messages if m.role == "tool")
    assert tool_message.extra[TOOL_METADATA_KEY] == SOURCES

    completed = next(e for e in response.events if e.event_type == "tool_completed")
    assert completed.tool_metadata == SOURCES
    assert completed.tool_name == "search"
    assert completed.tool_args == {"query": "CA 2025"}
    assert completed.content == '{"passages": ["Le chiffre d\'affaires a progressé de 12 %."]}'


@pytest.mark.parametrize("path", PATHS)
def test_model_reads_the_return_value_only(path):
    _, _, seen = _search_run(path)
    second_request = seen[1]
    tool_messages = [Message(**m) if isinstance(m, dict) else m for m in second_request]
    tool_message = next(m for m in tool_messages if m.role == "tool")
    assert tool_message.content == '{"passages": ["Le chiffre d\'affaires a progressé de 12 %."]}'
    assert "Rapport annuel" not in repr(tool_message.to_openai())


def test_long_output_is_clamped_but_metadata_survives():
    @dataclass(kw_only=True)
    class LongTool(BaseTool):
        name: str = "search"
        description: str = "Searches"

        def _run(self, query: str) -> ToolReturn:
            return ToolReturn(return_value="mot " * 5000, metadata=SOURCES)

    seen = []
    call = ToolCall(id="c1", name="search", arguments='{"query": "x"}')
    llm = _scripted_llm([_turn(tool_calls=[call]), _turn("ok")], seen)
    agent = Agent(llm=llm, tools=[LongTool()], max_tool_output_tokens=100)
    response = agent.run("go")

    tool_message = next(m for m in response.messages if m.role == "tool")
    assert "[Output truncated" in tool_message.content
    assert tool_message.extra[TOOL_METADATA_KEY] == SOURCES


@pytest.mark.parametrize("path", PATHS)
def test_tools_without_metadata_are_unchanged(path):
    seen = []
    call = ToolCall(id="c1", name="plain", arguments='{"query": "x"}')
    llm = _scripted_llm([_turn(tool_calls=[call]), _turn("done")], seen)
    response = _run(Agent(llm=llm, tools=[PlainTool()]), path)

    tool_message = next(m for m in response.messages if m.role == "tool")
    assert tool_message.content == '{"answer": "x"}'
    assert TOOL_METADATA_KEY not in tool_message.extra
    completed = next(e for e in response.events if e.event_type == "tool_completed")
    assert completed.tool_metadata is None


def test_failed_tool_has_no_metadata():
    @dataclass(kw_only=True)
    class FailingTool(BaseTool):
        name: str = "search"
        description: str = "Fails"
        max_retries: int = 1

        def _run(self, query: str) -> ToolReturn:
            raise RuntimeError("index unavailable")

    seen = []
    call = ToolCall(id="c1", name="search", arguments='{"query": "x"}')
    llm = _scripted_llm([_turn(tool_calls=[call]), _turn("sorry")], seen)
    response = Agent(llm=llm, tools=[FailingTool()]).run("go")
    tool_message = next(m for m in response.messages if m.role == "tool")
    assert tool_message.content.startswith("Error executing tool 'search'")
    assert TOOL_METADATA_KEY not in tool_message.extra


# ---------------------------------------------------------------------------
# Metadata stays with the application
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HandleTool(BaseTool):
    """Metadata the application alone can use: an object that can be neither copied nor serialized."""

    name: str = "search"
    description: str = "Searches"

    def _run(self, query: str) -> ToolReturn:
        return ToolReturn(return_value="hit", metadata={"lock": threading.Lock()})


@pytest.mark.parametrize("path", PATHS)
def test_metadata_is_left_out_of_the_model_request(path):
    _, _, seen = _search_run(path)
    tool_message = next(m for m in seen[1] if m["role"] == "tool")
    assert "extra" not in tool_message


@pytest.mark.parametrize("path", PATHS)
def test_metadata_that_cannot_be_copied_does_not_break_the_next_turn(path):
    seen = []
    call = ToolCall(id="c1", name="search", arguments='{"query": "x"}')
    llm = _scripted_llm([_turn(tool_calls=[call]), _turn("done")], seen)
    agent = Agent(llm=llm, tools=[HandleTool()])
    response = _run(agent, path)

    assert response.content == "done"
    ChatBase.input_to_message_list(llm, seen[1])  # what every client does with its input
    tool_message = next(m for m in agent.history.messages if m.role == "tool")
    assert "lock" in tool_message.extra[TOOL_METADATA_KEY]


def test_other_extra_keys_still_reach_the_client():
    from gwenflow.agents.agent import _model_messages

    message = Message(role="tool", tool_call_id="c1", content="x", extra={TOOL_METADATA_KEY: SOURCES, "pinned": True})
    assert _model_messages([message])[0]["extra"] == {"pinned": True}
    assert message.extra == {TOOL_METADATA_KEY: SOURCES, "pinned": True}


def test_clamping_a_long_result_does_not_copy_its_metadata():
    from gwenflow.memory import ChatMemoryBuffer

    history = ChatMemoryBuffer(token_limit=300)
    long_result = Message(
        role="tool", tool_call_id="c1", content="mot " * 2000, extra={TOOL_METADATA_KEY: {"lock": threading.Lock()}}
    )
    history.add_message(Message(role="user", content="q"))
    history.add_message(
        Message(
            role="assistant",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
        )
    )
    history.add_message(long_result)

    window = history.get()
    clamped = next(m for m in window if m.role == "tool")
    assert len(clamped.content) < len(long_result.content)
    assert long_result.content == "mot " * 2000  # the stored message is untouched


# ---------------------------------------------------------------------------
# Falsy results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, text", [(0, "0"), (False, "false"), ([], "[]"), ({}, "{}"), ("", "")])
def test_a_falsy_result_is_still_a_result(value, text):
    def lookup():
        """Looks something up."""
        return value

    assert Tool(lookup).run() == text
    assert (
        Tool(lambda: ToolReturn(return_value=value, metadata=1), name="t", description="d").run().return_value == text
    )


def test_no_result_is_still_empty():
    def nothing() -> None:
        """Returns nothing."""

    assert Tool(nothing).run() is None
