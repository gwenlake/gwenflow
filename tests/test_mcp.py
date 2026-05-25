"""Unit tests for MCP client integration.

We don't spin up a real MCP server; we mock the session/transport and verify:
- _MCPTool wrapping (name / description / parameters from the MCP definition)
- _as_tool produces a callable BaseTool that forwards args to the server
- get_tools() connects, lists, and disconnects
- _result_to_str handles single vs. multi-block responses
- TypedDict params accept the expected shapes
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import Tool as MCPToolDef

from gwenflow.tools.mcp import (
    MCPServer,
    MCPServerSse,
    MCPServerSseParams,
    MCPServerStdio,
    MCPServerStdioParams,
    _MCPTool,
    _result_to_str,
)


# ---------------------------------------------------------------------------
# _MCPTool — schema wiring
# ---------------------------------------------------------------------------


def test_mcp_tool_post_init_copies_fields_from_mcp_definition():
    tool = _MCPTool(
        _mcp_name="get_weather",
        _mcp_description="Get current weather for a city.",
        _mcp_parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        function=lambda **kw: f"weather-for-{kw['city']}",
    )
    assert tool.name == "get_weather"
    assert tool.description == "Get current weather for a city."
    assert tool.parameters["properties"]["city"]["type"] == "string"
    # OpenAI-style function_schema is built
    assert tool.function_schema["function"]["name"] == "get_weather"


def test_mcp_tool_run_forwards_to_underlying_function():
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return "ok"

    tool = _MCPTool(
        _mcp_name="echo",
        _mcp_description="Echo args.",
        _mcp_parameters={"type": "object"},
        function=fake,
    )
    out = tool.run(message="hi", n=3)
    assert out == "ok"
    assert captured == {"message": "hi", "n": 3}


# ---------------------------------------------------------------------------
# _result_to_str — translating CallToolResult
# ---------------------------------------------------------------------------


def test_result_to_str_empty_content():
    result = SimpleNamespace(content=[])
    assert _result_to_str(result) == ""


def test_result_to_str_single_block_uses_model_dump_json():
    block = SimpleNamespace(model_dump_json=lambda: '{"type":"text","text":"hi"}')
    result = SimpleNamespace(content=[block])
    assert _result_to_str(result) == '{"type":"text","text":"hi"}'


def test_result_to_str_multi_block_concats_as_json():
    block1 = SimpleNamespace(model_dump=lambda: {"type": "text", "text": "a"})
    block2 = SimpleNamespace(model_dump=lambda: {"type": "text", "text": "b"})
    result = SimpleNamespace(content=[block1, block2])
    out = _result_to_str(result)
    parsed = json.loads(out)
    assert parsed == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]


# ---------------------------------------------------------------------------
# MCPServerSse — construction and naming
# ---------------------------------------------------------------------------


def test_sse_server_default_name_from_url():
    server = MCPServerSse(MCPServerSseParams(url="https://example.com/mcp"))
    assert server.name == "sse:https://example.com/mcp"


def test_sse_server_explicit_name_wins():
    server = MCPServerSse(MCPServerSseParams(url="https://x/mcp"), name="my-server")
    assert server.name == "my-server"


def test_sse_server_as_tool_produces_mcp_tool_with_schema():
    server = MCPServerSse(MCPServerSseParams(url="https://x"))
    mcp_def = MCPToolDef(
        name="search",
        description="Search the web",
        inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    tool = server._as_tool(mcp_def)
    assert isinstance(tool, _MCPTool)
    assert tool.name == "search"
    assert tool.description == "Search the web"
    assert tool.parameters == {"type": "object", "properties": {"q": {"type": "string"}}}


def test_sse_server_as_tool_handles_missing_description():
    server = MCPServerSse(MCPServerSseParams(url="https://x"))
    mcp_def = MCPToolDef(name="t", description=None, inputSchema={"type": "object"})
    tool = server._as_tool(mcp_def)
    assert tool.description == ""


# ---------------------------------------------------------------------------
# MCPServerStdio — construction
# ---------------------------------------------------------------------------


def test_stdio_server_default_name_from_command():
    server = MCPServerStdio(MCPServerStdioParams(command="my-mcp-server"))
    assert "my-mcp-server" in server.name


def test_stdio_server_explicit_name():
    server = MCPServerStdio(MCPServerStdioParams(command="x"), name="custom")
    assert server.name == "custom"


# ---------------------------------------------------------------------------
# get_tools() lifecycle — mock the async list_tools, connect, cleanup
# ---------------------------------------------------------------------------


def _fake_mcp_def(name="get_weather"):
    return MCPToolDef(
        name=name,
        description=f"{name} description",
        inputSchema={"type": "object", "properties": {"city": {"type": "string"}}},
    )


def test_get_tools_calls_connect_list_cleanup_in_order(monkeypatch):
    server = MCPServerSse(MCPServerSseParams(url="https://x"))

    call_order = []

    async def fake_connect():
        call_order.append("connect")

    async def fake_list():
        call_order.append("list")
        return [_fake_mcp_def("get_weather"), _fake_mcp_def("get_news")]

    async def fake_cleanup():
        call_order.append("cleanup")

    monkeypatch.setattr(server, "connect", fake_connect)
    monkeypatch.setattr(server, "list_tools", fake_list)
    monkeypatch.setattr(server, "cleanup", fake_cleanup)

    tools = server.get_tools()
    assert call_order == ["connect", "list", "cleanup"]
    assert [t.name for t in tools] == ["get_weather", "get_news"]


def test_get_tools_cleanup_runs_even_if_list_fails(monkeypatch):
    server = MCPServerSse(MCPServerSseParams(url="https://x"))
    cleanup_called = []

    async def fake_connect():
        return None

    async def fake_list():
        raise RuntimeError("boom")

    async def fake_cleanup():
        cleanup_called.append(True)

    monkeypatch.setattr(server, "connect", fake_connect)
    monkeypatch.setattr(server, "list_tools", fake_list)
    monkeypatch.setattr(server, "cleanup", fake_cleanup)

    with pytest.raises(RuntimeError, match="boom"):
        server.get_tools()
    assert cleanup_called == [True]


# ---------------------------------------------------------------------------
# Session-bound methods reject calls when not connected
# ---------------------------------------------------------------------------


def test_list_tools_raises_when_not_connected():
    import asyncio

    server = MCPServerSse(MCPServerSseParams(url="https://x"))
    with pytest.raises(RuntimeError, match="not connected"):
        asyncio.run(server.list_tools())


def test_call_tool_raises_when_not_connected():
    import asyncio

    server = MCPServerSse(MCPServerSseParams(url="https://x"))
    with pytest.raises(RuntimeError, match="not connected"):
        asyncio.run(server.call_tool("any", {"x": 1}))


# ---------------------------------------------------------------------------
# Session is used when connected
# ---------------------------------------------------------------------------


def test_list_tools_delegates_to_session_when_connected():
    import asyncio

    server = MCPServerSse(MCPServerSseParams(url="https://x"))
    fake_session = MagicMock()
    fake_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[_fake_mcp_def("t1")]))
    server.session = fake_session

    tools = asyncio.run(server.list_tools())
    assert tools[0].name == "t1"
    fake_session.list_tools.assert_awaited_once()


def test_call_tool_delegates_to_session_when_connected():
    import asyncio

    server = MCPServerSse(MCPServerSseParams(url="https://x"))
    fake_session = MagicMock()
    fake_session.call_tool = AsyncMock(return_value="result")
    server.session = fake_session

    out = asyncio.run(server.call_tool("get_weather", {"city": "Paris"}))
    assert out == "result"
    fake_session.call_tool.assert_awaited_once_with("get_weather", {"city": "Paris"})


# ---------------------------------------------------------------------------
# Agent integration — MCPServer registered as a tool source
# ---------------------------------------------------------------------------


def test_agent_with_mcp_server_lazy_fetches_tools(monkeypatch):
    """Constructing an Agent with mcp_servers should call get_tools() on each
    server only when the agent actually needs the full tool list."""
    from gwenflow import Agent, ChatOpenAI

    server = MCPServerSse(MCPServerSseParams(url="https://x"))
    monkeypatch.setattr(server, "get_tools", lambda: [
        _MCPTool(
            _mcp_name="get_weather",
            _mcp_description="weather",
            _mcp_parameters={"type": "object"},
            function=lambda **kw: "sunny",
        )
    ])

    agent = Agent(name="test", llm=ChatOpenAI(api_key="test"), mcp_servers=[server])
    all_tools = agent.get_all_tools()
    assert any(t.name == "get_weather" for t in all_tools)
