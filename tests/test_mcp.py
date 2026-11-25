"""MCP client tests using mocked SSE sessions and an in-process Streamable HTTP server."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as MCPToolDef

from gwenflow.tools.mcp import (
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
    result = CallToolResult(content=[])
    assert _result_to_str(result) == ""


def test_result_to_str_single_block_uses_model_dump_json():
    block = TextContent(type="text", text="hi")
    result = CallToolResult(content=[block])
    assert _result_to_str(result) == block.model_dump_json()


def test_result_to_str_multi_block_concats_as_json():
    blocks = [TextContent(type="text", text=text) for text in ("a", "b")]
    result = CallToolResult(content=blocks)
    out = _result_to_str(result)
    parsed = json.loads(out)
    assert parsed == [block.model_dump() for block in blocks]


@pytest.mark.parametrize("value", [{"remaining": 17}, 0, False, [], {}, "", None])
def test_mcp_structured_only_result_preserves_json_values(value):
    result = CallToolResult(content=[], structured_content=value)
    assert json.loads(_result_to_str(result)) == value


@pytest.mark.parametrize("texts", [["summary"], ["summary", "details"]])
def test_mcp_mixed_result_preserves_blocks_and_structured_content(texts):
    blocks = [TextContent(type="text", text=text) for text in texts]
    result = CallToolResult(content=blocks, structured_content={"remaining": 17})
    payload = json.loads(_result_to_str(result))
    assert payload["structuredContent"] == {"remaining": 17}
    assert [block["text"] for block in payload.get("content", [payload])] == texts


def test_mcp_tool_error_reaches_agent_once(sse_endpoint):
    from gwenflow import Agent, ChatOpenAI
    from gwenflow.types import ToolCall

    session, _ = sse_endpoint
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="Unavailable")], is_error=True, structured_content={"code": 42}
    )
    server = MCPServerSse({"url": "https://example.test"})
    agent = Agent(llm=ChatOpenAI(api_key="test"), mcp_servers=[server])
    message = asyncio.run(agent.arun_tool(ToolCall(id="1", name="get_weather", arguments="{}")))
    error = json.loads(message.content)
    assert error["isError"] is True
    assert error["content"][0]["text"] == "Unavailable"
    assert error["structuredContent"] == {"code": 42}
    session.call_tool.assert_awaited_once()


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


def test_stdio_discovers_current_protocol_and_calls_local_server(tmp_path):
    script = tmp_path / "stdio_server.py"
    script.write_text(
        """from mcp.server import MCPServer

server = MCPServer("test")

@server.tool()
def echo(message: str) -> str:
    return message

server.run(transport="stdio")
""",
        encoding="utf-8",
    )
    server = MCPServerStdio({"command": sys.executable, "args": [str(script)]})

    async def run():
        async with asyncio.timeout(15), server:
            assert server.session.protocol_version == "2026-07-28"
            (definition,) = await server.list_tools()
            result = await server.call_tool(definition.name, {"message": "stdio"})
            assert result.content[0].text == "stdio"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# get_tools() lifecycle — mock the async list_tools, connect, cleanup
# ---------------------------------------------------------------------------


def _fake_mcp_def(name="get_weather"):
    return MCPToolDef(
        name=name,
        description=f"{name} description",
        inputSchema={"type": "object", "properties": {"city": {"type": "string"}}},
    )


def _mock_legacy_client(monkeypatch, new_session):
    @asynccontextmanager
    async def client(transport, *, mode, **kwargs):
        assert mode == "legacy"
        async with transport:
            session = new_session()
            yield SimpleNamespace(session=session, list_tools=session.list_tools, call_tool=session.call_tool)

    monkeypatch.setattr("mcp.Client", client)


@pytest.fixture
def sse_endpoint(monkeypatch):
    events = []
    session = AsyncMock()
    session.list_tools.return_value = ListToolsResult(tools=[_fake_mcp_def("get_weather")])
    session.call_tool.return_value = CallToolResult(content=[])

    @asynccontextmanager
    async def streams(url, headers=None, **kwargs):
        events.append(("connect", headers))
        try:
            yield object(), object()
        finally:
            events.append(("cleanup", headers))

    monkeypatch.setattr("mcp.client.sse.sse_client", streams)
    _mock_legacy_client(monkeypatch, lambda: session)
    return session, events


def test_get_tools_and_run_resolve_sse_headers_for_each_connection(sse_endpoint):
    session, events = sse_endpoint
    tokens = iter(["Bearer discovery", "Bearer tool-call"])
    server = MCPServerSse({"url": "https://x", "headers": lambda: {"Authorization": next(tokens)}})

    (tool,) = server.get_tools()
    assert tool.name == "get_weather"
    tool.run(city="Paris")

    assert events == [
        ("connect", {"Authorization": "Bearer discovery"}),
        ("cleanup", {"Authorization": "Bearer discovery"}),
        ("connect", {"Authorization": "Bearer tool-call"}),
        ("cleanup", {"Authorization": "Bearer tool-call"}),
    ]
    session.list_tools.assert_awaited_once()
    session.call_tool.assert_awaited_once_with("get_weather", {"city": "Paris"})


def test_get_tools_cleanup_runs_even_if_list_fails(sse_endpoint):
    session, events = sse_endpoint
    session.list_tools.side_effect = RuntimeError("boom")
    server = MCPServerSse({"url": "https://x", "headers": {"X-App": "test"}})

    with pytest.raises(RuntimeError, match="boom"):
        server.get_tools()
    assert events == [("connect", {"X-App": "test"}), ("cleanup", {"X-App": "test"})]


# ---------------------------------------------------------------------------
# Session-bound methods reject calls when not connected
# ---------------------------------------------------------------------------


def test_list_tools_raises_when_not_connected():
    server = MCPServerSse(MCPServerSseParams(url="https://x"))
    with pytest.raises(RuntimeError, match="not connected"):
        asyncio.run(server.list_tools())


def test_call_tool_raises_when_not_connected():
    server = MCPServerSse(MCPServerSseParams(url="https://x"))
    with pytest.raises(RuntimeError, match="not connected"):
        asyncio.run(server.call_tool("any", {"x": 1}))


# ---------------------------------------------------------------------------
# Session is used when connected
# ---------------------------------------------------------------------------


def test_explicit_session_reuses_connection_with_static_headers(sse_endpoint):
    session, events = sse_endpoint
    headers = {"Authorization": "Bearer user"}
    server = MCPServerSse({"url": "https://x", "headers": headers})

    async def run():
        async with server:
            assert server.session is session
            (definition,) = await server.list_tools()
            await server.call_tool(definition.name, {"city": "Paris"})
        assert server.session is None

    asyncio.run(run())
    assert events == [("connect", headers), ("cleanup", headers)]
    session.call_tool.assert_awaited_once_with("get_weather", {"city": "Paris"})


def test_simultaneous_explicit_sessions_are_task_local(sse_endpoint, monkeypatch):
    _mock_legacy_client(monkeypatch, AsyncMock)
    server = MCPServerSse({"url": "https://x"})

    async def run():
        barrier = asyncio.Barrier(2)

        async def use_session():
            async with server:
                session = server.session
                await barrier.wait()
                assert server.session is session
                await server.call_tool("echo", {})
                session.call_tool.assert_awaited_once_with("echo", {})
            assert server.session is None
            return session

        first, second = await asyncio.gather(use_session(), use_session())
        assert first is not second

    asyncio.run(run())


def test_raw_calls_refresh_dynamic_headers_inside_explicit_session(sse_endpoint):
    _, events = sse_endpoint
    tokens = iter(["scope", "list", "first-call", "second-call"])
    server = MCPServerSse({"url": "https://x", "headers": lambda: {"Authorization": next(tokens)}})

    async def run():
        async with server:
            (definition,) = await server.list_tools()
            await server.call_tool(definition.name, {})
            await server.call_tool(definition.name, {})

    asyncio.run(run())
    assert [headers["Authorization"] for event, headers in events if event == "connect"] == [
        "scope",
        "list",
        "first-call",
        "second-call",
    ]


# ---------------------------------------------------------------------------
# Agent integration — MCPServer registered as a tool source
# ---------------------------------------------------------------------------


def test_agent_includes_tools_from_mcp_servers(monkeypatch):
    from gwenflow import Agent, ChatOpenAI

    server = MCPServerSse(MCPServerSseParams(url="https://x"))
    monkeypatch.setattr(
        server,
        "get_tools",
        lambda: [
            _MCPTool(
                _mcp_name="get_weather",
                _mcp_description="weather",
                _mcp_parameters={"type": "object"},
                function=lambda **kw: "sunny",
            )
        ],
    )

    agent = Agent(name="test", llm=ChatOpenAI(api_key="test"), mcp_servers=[server])
    all_tools = agent.get_all_tools()
    assert any(t.name == "get_weather" for t in all_tools)


@pytest.mark.parametrize("json_response", [True, False], ids=["json", "event-stream"])
def test_streamable_http_isolates_concurrent_discovery_and_tool_calls(monkeypatch, json_response):
    from mcp.server.mcpserver import MCPServer as SDKServer

    from gwenflow.tools import MCPServerStreamableHttp

    backend = SDKServer("test")

    @backend.tool()
    def echo(value: str) -> str:
        return value

    app = backend.streamable_http_app(stateless_http=True, json_response=json_response)
    transport = httpx2.ASGITransport(app=app)
    requests = []
    versions = []

    async def dispatch(self, request):
        body = json.loads(await request.aread())
        requests.append((body, request.headers["Authorization"]))
        metadata = body.get("params", {}).get("_meta", {})
        versions.append(
            (request.headers.get("MCP-Protocol-Version"), metadata.get("io.modelcontextprotocol/protocolVersion"))
        )
        return await transport.handle_async_request(request)

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", dispatch)
    token = ContextVar("token")

    async def headers():
        await asyncio.sleep(0)
        return {"Authorization": token.get()}

    server = MCPServerStreamableHttp({"url": "http://127.0.0.1:9000/mcp", "headers": headers})

    async def discover(user):
        token.set(user)
        (tool,) = await server.aget_tools()
        return tool

    async def call(tool, user):
        token.set(user)
        result = await tool.arun(value=user)
        assert json.loads(result)["text"] == user

    async def run():
        async with asyncio.timeout(10), app.router.lifespan_context(app):
            alice, bob = await asyncio.gather(discover("alice"), discover("bob"))
            assert alice.name == bob.name == "echo"
            assert sorted(auth for body, auth in requests if body["method"] == "tools/list") == ["alice", "bob"]
            await asyncio.gather(call(alice, "alice-renewed"), call(alice, "bob-renewed"))

    asyncio.run(run())
    assert {body["method"] for body, _ in requests} == {"server/discover", "tools/list", "tools/call"}
    assert set(versions) == {("2026-07-28", "2026-07-28")}
    calls = [(body["params"]["arguments"], auth) for body, auth in requests if body["method"] == "tools/call"]
    assert len(calls) == 2
    for arguments, auth in calls:
        assert arguments == {"value": auth}


@pytest.mark.parametrize("wrapped", [False, True], ids=["call-tool", "discovered-tool"])
def test_streamable_http_sends_headers_from_the_callers_paginated_schema(monkeypatch, wrapped):
    from mcp.server import Server
    from mcp.types import CallToolResult, TextContent

    from gwenflow.tools import MCPServerStreamableHttp

    async def list_tools(ctx, params):
        if not params or params.cursor is None:
            return ListToolsResult(tools=[], next_cursor="second")
        assert params.cursor == "second"
        user = ctx.request.headers["Authorization"].removeprefix("Bearer ")
        schema = {"type": "object", "properties": {"value": {"type": "string", "x-mcp-header": user}}}
        return ListToolsResult(tools=[MCPToolDef(name="echo", input_schema=schema)])

    async def echo(ctx, params):
        return CallToolResult(content=[TextContent(type="text", text=params.arguments["value"])])

    app = Server("test", on_list_tools=list_tools, on_call_tool=echo).streamable_http_app(
        stateless_http=True, json_response=True
    )
    transport = httpx2.ASGITransport(app=app)
    attempts = []

    async def dispatch(self, request):
        body = json.loads(await request.aread())
        response = await transport.handle_async_request(request)
        if body["method"] == "tools/call":
            headers = {k: v for k, v in request.headers.items() if k.startswith("mcp-param-")}
            attempts.append((request.headers["Authorization"], headers, response.status_code))
        return response

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", dispatch)
    user = ContextVar("user", default="alice")
    server = MCPServerStreamableHttp(
        {"url": "http://127.0.0.1:9000/mcp", "headers": lambda: {"Authorization": f"Bearer {user.get()}"}}
    )

    async def run():
        async with asyncio.timeout(10), app.router.lifespan_context(app), server:
            if wrapped:
                (tool,) = await server.aget_tools()
            for caller in ("alice", "bob"):
                user.set(caller)
                if wrapped:
                    assert json.loads(await tool.arun(value=caller))["text"] == caller
                else:
                    result = await server.call_tool("echo", {"value": caller})
                    assert result.content[0].text == caller

    asyncio.run(run())
    assert attempts == [(f"Bearer {caller}", {f"mcp-param-{caller}": caller}, 200) for caller in ("alice", "bob")]


def test_streamable_http_falls_back_when_old_server_has_no_discover(monkeypatch):
    from mcp.server.mcpserver import MCPServer as SDKServer

    from gwenflow.tools import MCPServerStreamableHttp

    backend = SDKServer("legacy")

    @backend.tool()
    def ping() -> str:
        return "pong"

    app = backend.streamable_http_app(stateless_http=True, json_response=True)
    transport = httpx2.ASGITransport(app=app)
    calls = []

    async def dispatch(self, request):
        body = json.loads(await request.aread())
        method = body["method"]
        calls.append((method, request.headers.get("MCP-Protocol-Version")))
        if method == "server/discover":
            return httpx2.Response(
                404, json={"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32601, "message": "Method not found"}}
            )
        return await transport.handle_async_request(request)

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", dispatch)
    server = MCPServerStreamableHttp({"url": "http://127.0.0.1:9000/mcp"})

    async def run():
        async with asyncio.timeout(10), app.router.lifespan_context(app), server:
            (tool,) = await server.list_tools()
            result = await server.call_tool(tool.name, {})
            assert result.content[0].text == "pong"

    asyncio.run(run())
    assert calls[0] == ("server/discover", "2026-07-28")
    assert calls[1][0] == "initialize"
    assert ("tools/list", "2025-11-25") in calls
    assert ("tools/call", "2025-11-25") in calls


@pytest.mark.parametrize("disjoint", [False, True], ids=["access-denied", "disjoint-versions"])
def test_streamable_http_propagates_negotiation_failure(monkeypatch, disjoint):
    from gwenflow.tools import MCPServerStreamableHttp

    methods = []
    if disjoint:
        error = {
            "code": -32022,
            "message": "Unsupported protocol version",
            "data": {"supported": ["2099-01-01"], "requested": "2026-07-28"},
        }
    else:
        error = {"code": -32603, "message": "Access denied"}

    async def dispatch(self, request):
        body = json.loads(await request.aread())
        methods.append(body["method"])
        return httpx2.Response(400 if disjoint else 403, json={"jsonrpc": "2.0", "id": body["id"], "error": error})

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", dispatch)
    server = MCPServerStreamableHttp({"url": "http://127.0.0.1:9000/mcp"})
    with pytest.raises(ExceptionGroup) as exc:
        asyncio.run(server.aget_tools())
    assert exc.group_contains(MCPError, match=error["message"])
    assert methods[0] == "server/discover"
    assert set(methods) <= ({"server/discover"} if disjoint else {"server/discover", "initialize"})


@pytest.mark.parametrize("stream", [False, True], ids=["arun", "arun_stream"])
def test_async_agent_discovers_and_calls_mcp_tools(sse_endpoint, stream):
    from gwenflow import Agent, ChatOpenAI
    from gwenflow.types import ModelResponse, TextContent, ToolCall

    session, _ = sse_endpoint
    server = MCPServerSse({"url": "https://x"})
    invocation = ToolCall(id="c1", name="get_weather", arguments='{"city":"Paris"}')

    async def run():
        agent = Agent(llm=ChatOpenAI(api_key="test"), mcp_servers=[server])
        agent.history.tokenizer_fn = len
        turns = iter(
            [
                ModelResponse(parts=[invocation], finish_reason="tool_calls"),
                ModelResponse(parts=[TextContent(content="done")], finish_reason="stop"),
            ]
        )

        async def reply(input):
            assert any(tool.name == "get_weather" for tool in agent.llm.tools)
            return next(turns)

        async def chunks(input):
            yield await reply(input)

        if stream:
            agent.llm.astream = chunks
            events = [event async for event in agent.arun_stream("Weather in Paris?")]
            result = events[-1]
        else:
            agent.llm.ainvoke = reply
            result = await agent.arun("Weather in Paris?")
        assert result.content == "done"
        session.list_tools.assert_awaited_once()
        session.call_tool.assert_awaited_once_with("get_weather", {"city": "Paris"})

    asyncio.run(run())
