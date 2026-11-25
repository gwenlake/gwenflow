"""Exercise an async FastAPI route against a Streamable HTTP MCP server in memory."""

import asyncio
import json
from collections import Counter
from contextvars import ContextVar
from typing import Annotated
from unittest.mock import AsyncMock

import httpx
import httpx2
from fastapi import FastAPI
from mcp.server.mcpserver import Context
from mcp.server.mcpserver import MCPServer as SDKServer
from pydantic import Field

from gwenflow import Agent, ChatOpenAI
from gwenflow.tools import MCPServerStreamableHttp
from gwenflow.types import ModelResponse, TextContent, ToolCall


def test_concurrent_fastapi_requests_refresh_credentials_for_each_mcp_call(monkeypatch):
    backend = SDKServer("test")
    received = []

    @backend.tool()
    def echo(label: Annotated[str, Field(json_schema_extra={"x-mcp-header": "label"})], ctx: Context) -> str:
        received.append((label, ctx.request_context.request.headers))
        return label

    backend_app = backend.streamable_http_app(stateless_http=True, json_response=True)
    transport = httpx2.ASGITransport(app=backend_app)
    discovery_tokens = []
    attempts = []

    async def dispatch(self, request):
        body = json.loads(await request.aread())
        response = await transport.handle_async_request(request)
        if body["method"] == "tools/list":
            discovery_tokens.append(request.headers["Authorization"])
        if body["method"] == "tools/call":
            label = body["params"]["arguments"]["label"]
            attempts.append((label, request.headers.get("Mcp-Param-Label"), response.status_code))
        return response

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", dispatch)
    current_user = ContextVar("current_user")
    issued = Counter()
    both_discovering = asyncio.Barrier(2)

    async def headers():
        user = current_user.get()
        issued[user] += 1
        sequence = issued[user]
        if sequence == 1:
            await both_discovering.wait()
        return {"Authorization": f"Bearer {user}-{sequence}"}

    server = MCPServerStreamableHttp({"url": "http://127.0.0.1:9000/mcp", "headers": headers})
    api = FastAPI()

    @api.get("/run/{user}")
    async def run_agent(user: str):
        token = current_user.set(user)
        try:
            agent = Agent(llm=ChatOpenAI(api_key="test"), mcp_servers=[server])
            agent.history.tokenizer_fn = len
            calls = [ToolCall(id=f"c{i}", name="echo", arguments=json.dumps({"label": f"{user}/{i}"})) for i in (1, 2)]
            agent.llm.ainvoke = AsyncMock(
                side_effect=[
                    ModelResponse(parts=calls, finish_reason="tool_calls"),
                    ModelResponse(parts=[TextContent(content="done")], finish_reason="stop"),
                ]
            )
            await agent.arun("Call echo twice")
            return [json.loads(msg.content)["text"] for msg in agent.history.get() if msg.role == "tool"]
        finally:
            current_user.reset(token)

    async def run():
        async with (
            asyncio.timeout(10),
            backend_app.router.lifespan_context(backend_app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://testserver") as client,
        ):
            return await asyncio.gather(client.get("/run/alice"), client.get("/run/bob"))

    responses = asyncio.run(run())
    assert Counter(attempts) == Counter((label, label, 200) for label in ("alice/1", "alice/2", "bob/1", "bob/2"))
    assert len(received) == 4
    assert {headers["MCP-Protocol-Version"] for _, headers in received} == {"2026-07-28"}
    assert all(headers["Mcp-Param-Label"] == label for label, headers in received)
    for user, response in zip(("alice", "bob"), responses, strict=True):
        response.raise_for_status()
        assert sorted(response.json()) == [f"{user}/1", f"{user}/2"]
        assert issued[user] == 3
        assert f"Bearer {user}-1" in discovery_tokens
        assert sorted(headers["Authorization"] for label, headers in received if label.startswith(f"{user}/")) == [
            f"Bearer {user}-2",
            f"Bearer {user}-3",
        ]
