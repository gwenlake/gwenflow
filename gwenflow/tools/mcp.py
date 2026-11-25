from __future__ import annotations

import abc
import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import NotRequired, TypedDict

from gwenflow.tools.tool import BaseTool, _require_sync_context

if TYPE_CHECKING:
    # The `mcp` SDK drags a whole Starlette/uvicorn stack in (~14 MiB of RSS) that
    # a client-side deployment never runs. Annotations are strings here thanks to
    # `from __future__ import annotations`, so the SDK is only imported for real
    # where a connection is actually opened.
    from mcp import Client, ClientSession
    from mcp.client import Transport
    from mcp.types import CallToolResult
    from mcp.types import Tool as MCPToolDef


@dataclass(kw_only=True)
class _MCPTool(BaseTool):
    """A Tool wrapping a single MCP server tool call."""

    _mcp_name: str = ""
    _mcp_description: str = ""
    _mcp_parameters: dict = field(default_factory=dict)
    function: Callable = None

    def __post_init__(self) -> None:
        self.name = self._mcp_name
        self.description = self._mcp_description
        self.parameters = self._mcp_parameters
        self.function_schema = {
            "type": "function",
            "function": {
                "name": self._mcp_name,
                "description": self._mcp_description,
                "parameters": self._mcp_parameters,
            },
        }

    def run(self, **kwargs: Any) -> str:
        if inspect.iscoroutinefunction(self.function):
            _require_sync_context("tool.arun(...)")
        return super().run(**kwargs)

    def _run(self, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(self.function):
            return asyncio.run(self.function(**kwargs))
        return self.function(**kwargs)

    async def _arun(self, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(self.function):
            return await self.function(**kwargs)
        return await super()._arun(**kwargs)


def _result_to_str(result: CallToolResult) -> str:
    # Explicit JSON null is a valid structured result on modern MCP servers.
    has_structured = "structured_content" in result.model_fields_set
    blocks = [item.model_dump(mode="json") for item in result.content]
    if result.is_error:
        payload = {"isError": True, "content": blocks}
    elif not blocks:
        return json.dumps(result.structured_content, ensure_ascii=False) if has_structured else ""
    elif len(blocks) == 1:
        # Keep fields such as "text" directly accessible to existing callers.
        payload = blocks[0]
    elif has_structured:
        payload = {"content": blocks}
    else:
        return json.dumps(blocks)

    if has_structured:
        payload["structuredContent"] = result.structured_content
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


_Headers = Mapping[str, str]
_HeaderSource = _Headers | Callable[[], _Headers | Awaitable[_Headers]]


async def _resolve_headers(source: _HeaderSource | None) -> dict[str, str] | None:
    headers = source() if callable(source) else source
    if inspect.isawaitable(headers):
        headers = await headers
    return dict(headers) if headers is not None else None


class MCPServerSseParams(TypedDict):
    url: str
    headers: NotRequired[_HeaderSource]
    timeout: NotRequired[float]
    sse_read_timeout: NotRequired[float]


class MCPServerStreamableHttpParams(MCPServerSseParams):
    terminate_on_close: NotRequired[bool]


class MCPServerStdioParams(TypedDict):
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]
    cwd: NotRequired[str | Path]
    encoding: NotRequired[str]
    encoding_error_handler: NotRequired[Literal["strict", "ignore", "replace"]]


class MCPServer(abc.ABC):
    """Base class for MCP servers."""

    @abc.abstractmethod
    async def connect(self) -> None:
        pass

    @abc.abstractmethod
    async def cleanup(self) -> None:
        pass

    @abc.abstractmethod
    async def list_tools(self) -> list[MCPToolDef]:
        pass

    @abc.abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    def get_tools(self) -> list[BaseTool]:
        """Synchronously fetch all tools from the server."""
        _require_sync_context("server.aget_tools()")
        return asyncio.run(self.aget_tools())

    async def aget_tools(self) -> list[BaseTool]:
        """Fetch tool definitions in the caller's event loop."""
        await self.connect()
        try:
            return [self._as_tool(tool) for tool in await self.list_tools()]
        finally:
            await self.cleanup()

    @abc.abstractmethod
    def _as_tool(self, mcp_tool: MCPToolDef) -> BaseTool:
        pass


@dataclass
class _BoundSession:
    """Expose the raw session while routing tool operations through its SDK client when available."""

    session: ClientSession
    api: Client | ClientSession
    streamable_http: bool
    _listed_tools: set[str] = field(default_factory=set, init=False)

    async def list_tools(self, *, until: str | None = None) -> list[MCPToolDef]:
        from mcp.types import PaginatedRequestParams

        definitions: list[MCPToolDef] = []
        seen_cursors: set[str] = set()
        page = await self.session.list_tools()
        while True:
            definitions.extend(page.tools)
            if page.next_cursor is None or any(tool.name == until for tool in page.tools):
                break
            if page.next_cursor in seen_cursors:
                raise RuntimeError("MCP server repeated a tools/list pagination cursor")
            seen_cursors.add(page.next_cursor)
            page = await self.session.list_tools(params=PaginatedRequestParams(cursor=page.next_cursor))
        self._listed_tools = {tool.name for tool in definitions}
        return definitions

    async def call_tool(self, name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        from mcp.types.version import MODERN_PROTOCOL_VERSIONS

        if (
            self.streamable_http
            and arguments
            and name not in self._listed_tools
            and self.session.protocol_version in MODERN_PROTOCOL_VERSIONS
        ):
            # The SDK derives Mcp-Param-* headers from this authenticated session's listing.
            await self.list_tools(until=name)
        return await self.api.call_tool(name, arguments)


class _SessionMCPServer(MCPServer, abc.ABC):
    """Base for MCP servers that communicate via a ClientSession."""

    _connect_mode: Literal["auto", "legacy"] = "auto"
    _streamable_http = False

    def __init__(self) -> None:
        self._bound: ContextVar[_BoundSession | None] = ContextVar("mcp_session", default=None)
        self._exit_stack: ContextVar[AsyncExitStack | None] = ContextVar("mcp_exit_stack", default=None)

    @property
    def session(self) -> ClientSession | None:
        bound = self._bound.get()
        return bound.session if bound else None

    @session.setter
    def session(self, session: ClientSession | None) -> None:
        self._bound.set(_BoundSession(session, session, self._streamable_http) if session is not None else None)

    @abc.abstractmethod
    def _create_streams(self) -> Transport:
        pass

    @asynccontextmanager
    async def _open_client(self) -> AsyncIterator[_BoundSession]:
        from mcp import Client

        async with Client(self._create_streams(), mode=self._connect_mode, cache=None) as client:
            yield _BoundSession(client.session, client, self._streamable_http)

    async def aget_tools(self) -> list[BaseTool]:
        # Discovery and wrapped calls may overlap on a shared server configuration.
        # Each operation owns its transport, credentials and cleanup in the same task.
        async with self._open_client() as client:
            definitions = await client.list_tools()
        return [self._as_tool(tool) for tool in definitions]

    def _as_tool(self, mcp_tool: MCPToolDef) -> BaseTool:
        async def call(**kwargs: Any) -> str:
            async with self._open_client() as client:
                result = await client.call_tool(mcp_tool.name, kwargs or None)
            return _result_to_str(result)

        return _MCPTool(
            _mcp_name=mcp_tool.name,
            _mcp_description=mcp_tool.description or "",
            _mcp_parameters=mcp_tool.input_schema,
            function=call,
        )

    async def connect(self) -> None:
        """Open an explicit session for list_tools() and call_tool() in the current task."""
        stack = AsyncExitStack()
        client = await stack.enter_async_context(self._open_client())
        stack.callback(self._bound.reset, self._bound.set(client))
        stack.callback(self._exit_stack.reset, self._exit_stack.set(stack))

    async def cleanup(self) -> None:
        stack = self._exit_stack.get()
        if stack is not None:
            await stack.aclose()

    def _request_client(self) -> AbstractAsyncContextManager[_BoundSession]:
        bound = self._bound.get()
        if bound is None:
            raise RuntimeError(f"MCP server {self.name} not connected. Call connect() first.")
        return nullcontext(bound)

    async def list_tools(self) -> list[MCPToolDef]:
        async with self._request_client() as client:
            return await client.list_tools()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        async with self._request_client() as client:
            return await client.call_tool(tool_name, arguments)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.cleanup()


class _HttpMCPServer(_SessionMCPServer):
    params: MCPServerSseParams

    def _request_client(self) -> AbstractAsyncContextManager[_BoundSession]:
        # HTTP headers belong to the transport. Open one per operation when
        # the caller supplies credentials dynamically, including for raw calls.
        if callable(self.params.get("headers")):
            return self._open_client()
        return super()._request_client()


class MCPServerSse(_HttpMCPServer):
    """MCP server using HTTP + SSE transport."""

    _connect_mode = "legacy"

    def __init__(
        self,
        params: MCPServerSseParams,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.params = params
        self._name = name or f"sse:{params['url']}"

    @property
    def name(self) -> str:
        return self._name

    @asynccontextmanager
    async def _create_streams(self):
        import httpx2
        from mcp.client.sse import sse_client

        # Keep request-scoped credentials on the configured endpoint.
        options = {"httpx_client_factory": httpx2.AsyncClient} if callable(self.params.get("headers")) else {}
        async with sse_client(
            url=self.params["url"],
            headers=await _resolve_headers(self.params.get("headers")),
            timeout=self.params.get("timeout", 5),
            sse_read_timeout=self.params.get("sse_read_timeout", 300),
            **options,
        ) as streams:
            yield streams


class MCPServerStreamableHttp(_HttpMCPServer):
    """MCP server using the Streamable HTTP transport."""

    _streamable_http = True

    def __init__(self, params: MCPServerStreamableHttpParams, name: str | None = None) -> None:
        super().__init__()
        self.params = params
        self._name = name or f"streamable_http:{params['url']}"

    @property
    def name(self) -> str:
        return self._name

    @asynccontextmanager
    async def _create_streams(self):
        import httpx2
        from mcp.client.streamable_http import streamable_http_client

        timeout = httpx2.Timeout(self.params.get("timeout", 30), read=self.params.get("sse_read_timeout", 300))
        headers = await _resolve_headers(self.params.get("headers"))
        async with httpx2.AsyncClient(headers=headers, timeout=timeout) as client:
            async with streamable_http_client(
                self.params["url"], http_client=client, terminate_on_close=self.params.get("terminate_on_close", True)
            ) as streams:
                yield streams


class MCPServerStdio(_SessionMCPServer):
    """MCP server using stdio transport."""

    def __init__(
        self,
        params: MCPServerStdioParams,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.params = params
        self._name = name or f"stdio:{params['command']}"

    @property
    def name(self) -> str:
        return self._name

    def _create_streams(self):
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=self.params["command"],
            args=self.params.get("args", []),
            env=self.params.get("env", None),
            cwd=self.params.get("cwd", None),
            encoding=self.params.get("encoding", "utf-8"),
            encoding_error_handler=self.params.get("encoding_error_handler", "strict"),
        )
        return stdio_client(server_params)
