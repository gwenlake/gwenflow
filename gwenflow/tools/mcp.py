from __future__ import annotations

import abc
import asyncio
import json
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from pathlib import Path
from typing import Any, Literal

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, JSONRPCMessage, Tool as MCPToolDef
from typing_extensions import NotRequired, TypedDict

from gwenflow.logger import logger
from gwenflow.tools.function import FunctionTool


def _result_to_str(result: CallToolResult) -> str:
    if not result.content:
        return ""
    if len(result.content) == 1:
        return result.content[0].model_dump_json()
    return json.dumps([item.model_dump() for item in result.content])


class MCPServerSseParams(TypedDict):
    url: str
    headers: NotRequired[dict[str, str]]
    timeout: NotRequired[float]
    sse_read_timeout: NotRequired[float]


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

    def get_tools(self) -> list[FunctionTool]:
        """Synchronously fetch all tools from the server as FunctionTool instances."""

        async def _list():
            await self.connect()
            try:
                return await self.list_tools()
            finally:
                await self.cleanup()

        mcp_tools = asyncio.run(_list())
        return [self._as_function_tool(t) for t in mcp_tools]

    @abc.abstractmethod
    def _as_function_tool(self, mcp_tool: MCPToolDef) -> FunctionTool:
        pass


class _SessionMCPServer(MCPServer, abc.ABC):
    """Base for MCP servers that communicate via a ClientSession."""

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock | None = None

    def _get_cleanup_lock(self) -> asyncio.Lock:
        if self._cleanup_lock is None:
            self._cleanup_lock = asyncio.Lock()
        return self._cleanup_lock

    @abc.abstractmethod
    def _create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        pass

    async def connect(self) -> None:
        try:
            transport = await self.exit_stack.enter_async_context(self._create_streams())
            read, write = transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
        except Exception as e:
            logger.error(f"Error connecting to MCP server {self.name}: {e}")
            await self.cleanup()
            raise

    async def cleanup(self) -> None:
        async with self._get_cleanup_lock():
            try:
                await self.exit_stack.aclose()
                self.session = None
            except Exception as e:
                logger.error(f"Error cleaning up MCP server {self.name}: {e}")

    async def list_tools(self) -> list[MCPToolDef]:
        if not self.session:
            raise RuntimeError(f"MCP server {self.name} not connected. Call connect() first.")
        return (await self.session.list_tools()).tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        if not self.session:
            raise RuntimeError(f"MCP server {self.name} not connected. Call connect() first.")
        return await self.session.call_tool(tool_name, arguments)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.cleanup()


class MCPServerSse(_SessionMCPServer):
    """MCP server using HTTP + SSE transport."""

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

    def _create_streams(self):
        return sse_client(
            url=self.params["url"],
            headers=self.params.get("headers", None),
            timeout=self.params.get("timeout", 5),
            sse_read_timeout=self.params.get("sse_read_timeout", 300),
        )

    def _as_function_tool(self, mcp_tool: MCPToolDef) -> FunctionTool:
        params = self.params
        tool_name = mcp_tool.name

        def _run(**kwargs: Any) -> str:
            async def _call():
                server = MCPServerSse(params)
                async with server:
                    result = await server.call_tool(tool_name, kwargs or None)
                    return _result_to_str(result)

            return asyncio.run(_call())

        return FunctionTool(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            parameters=mcp_tool.inputSchema,
            func=_run,
            tool_type="function",
        )


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

    def _as_function_tool(self, mcp_tool: MCPToolDef) -> FunctionTool:
        params = self.params
        tool_name = mcp_tool.name

        def _run(**kwargs: Any) -> str:
            async def _call():
                server = MCPServerStdio(params)
                async with server:
                    result = await server.call_tool(tool_name, kwargs or None)
                    return _result_to_str(result)

            return asyncio.run(_call())

        return FunctionTool(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            parameters=mcp_tool.inputSchema,
            func=_run,
            tool_type="function",
        )
