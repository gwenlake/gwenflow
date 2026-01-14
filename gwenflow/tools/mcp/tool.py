from typing import Any, Optional
from pydantic import model_validator


from gwenflow.tools.base import BaseTool
from gwenflow.tools.mcp.server import MCPServerSse, MCPServerSseParams


class MCPTool(BaseTool):
    def _run(self, **kwargs: Any) -> Any:
        params = MCPServerSseParams(url=self.url, headers=self.headers)
        server = MCPServerSse(params=params)
        return server.call_tool(tool=self.tool, **kwargs)

    @classmethod
    async def from_server(cls, url: str, name: str, headers: dict = None) -> "MCPTool":
        params = MCPServerSseParams(url=url, headers=headers)
        server = MCPServerSse(params=params)
        await server.connect()

        tool = None
        response = await server.list_tools()
        for t in response.tools:
            if t.name == name:
                tool = t
                break

        if not tool:
            raise ValueError("Tool not found not in MCP.")

        return MCPTool(
            name=tool.__name__,
            description=tool.description,
            params_json_schema=tool.inputSchema,
            tool_type="mcp",
        )
