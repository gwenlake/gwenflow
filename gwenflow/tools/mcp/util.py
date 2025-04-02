import functools
import json
from typing import TYPE_CHECKING, Any

from gwenflow.exceptions import GwenflowException, UserError, ModelBehaviorError
from gwenflow.logger import logger
from gwenflow.tools import FunctionTool, BaseTool
from gwenflow.tools.mcp.server import MCPServer

from mcp.types import Tool as MCPTool


class MCPUtil:
    """Set of utilities for interop between MCP and Gwenflow tools."""

    @classmethod
    async def get_all_function_tools(cls, servers: list["MCPServer"]) -> list[BaseTool]:
        """Get all function tools from a list of MCP servers."""
        tools = []
        tool_names: set[str] = set()
        for server in servers:
            server_tools = await cls.get_function_tools(server)
            server_tool_names = {tool.name for tool in server_tools}
            if len(server_tool_names & tool_names) > 0:
                raise UserError(
                    f"Duplicate tool names found across MCP servers: "
                    f"{server_tool_names & tool_names}"
                )
            tool_names.update(server_tool_names)
            tools.extend(server_tools)

        return tools

    @classmethod
    async def get_function_tools(cls, server: "MCPServer") -> list[BaseTool]:
        """Get all function tools from a single MCP server."""
        tools = await server.list_tools()
        return [cls.to_function_tool(tool, server) for tool in tools]

    @classmethod
    def to_function_tool(cls, tool: "MCPTool", server: "MCPServer") -> FunctionTool:
        """Convert an MCP tool to a function tool."""
        return FunctionTool(
            name=tool.name,
            description=tool.description or "",
            params_json_schema=tool.inputSchema,
            func=functools.partial(cls.invoke_mcp_tool, server, tool),
            tool_type="function",
        )

    @classmethod
    async def invoke_mcp_tool(cls, server: "MCPServer", tool: "MCPTool", input_json: str) -> str:
        """Invoke an MCP tool and return the result as a string."""
        try:
            json_data: dict[str, Any] = json.loads(input_json) if input_json else {}
        except Exception as e:
            logger.debug(f"Invalid JSON input for tool {tool.name}: {input_json}")
            raise ModelBehaviorError(
                f"Invalid JSON input for tool {tool.name}: {input_json}"
            ) from e

        logger.debug(f"Invoking MCP tool {tool.name} with input {input_json}")

        try:
            result = await server.call_tool(tool.name, json_data)
        except Exception as e:
            logger.error(f"Error invoking MCP tool {tool.name}: {e}")
            raise GwenflowException(f"Error invoking MCP tool {tool.name}: {e}") from e

        logger.debug(f"MCP tool {tool.name} returned {result}")

        # The MCP tool result is a list of content items, whereas OpenAI tool outputs are a single
        # string. We'll try to convert.
        if len(result.content) == 1:
            tool_output = result.content[0].model_dump_json()
        elif len(result.content) > 1:
            tool_output = json.dumps([item.model_dump() for item in result.content])
        else:
            logger.error(f"Errored MCP tool result: {result}")
            tool_output = "Error running tool."

        return tool_output