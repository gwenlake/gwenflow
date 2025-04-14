import context
import requests
import json
import asyncio

from gwenflow import ChatOpenAI, Agent
# from gwenflow.tools import WebsiteReaderTool, PDFTool, DuckDuckGoSearchTool, DuckDuckGoNewsTool, TavilyWebSearchTool
# from gwenflow import set_log_level_to_debug
from gwenflow.tools.mcp import MCPServerSse, MCPServerSseParams

# from agents import Agent
# from agents.mcp import MCPServerSse

# set_log_level_to_debug()

async def run_agent_with_remote_mcp():
    async with MCPServerSse(
            name="test",
            params={
                "url": "http://localhost:8000/sse",
            #     "headers": {
            #         "Authorization": "Bearer your_api_key_here"
            #     }
            },
        ) as _server:

        agent = Agent(
            name="Assistant",
            instructions="Use the filesystem tools to help the user with their tasks.",
            tool_choice="required",
            mcp_servers=[_server]
        )

        # Run the agent
        result = agent.arun("List the files in the directory.")
        print(result.content)

if __name__ == "__main__":
    asyncio.run(run_agent_with_remote_mcp())
