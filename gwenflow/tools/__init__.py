"""Built-in tools.

Only `BaseTool` and `Tool` are imported eagerly: `gwenflow.llms.base` needs
`Tool` to type an agent's toolset, so anything else imported here would be paid
for by every process that merely builds an LLM client.

The concrete tools are resolved lazily (PEP 562) because each drags its own
dependency -- the MCP servers alone pull `mcp`, and with it a whole Starlette /
uvicorn stack that a client-side deployment never runs. They stay importable
from this package as before; the dependency lands on first attribute access.
"""

import importlib
from typing import TYPE_CHECKING, Any

from gwenflow.tools.tool import BaseTool, Tool

if TYPE_CHECKING:
    from gwenflow.tools.clinicaltrials import ClinicalTrialsTool
    from gwenflow.tools.coding import (
        EditFileTool,
        FindTool,
        GrepTool,
        LsTool,
        ReadFileTool,
        WriteFileTool,
    )
    from gwenflow.tools.docker_code import DockerCodeTool
    from gwenflow.tools.duckduckgo import DuckDuckGoNewsTool, DuckDuckGoSearchTool
    from gwenflow.tools.local_file_system import LocalFileReadTool, LocalFileWriteTool
    from gwenflow.tools.mcp import (
        MCPServer,
        MCPServerSse,
        MCPServerSseParams,
        MCPServerStdio,
        MCPServerStdioParams,
    )
    from gwenflow.tools.pdf import PDFReaderTool
    from gwenflow.tools.pubmed import PubMedTool
    from gwenflow.tools.python import PythonCodeTool
    from gwenflow.tools.retriever import RetrieverTool
    from gwenflow.tools.shell import ShellTool
    from gwenflow.tools.tavily import TavilyWebSearchTool
    from gwenflow.tools.website import WebsiteReaderTool
    from gwenflow.tools.wikipedia import WikipediaTool
    from gwenflow.tools.yahoofinance import (
        YahooFinanceNews,
        YahooFinanceScreen,
        YahooFinanceStock,
    )

_LAZY_TOOLS = {
    "ClinicalTrialsTool": "gwenflow.tools.clinicaltrials",
    "DockerCodeTool": "gwenflow.tools.docker_code",
    "DuckDuckGoNewsTool": "gwenflow.tools.duckduckgo",
    "DuckDuckGoSearchTool": "gwenflow.tools.duckduckgo",
    "EditFileTool": "gwenflow.tools.coding",
    "FindTool": "gwenflow.tools.coding",
    "GrepTool": "gwenflow.tools.coding",
    "LocalFileReadTool": "gwenflow.tools.local_file_system",
    "LocalFileWriteTool": "gwenflow.tools.local_file_system",
    "LsTool": "gwenflow.tools.coding",
    "MCPServer": "gwenflow.tools.mcp",
    "MCPServerSse": "gwenflow.tools.mcp",
    "MCPServerSseParams": "gwenflow.tools.mcp",
    "MCPServerStdio": "gwenflow.tools.mcp",
    "MCPServerStdioParams": "gwenflow.tools.mcp",
    "PDFReaderTool": "gwenflow.tools.pdf",
    "PubMedTool": "gwenflow.tools.pubmed",
    "PythonCodeTool": "gwenflow.tools.python",
    "ReadFileTool": "gwenflow.tools.coding",
    "RetrieverTool": "gwenflow.tools.retriever",
    "ShellTool": "gwenflow.tools.shell",
    "TavilyWebSearchTool": "gwenflow.tools.tavily",
    "WebsiteReaderTool": "gwenflow.tools.website",
    "WikipediaTool": "gwenflow.tools.wikipedia",
    "WriteFileTool": "gwenflow.tools.coding",
    "YahooFinanceNews": "gwenflow.tools.yahoofinance",
    "YahooFinanceScreen": "gwenflow.tools.yahoofinance",
    "YahooFinanceStock": "gwenflow.tools.yahoofinance",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_TOOLS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache it so __getattr__ runs only once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_TOOLS))


__all__ = [
    "BaseTool",
    "Tool",
    "ShellTool",
    "PythonCodeTool",
    "DockerCodeTool",
    "RetrieverTool",
    "WikipediaTool",
    "WebsiteReaderTool",
    "PDFReaderTool",
    "PubMedTool",
    "ClinicalTrialsTool",
    "ReadFileTool",
    "EditFileTool",
    "WriteFileTool",
    "GrepTool",
    "FindTool",
    "LsTool",
    "LocalFileWriteTool",
    "LocalFileReadTool",
    "DuckDuckGoSearchTool",
    "DuckDuckGoNewsTool",
    "YahooFinanceNews",
    "YahooFinanceStock",
    "YahooFinanceScreen",
    "TavilyWebSearchTool",
    "MCPServer",
    "MCPServerSse",
    "MCPServerSseParams",
    "MCPServerStdio",
    "MCPServerStdioParams",
]
