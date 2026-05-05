import importlib

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

console = Console()

DEFAULT_MODEL = "openai/gpt-4o-mini"

_PROVIDERS = {
    "openai": ("gwenflow.llms.openai", "ChatOpenAI"),
    "anthropic": ("gwenflow.llms.anthropic", "ChatAnthropic"),
    "azure": ("gwenflow.llms.azure", "ChatAzureOpenAI"),
    "google": ("gwenflow.llms.google", "ChatGemini"),
    "gemini": ("gwenflow.llms.google", "ChatGemini"),
    "mistral": ("gwenflow.llms.mistral", "ChatMistral"),
    "ollama": ("gwenflow.llms.ollama", "ChatOllama"),
    "deepseek": ("gwenflow.llms.deepseek", "ChatDeepSeek"),
    "gwenlake": ("gwenflow.llms.gwenlake", "ChatGwenlake"),
}

_TOOLS = {
    "wikipedia": ("gwenflow.tools.wikipedia", "WikipediaTool"),
    "duckduckgo": ("gwenflow.tools.duckduckgo", "DuckDuckGoSearchTool"),
    "duckduckgo-news": ("gwenflow.tools.duckduckgo", "DuckDuckGoNewsTool"),
    "tavily": ("gwenflow.tools.tavily", "TavilyWebSearchTool"),
    "python": ("gwenflow.tools.python", "PythonCodeTool"),
    "shell": ("gwenflow.tools.shell", "ShellTool"),
    "website": ("gwenflow.tools.website", "WebsiteReaderTool"),
    "pdf": ("gwenflow.tools.pdf", "PDFReaderTool"),
    "yahoo-stock": ("gwenflow.tools.yahoofinance", "YahooFinanceStock"),
    "yahoo-news": ("gwenflow.tools.yahoofinance", "YahooFinanceNews"),
    "yahoo-screen": ("gwenflow.tools.yahoofinance", "YahooFinanceScreen"),
}


def _make_llm(model: str):
    provider, _, model_name = model.partition("/")
    if not model_name:
        model_name = provider
        provider = "openai"

    entry = _PROVIDERS.get(provider.lower())
    if entry is None:
        raise click.BadParameter(
            f"Unknown provider '{provider}'. Choose from: {', '.join(_PROVIDERS)}",
            param_hint="--model",
        )

    module_path, cls_name = entry
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls(model=model_name)


def _make_tools(tool_names: tuple[str, ...]) -> list:
    tools = []
    for name in tool_names:
        entry = _TOOLS.get(name.lower())
        if entry is None:
            raise click.BadParameter(
                f"Unknown tool '{name}'. Run 'gwenflow tools' to list available tools.",
                param_hint="--tools",
            )
        module_path, cls_name = entry
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        tools.append(cls())
    return tools


@click.group()
@click.version_option(package_name="gwenflow")
def main():
    """Gwenflow — orchestrate AI agents and workflows."""


@main.command()
@click.argument("flow_file", type=click.Path(exists=True))
@click.option("--plot", is_flag=True, help="Render a static PNG of the flow graph (requires pydot).")
@click.option("--interactive", is_flag=True, help="Open an interactive HTML flow graph (requires pyvis).")
def run(flow_file, plot, interactive):
    """Run a flow from a JSON or YAML file."""
    from gwenflow.flows import FlowRunner

    runner = FlowRunner(flow_file)
    if interactive:
        runner.plot_interactive()
    elif plot:
        runner.plot()
    else:
        runner.run()


@main.command()
@click.argument("query")
@click.option("--model", "-m", default=DEFAULT_MODEL, show_default=True, help="LLM to use (provider/model-name).")
@click.option(
    "--tools", "-t", multiple=True, metavar="TOOL", help="Built-in tool to equip the agent with (repeatable)."
)
@click.option(
    "--instructions",
    "-i",
    default="You are a helpful assistant.",
    show_default=True,
    help="System instructions for the agent.",
)
def ask(query, model, tools, instructions):
    """Run a single agent with QUERY."""
    from gwenflow.agents import Agent

    llm = _make_llm(model)
    agent = Agent(name="CLI Agent", llm=llm, tools=_make_tools(tools), instructions=[instructions])
    response = agent.run(query)
    console.print(Markdown(response.content))


@main.command()
@click.argument("query")
@click.option("--model", "-m", default=DEFAULT_MODEL, show_default=True, help="LLM to use (provider/model-name).")
@click.option("--tools", "-t", multiple=True, metavar="TOOL", help="Tools available to generated agents (repeatable).")
def autoflow(query, model, tools):
    """Auto-generate and run a multi-agent workflow for QUERY."""
    from gwenflow.flows import AutoFlow

    llm = _make_llm(model)
    flow = AutoFlow(llm=llm, tools=_make_tools(tools))
    result = flow.run(query)
    console.print(Markdown(result))


@main.command(name="tools")
def list_tools():
    """List available built-in tools."""
    table = Table(title="Available tools", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Class")
    for name, (_, cls) in sorted(_TOOLS.items()):
        table.add_row(name, cls)
    console.print(table)


if __name__ == "__main__":
    main()
