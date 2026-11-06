import click
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from gwenflow.registry import DEFAULT_MODEL, make_llm, make_tools
from gwenflow.registry import TOOLS as _TOOLS

console = Console()


def _make_llm(model: str):
    try:
        return make_llm(model)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--model") from e


def _make_tools(tool_names: tuple[str, ...]) -> list:
    try:
        return make_tools(tool_names)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--tools") from e


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
    from gwenflow.flows import FlowError, FlowRunner

    runner = FlowRunner(flow_file)
    if interactive:
        runner.plot_interactive()
    elif plot:
        runner.plot()
    else:
        try:
            results = runner.run()
        except FlowError as e:
            console.print(f"[red]{e}[/red]")
            raise SystemExit(1) from e
        for name in runner.flow.sinks():
            value = results[name]
            console.print(Markdown(value) if isinstance(value, str) else value)


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
