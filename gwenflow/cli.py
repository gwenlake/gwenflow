import asyncio

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
            # `arun`, not `run`: `run()` prints the last result itself, and the outputs are shown below.
            results = asyncio.run(runner.arun())
        except FlowError as e:
            console.print(f"[red]{e}[/red]")
            raise SystemExit(1) from e
        for name in runner.flow.sinks():
            value = results[name]
            console.print(Markdown(value) if isinstance(value, str) else value)


@main.command()
@click.argument("flow_file", type=click.Path(exists=True))
def check(flow_file):
    """Check a flow file and list the nodes that may run at the same time.

    Two nodes without a path between them run in parallel (unless the flow sets
    `settings.parallel: false`). A pair is flagged when both nodes declare the
    same file or SQLite database.
    """
    from gwenflow.flows import Flow, FlowError

    try:
        flow = Flow.from_file(flow_file)
        flow.validate()
    except FlowError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1) from e

    parallel = flow.settings.get("parallel", True)
    console.print(f"Flow '{flow.name}': {len(flow.nodes)} node(s), valid.")
    if not parallel:
        console.print("`settings.parallel` is false: nodes run one at a time, in this order:")
        console.print("  " + " -> ".join(flow.sequential_order()))
    report = flow.concurrent_report()
    if not report:
        console.print("No two nodes can run at the same time.")
        return
    click.echo("Nodes that may run at the same time:" if parallel else "Nodes that would run at the same time:")
    for pair in report:
        # One plain line per pair: paths are long, and the output is meant to be searched.
        shared = f"  <- both declare {', '.join(pair.shared_paths)}" if pair.shared_paths else ""
        click.echo(f"  {pair.first} | {pair.second}{shared}")
    conflicts = sum(1 for pair in report if pair.shared_paths)
    if conflicts:
        console.print(
            f"[yellow]{conflicts} pair(s) declare the same path: connect the two nodes, "
            "or set `settings.parallel: false`.[/yellow]"
        )


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


@main.group()
def harness():
    """Check harnesses (agents described by files)."""


@harness.command(name="check")
@click.argument("path", type=click.Path(exists=True))
def harness_check(path):
    """Load the harness directory or agent file (Pydantic AI spec or .md) at PATH and report what it holds."""
    from gwenflow.harness import Harness, HarnessError

    try:
        h = Harness.load(path)
    except HarnessError as e:
        console.print(f"[red]✗[/red] {e.message}")
        raise SystemExit(1) from e
    table = Table(title=f"Harness {h.name} (version {h.config.version})", show_header=False)
    table.add_column("Part", style="green")
    table.add_column("Content")
    table.add_row("model", h.config.model or "(given in code)")
    table.add_row("prompt", ", ".join(h.prompt_parts()) or "-")
    table.add_row(
        "skills", f"{', '.join(s.name for s in h.skills) or '-'} ({'inline' if h.skills_inline else 'on demand'})"
    )
    table.add_row("subagents", ", ".join(s.name for s in h.subagents) or "-")
    table.add_row("tools", ", ".join(t.name + (" (approval)" if t.requires_approval else "") for t in h.tools) or "-")
    table.add_row("mcp servers", ", ".join(c.id or c.url or c.command for c in h.mcp_servers) or "-")
    table.add_row("output schema", "yes" if h.config.output_schema is not None else "-")
    console.print(table)


@harness.command(name="schema")
def harness_schema():
    """Print the JSON Schema of harness.yaml (for editor completion)."""
    import json

    from gwenflow.harness import harness_json_schema

    click.echo(json.dumps(harness_json_schema(), indent=2))


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
