from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from gwenflow.agents.agent import Agent
from gwenflow.tools import (
    BaseTool,
    EditFileTool,
    FindTool,
    GrepTool,
    LocalFileWriteTool,
    LsTool,
    ReadFileTool,
    ShellTool,
    WebsiteReaderTool,
    WriteFileTool,
)

CODING_AGENT_INSTRUCTIONS = """\
You are a coding agent operating inside a single working directory (base_dir).

Workflow:
- Use LsTool / FindTool to discover files; GrepTool to locate symbols or patterns.
- ALWAYS call ReadFileTool on a file before editing it.
- Use EditFileTool for targeted changes (it requires an exact, unique match). \
For new files or full rewrites use WriteFileTool.
- Use ShellTool to run tests, linters, git commands, or package managers. Keep \
commands single-line and quote paths.
- Use LocalFileWriteTool when the user asks you to save an artifact (a report, \
summary, generated dataset) somewhere outside the code tree.
- Use WebsiteReaderTool to consult external documentation when needed.

Principles:
- Make small, verifiable changes; run tests after edits.
- Prefer editing existing files to creating new ones.
- Never invent file paths — list or search first.
- Stop and report when the task is done or you are blocked.\
"""


def build_coding_tools(
    base_dir: Optional[Union[Path, str]] = None,
    target_directory: Optional[Union[Path, str]] = None,
    include_shell: bool = True,
    include_web_reader: bool = True,
    include_local_file_writer: bool = True,
    restrict_to_base_dir: bool = True,
) -> list[BaseTool]:
    """Return a default tool bundle for a coding agent.

    base_dir scopes all file and shell operations. target_directory (defaults
    to base_dir) is where LocalFileWriteTool drops generated artifacts.
    """
    tools: list[BaseTool] = [
        ReadFileTool(base_dir=base_dir, restrict_to_base_dir=restrict_to_base_dir),
        EditFileTool(base_dir=base_dir, restrict_to_base_dir=restrict_to_base_dir),
        WriteFileTool(base_dir=base_dir, restrict_to_base_dir=restrict_to_base_dir),
        GrepTool(base_dir=base_dir, restrict_to_base_dir=restrict_to_base_dir),
        FindTool(base_dir=base_dir, restrict_to_base_dir=restrict_to_base_dir),
        LsTool(base_dir=base_dir, restrict_to_base_dir=restrict_to_base_dir),
    ]
    if include_shell:
        tools.append(ShellTool(base_dir=base_dir))
    if include_local_file_writer:
        tools.append(LocalFileWriteTool(target_directory=target_directory or base_dir))
    if include_web_reader:
        tools.append(WebsiteReaderTool())
    return tools


@dataclass
class CodingAgent(Agent):
    """Coding-focused Agent preset.

    Bundles ReadFile / EditFile / WriteFile / Grep / Find / Ls + Shell +
    LocalFileWrite + WebsiteReader, all sandboxed to ``base_dir``. Any tools
    passed via ``tools=`` are appended to the bundled set, not replaced.
    """

    base_dir: Optional[Union[Path, str]] = None
    target_directory: Optional[Union[Path, str]] = None
    include_shell: bool = True
    include_web_reader: bool = True
    include_local_file_writer: bool = True
    restrict_to_base_dir: bool = True

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = "coding_agent"
        if self.description is None:
            self.description = "An autonomous coding agent that reads, edits, runs, and verifies code."
        if self.instructions is None:
            self.instructions = CODING_AGENT_INSTRUCTIONS

        bundled = build_coding_tools(
            base_dir=self.base_dir,
            target_directory=self.target_directory,
            include_shell=self.include_shell,
            include_web_reader=self.include_web_reader,
            include_local_file_writer=self.include_local_file_writer,
            restrict_to_base_dir=self.restrict_to_base_dir,
        )
        self.tools = bundled + list(self.tools or [])

        super().__post_init__()
