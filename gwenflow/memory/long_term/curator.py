"""The memory curator: an agent that tidies a user's memory directory.

Runs when the user is idle (the application decides when). It gets every memory
tool, ``memory_move`` included, with the recent thread notes as always-loaded
files: it integrates their "Candidate memories", merges duplicates, resolves
contradictions, keeps the core file short and archives files nobody reads.
Symmetric with :class:`~gwenflow.memory.long_term.consolidator.MemoryConsolidator`::

    curator = MemoryCurator(llm)
    await toolset.aload(show=recently_changed_topic_files)
    await curator.arun(toolset)
    await toolset.aflush()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gwenflow.memory.long_term.prompts import CURATION_PROMPT, CURATION_TASK, FORBIDDEN_CONTENT
from gwenflow.memory.long_term.toolset import MemoryToolset
from gwenflow.types import AgentUsage

if TYPE_CHECKING:
    from gwenflow.agents import Agent
    from gwenflow.types import AgentResponse


class MemoryCurator:
    """Tidies a memory directory with an :class:`~gwenflow.agents.Agent` driven by ``llm``.

    Args:
        llm: the chat model of the curator agent (a cheap one does).
        archive_after_days: topic files last read longer ago are moved under ``archive/``.
        max_turns: cap on the agent's tool rounds per run.
        tool_choice: ``"required"`` makes the first round a tool call (the run loop
            relaxes it to ``"auto"`` afterwards), which keeps small models from
            narrating changes instead of making them.
        name: the agent's name, for traces.
    """

    def __init__(
        self,
        llm: Any,
        *,
        archive_after_days: int = 90,
        max_turns: int = 12,
        tool_choice: str = "required",
        name: str = "Memory curator",
    ) -> None:
        self.llm = llm
        self.archive_after_days = archive_after_days
        self.max_turns = max_turns
        self.tool_choice = tool_choice
        self.name = name
        self.usage = AgentUsage()

    def agent(self, toolset: MemoryToolset) -> Agent:
        """The curator agent wired on ``toolset``, for callers who want to run it themselves.

        The toolset's ``always`` files are the core memory file(s) and the recent thread
        notes, its ``browse`` prefix the user's directory. The curator needs every tool
        (archiving included) and its own instructions, so the toolset is set up for it:
        ``allow_move`` is turned on, a tool restriction is lifted, and the conversational
        protocol is dropped unless one was set explicitly. Start the run with
        :data:`~gwenflow.memory.long_term.prompts.CURATION_TASK`, as :meth:`arun` does.
        """
        from gwenflow.agents import Agent

        toolset.allow_move = True
        toolset.tool_names = None
        if toolset.protocol is None:
            toolset.protocol = ""

        core = ", ".join(f"`{p}`" for p in toolset.always if p.endswith("MEMORY.md")) or "the core memory file"
        browse = toolset.browse[0] if toolset.browse else "/memories/user/"
        instructions = CURATION_PROMPT.format(
            core=core,
            browse=browse,
            max_lines=toolset.max_lines,
            archive_after_days=self.archive_after_days,
            forbidden=FORBIDDEN_CONTENT,
        )
        return Agent(
            name=self.name,
            llm=self.llm,
            instructions=instructions,
            memory=toolset,
            tool_choice=self.tool_choice,
            max_turns=self.max_turns,
        )

    async def arun(self, toolset: MemoryToolset) -> AgentResponse:
        """Curate a loaded ``toolset``: the edits land in its working copy, persist with ``await toolset.aflush()``."""
        agent = self.agent(toolset)
        response = await agent.arun(CURATION_TASK)
        self.usage.add(agent.total_usage)
        return response
