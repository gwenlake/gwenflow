"""Agent.memory: tools registered, prompt block placed, no change without memory."""

import asyncio

from gwenflow import Agent
from gwenflow.llms import ChatBase
from gwenflow.memory import InMemoryMemoryStore, MemoryToolset
from gwenflow.types import ModelResponse, RequestUsage, TextContent

USER = "/memories/user/MEMORY.md"


class EchoLLM:
    def __init__(self):
        self.tools = []
        self.tool_choice = None
        self.response_format = None
        self.prompts = []

    def get_context_size(self):
        return 100_000

    def input_to_message_list(self, input):
        return ChatBase.input_to_message_list(self, input)

    def get_thinking_parts(self, response):
        return None

    def invoke(self, input=None, **kwargs):
        self.prompts.append(input)
        return ModelResponse(
            parts=[TextContent(content="ok")], finish_reason="stop", usage=RequestUsage(input_tokens=1, output_tokens=1)
        )


def loaded_toolset():
    store = InMemoryMemoryStore()
    store.write(USER, "# User memory\n## Profile\n- sales rep\n")
    toolset = MemoryToolset(store, always=[USER], browse=["/memories/user/"])
    asyncio.run(toolset.aload())
    return toolset


def test_agent_with_memory_registers_tools_and_prompt_block():
    llm = EchoLLM()
    agent = Agent(name="a", llm=llm, instructions="Be terse.", memory=loaded_toolset())
    assert {t.name for t in agent.tools} == {"read_memory", "memory_edit", "memory_write", "write_topic_file"}
    assert {t.name for t in agent.llm.tools} >= {"read_memory"}
    prompt = agent.get_system_prompt(task="hi", context={"today": "2026-10-09"})
    i_instr, i_mem, i_ctx = prompt.index("## Instructions:"), prompt.index("## Memory:"), prompt.index("<context>")
    assert i_instr < i_mem < i_ctx
    assert "- sales rep" in prompt
    assert "memory:          1 core file(s), 0 topic file(s)" in agent._config_report()


def test_agent_run_sends_memory_block_once_per_run():
    llm = EchoLLM()
    agent = Agent(name="a", llm=llm, instructions="Be terse.", memory=loaded_toolset())
    agent.run("hello")
    system = llm.prompts[0][0]["content"]
    assert "## Memory:" in system and system.count("## Memory:") == 1


def test_agent_without_memory_is_unchanged():
    llm = EchoLLM()
    agent = Agent(name="a", llm=llm, instructions="Be terse.")
    assert agent.tools == []
    prompt = agent.get_system_prompt(task="hi")
    assert "## Memory:" not in prompt
    assert "memory:          " not in agent._config_report().replace("memory:          ChatSummary", "")


def test_system_prompt_override_skips_memory_block_like_skills():
    llm = EchoLLM()
    agent = Agent(name="a", llm=llm, system_prompt="FIXED", memory=loaded_toolset())
    assert agent.get_system_prompt(task="hi") == "FIXED"
    assert any(t.name == "read_memory" for t in agent.tools)
