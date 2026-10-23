import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

import gwenflow.agents.agent as agent_mod
from gwenflow.agents.agent import UNANSWERED_APPROVAL, Agent
from gwenflow.llms.base import ChatBase
from gwenflow.tools import BaseTool, Tool
from gwenflow.types import (
    AgentEventToolApprovalRequired,
    AgentEventToolCompleted,
    ModelResponse,
    RequestUsage,
    TextContent,
    ToolCall,
    ToolDenied,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DeleteTool(BaseTool):
    name: str = "delete_file"
    description: str = "Deletes a file"
    requires_approval: bool = True
    deleted: list = field(default_factory=list)

    def _run(self, path: str) -> str:
        self.deleted.append(path)
        return f"deleted {path}"


@dataclass(kw_only=True)
class AddTool(BaseTool):
    name: str = "add"
    description: str = "Adds two numbers"

    def _run(self, a: int, b: int) -> int:
        return a + b


DELETE_CALL = ToolCall(id="call_del", name="delete_file", arguments='{"path": "foo.txt"}')
ADD_CALL = ToolCall(id="call_add", name="add", arguments='{"a": 2, "b": 3}')


def _response(content=None, tool_calls=None):
    parts = []
    if content:
        parts.append(TextContent(content=content))
    if tool_calls:
        parts.extend(tool_calls)
    return ModelResponse(
        parts=parts,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=RequestUsage(input_tokens=10, output_tokens=5),
    )


def _mock_llm(responses=()):
    llm = MagicMock()
    llm.get_context_size.return_value = 96000
    llm.response_format = None
    llm.tools = []
    llm.tool_choice = None
    llm.invoke.side_effect = list(responses)
    llm.input_to_message_list.side_effect = ChatBase.input_to_message_list.__get__(llm)
    llm.get_thinking_parts.return_value = None
    return llm


def _streaming_llm(responses):
    """An LLM whose stream/astream replay `responses`, one per turn."""
    llm = _mock_llm()
    turns = iter(responses)

    def stream(input=None, **kwargs):
        yield next(turns)

    async def astream(input=None, **kwargs):
        yield next(turns)

    llm.stream.side_effect = stream
    llm.astream.side_effect = astream
    return llm


def _sent_messages(llm, call_index=-1):
    return llm.invoke.call_args_list[call_index].kwargs["input"]


def _tool_result(messages, tool_call_id):
    return next(m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id)


# ---------------------------------------------------------------------------
# Declaring a tool that requires approval
# ---------------------------------------------------------------------------


def test_requires_approval_defaults_to_false():
    assert AddTool().requires_approval is False
    assert Tool(lambda x: x).requires_approval is False


def test_function_tool_requires_approval():
    def delete_file(path: str) -> str:
        return path

    assert Tool(delete_file, requires_approval=True).requires_approval is True


def test_tool_decorator_with_and_without_options():
    agent = Agent(llm=_mock_llm())

    @agent.tool
    def read_file(path: str) -> str:
        """Reads a file."""
        return path

    @agent.tool(requires_approval=True)
    def delete_file(path: str) -> str:
        """Deletes a file."""
        return path

    tools = {t.name: t for t in agent.tools}
    assert tools["read_file"].requires_approval is False
    assert tools["delete_file"].requires_approval is True
    assert delete_file("x") == "x"  # the decorator returns the function itself


# ---------------------------------------------------------------------------
# Pause
# ---------------------------------------------------------------------------


def test_run_pauses_before_a_tool_that_requires_approval():
    tool = DeleteTool()
    llm = _mock_llm([_response("I'm about to delete foo.txt.", tool_calls=[DELETE_CALL])])
    agent = Agent(llm=llm, tools=[tool])

    result = agent.run("Delete foo.txt")

    assert result.finish_reason == "approval_required"
    assert result.content == "I'm about to delete foo.txt."
    assert tool.deleted == []
    assert [(r.tool_call_id, r.tool_name, r.tool_args) for r in result.pending_approvals] == [
        ("call_del", "delete_file", {"path": "foo.txt"})
    ]
    events = [e for e in result.events if isinstance(e, AgentEventToolApprovalRequired)]
    assert [(e.tool_call_id, e.tool_args) for e in events] == [("call_del", {"path": "foo.txt"})]
    assert llm.invoke.call_count == 1
    assert [r.tool_call_id for r in agent.get_pending_approvals()] == ["call_del"]


def test_free_tools_of_the_same_turn_run_before_the_pause():
    tool = DeleteTool()
    llm = _mock_llm([_response(tool_calls=[ADD_CALL, DELETE_CALL])])
    agent = Agent(llm=llm, tools=[AddTool(), tool])

    result = agent.run("Add then delete")

    assert result.finish_reason == "approval_required"
    assert result.usage.tool_calls == 1
    assert [m.tool_call_id for m in result.messages] == ["call_add"]
    assert [r.tool_call_id for r in agent.get_pending_approvals()] == ["call_del"]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_approved_call_runs_and_the_run_continues():
    tool = DeleteTool()
    llm = _mock_llm([_response(tool_calls=[DELETE_CALL]), _response("foo.txt is deleted.")])
    agent = Agent(llm=llm, tools=[tool])
    agent.run("Delete foo.txt")

    result = agent.run(approvals={"call_del": True})

    assert result.finish_reason == "stop"
    assert result.content == "foo.txt is deleted."
    assert tool.deleted == ["foo.txt"]
    assert result.usage.tool_calls == 1
    assert _tool_result(_sent_messages(llm), "call_del")["content"] == "deleted foo.txt"
    assert agent.get_pending_approvals() == []


def test_denied_call_gives_the_reason_to_the_model():
    tool = DeleteTool()
    llm = _mock_llm([_response(tool_calls=[DELETE_CALL]), _response("OK, I won't delete it.")])
    agent = Agent(llm=llm, tools=[tool])
    agent.run("Delete foo.txt")

    result = agent.run(approvals={"call_del": ToolDenied("Keep this file, archive it instead.")})

    assert result.finish_reason == "stop"
    assert tool.deleted == []
    assert result.usage.tool_calls == 0
    sent = _sent_messages(llm)
    assert _tool_result(sent, "call_del")["content"] == "Keep this file, archive it instead."
    completed = [e for e in result.events if isinstance(e, AgentEventToolCompleted)]
    assert [e.tool_call_id for e in completed] == ["call_del"]


def test_false_denies_with_the_default_reason():
    tool = DeleteTool()
    llm = _mock_llm([_response(tool_calls=[DELETE_CALL]), _response("Cancelled.")])
    agent = Agent(llm=llm, tools=[tool])
    agent.run("Delete foo.txt")

    agent.run(approvals={"call_del": False})

    assert tool.deleted == []
    assert _tool_result(_sent_messages(llm), "call_del")["content"] == ToolDenied().message


def test_only_an_explicit_true_approves():
    tool = DeleteTool()
    llm = _mock_llm([_response(tool_calls=[DELETE_CALL]), _response("Cancelled.")])
    agent = Agent(llm=llm, tools=[tool])
    agent.run("Delete foo.txt")

    agent.run(approvals={"call_del": "yes"})

    assert tool.deleted == []


def test_new_message_without_decision_denies_the_pending_call():
    tool = DeleteTool()
    llm = _mock_llm([_response(tool_calls=[DELETE_CALL]), _response("Sure, what's next?")])
    agent = Agent(llm=llm, tools=[tool])
    agent.run("Delete foo.txt")

    result = agent.run("Actually, never mind.")

    assert result.finish_reason == "stop"
    assert tool.deleted == []
    sent = [m for m in _sent_messages(llm) if m["role"] != "system"]
    # The tool result closes the pending call before the new user message.
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "user"]
    assert sent[2]["content"] == UNANSWERED_APPROVAL
    assert sent[3]["content"] == "Actually, never mind."


def test_approval_and_new_message_together():
    tool = DeleteTool()
    llm = _mock_llm([_response(tool_calls=[DELETE_CALL]), _response("Deleted, and noted.")])
    agent = Agent(llm=llm, tools=[tool])
    agent.run("Delete foo.txt")

    agent.run("And tell me when it's done.", approvals={"call_del": True})

    assert tool.deleted == ["foo.txt"]
    sent = [m for m in _sent_messages(llm) if m["role"] != "system"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "user"]


def test_approvals_for_unknown_calls_are_ignored(monkeypatch):
    warnings = []
    monkeypatch.setattr(agent_mod.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
    tool = DeleteTool()
    llm = _mock_llm([_response(tool_calls=[DELETE_CALL]), _response("done")])
    agent = Agent(llm=llm, tools=[tool])
    agent.run("Delete foo.txt")

    agent.run(approvals={"call_del": True, "stale_id": True})

    assert tool.deleted == ["foo.txt"]
    assert any("stale_id" in w for w in warnings)


def test_run_without_input_needs_a_pending_call():
    agent = Agent(llm=_mock_llm(), tools=[DeleteTool()])
    with pytest.raises(ValueError, match="input"):
        agent.run()


def test_resume_after_save_and_load_in_a_new_agent(tmp_path):
    path = tmp_path / "session.json"
    first = Agent(llm=_mock_llm([_response(tool_calls=[DELETE_CALL])]), tools=[DeleteTool()])
    first.run("Delete foo.txt")
    first.run(f"/save {path}")

    tool = DeleteTool()
    llm = _mock_llm([_response("foo.txt is deleted.")])
    second = Agent(llm=llm, tools=[tool])
    second.run(f"/load {path}")
    assert [r.tool_call_id for r in second.get_pending_approvals()] == ["call_del"]

    result = second.run(approvals={"call_del": True})

    assert result.content == "foo.txt is deleted."
    assert tool.deleted == ["foo.txt"]


# ---------------------------------------------------------------------------
# Async and streaming paths
# ---------------------------------------------------------------------------


def test_arun_pause_and_resume():
    tool = DeleteTool()
    llm = _mock_llm()
    llm.ainvoke = AsyncMock(side_effect=[_response(tool_calls=[DELETE_CALL]), _response("done")])
    agent = Agent(llm=llm, tools=[tool])

    async def go():
        paused = await agent.arun("Delete foo.txt")
        resumed = await agent.arun(approvals={"call_del": True})
        return paused, resumed

    paused, resumed = asyncio.run(go())

    assert paused.finish_reason == "approval_required"
    assert resumed.finish_reason == "stop"
    assert tool.deleted == ["foo.txt"]


def test_run_stream_pause_and_resume():
    tool = DeleteTool()
    llm = _streaming_llm([_response("Deleting foo.txt.", tool_calls=[DELETE_CALL]), _response("done")])
    agent = Agent(llm=llm, tools=[tool])

    *events, paused = list(agent.run_stream("Delete foo.txt"))
    assert paused.finish_reason == "approval_required"
    assert paused.content == "Deleting foo.txt."
    assert [e.tool_call_id for e in events if isinstance(e, AgentEventToolApprovalRequired)] == ["call_del"]
    assert tool.deleted == []

    *events, resumed = list(agent.run_stream(approvals={"call_del": True}))
    assert resumed.finish_reason == "stop"
    assert resumed.content == "done"
    assert tool.deleted == ["foo.txt"]
    assert [e.tool_call_id for e in events if isinstance(e, AgentEventToolCompleted)] == ["call_del"]


def test_arun_stream_pause_and_resume():
    tool = DeleteTool()
    llm = _streaming_llm([_response(tool_calls=[ADD_CALL, DELETE_CALL]), _response("done")])
    agent = Agent(llm=llm, tools=[AddTool(), tool])

    async def go():
        paused = [e async for e in agent.arun_stream("Add then delete")]
        resumed = [e async for e in agent.arun_stream(approvals={"call_del": ToolDenied("No.")})]
        return paused, resumed

    paused, resumed = asyncio.run(go())

    assert paused[-1].finish_reason == "approval_required"
    assert [e.tool_call_id for e in paused if isinstance(e, AgentEventToolCompleted)] == ["call_add"]
    assert resumed[-1].finish_reason == "stop"
    assert [e.content for e in resumed if isinstance(e, AgentEventToolCompleted)] == ["No."]
    assert tool.deleted == []


# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------


def test_handoff_to_a_member_that_needs_approval_fails_explicitly():
    member_llm = _mock_llm([_response(tool_calls=[DELETE_CALL])])
    member = Agent(name="Janitor", llm=member_llm, tools=[DeleteTool()])
    handoff = ToolCall(id="call_ask", name="ask_janitor", arguments='{"task": "Delete foo.txt"}')
    llm = _mock_llm([_response(tool_calls=[handoff]), _response("The janitor could not do it.")])
    agent = Agent(llm=llm, team=[member])

    result = agent.run("Have the janitor delete foo.txt")

    assert result.finish_reason == "stop"
    content = _tool_result(_sent_messages(llm), "call_ask")["content"]
    assert content.startswith("[handoff failed]")
    assert "delete_file" in content
