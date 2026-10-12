"""Tool calls written as text by the model are recovered (only for tools on offer)."""

import json

from openai.types.chat import ChatCompletion

from gwenflow import ChatOpenAI
from gwenflow.llms.openai import _recover_tool_calls as recover_tool_calls
from gwenflow.tools import Tool

TOOLS = {"read_memory", "memory_edit"}

# Real Qwen outputs that the server returned as text.
QWEN_MISSING_FUNCTION_TAG = """<tool_call>read_memory
<parameter=path>
/memories/user/MEMORY.md
</parameter>
</function>
</tool_call>
<tool_call>read_memory>
<parameter=path>
/memories/user/projects/orion.md
</parameter>
</function>
</tool_call>"""


def args(call):
    return json.loads(call.arguments)


def test_recovers_calls_with_the_function_tag_missing():
    calls, rest = recover_tool_calls(QWEN_MISSING_FUNCTION_TAG, TOOLS)
    assert [c.name for c in calls] == ["read_memory", "read_memory"]
    assert [args(c) for c in calls] == [
        {"path": "/memories/user/MEMORY.md"},
        {"path": "/memories/user/projects/orion.md"},
    ]
    assert rest == ""
    assert calls[0].id != calls[1].id


def test_recovers_well_formed_xml_with_multiline_values():
    text = (
        "Let me fix that.\n<tool_call>\n<function=memory_edit>\n<parameter=path>\n/memories/user/MEMORY.md\n"
        "</parameter>\n<parameter=old_str>\n- a\n- b\n</parameter>\n<parameter=new_str>\n- a\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls, rest = recover_tool_calls(text, TOOLS)
    assert args(calls[0]) == {"path": "/memories/user/MEMORY.md", "old_str": "- a\n- b", "new_str": "- a"}
    assert rest == "Let me fix that."


def test_recovers_json_calls():
    text = '<tool_call>{"name": "read_memory", "arguments": {"path": "/memories/user/MEMORY.md"}}</tool_call>'
    calls, _ = recover_tool_calls(text, TOOLS)
    assert calls[0].name == "read_memory" and args(calls[0]) == {"path": "/memories/user/MEMORY.md"}
    text = '<tool_call>{"name": "read_memory", "arguments": "{\\"path\\": \\"/x.md\\"}"}</tool_call>'
    assert args(recover_tool_calls(text, TOOLS)[0][0]) == {"path": "/x.md"}


def test_leaves_text_alone_when_no_known_tool_is_called():
    text = "<tool_call>delete_everything\n<parameter=path>/</parameter></tool_call>"
    assert recover_tool_calls(text, TOOLS) == ([], text)
    assert recover_tool_calls("plain answer", TOOLS) == ([], "plain answer")
    assert recover_tool_calls("", TOOLS) == ([], "")


def _completion(content):
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def test_parse_response_turns_text_calls_into_tool_calls_only_when_tools_are_offered():
    def read_memory(path: str) -> str:
        """Read a memory file."""
        return path

    llm = ChatOpenAI(model="m", api_key="x")
    assert llm._parse_response(_completion(QWEN_MISSING_FUNCTION_TAG)).tool_calls == []  # no tools on offer
    llm.tools = [Tool(read_memory)]
    response = llm._parse_response(_completion(QWEN_MISSING_FUNCTION_TAG))
    assert [c.name for c in response.tool_calls] == ["read_memory", "read_memory"]
    assert not response.text
