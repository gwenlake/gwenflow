"""Context management.

Covers window resolution, token calibration, tool-result lightening and long
tool outputs set aside.
"""

import json
from types import SimpleNamespace

from gwenflow import Agent, ChatOpenAI
from gwenflow.llms.base import CONTEXT_SAFETY_RATIO, DEFAULT_CONTEXT_SIZE, lookup_model_info, normalize_model_name
from gwenflow.llms.models import MODELS
from gwenflow.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from gwenflow.tools import Tool
from gwenflow.types import Message, ToolCall


def words(text) -> int:
    """Deterministic tokenizer for the tests: one token per word."""
    return len(str(text).split())


# ---------------------------------------------------------------------------
# Context window resolution
# ---------------------------------------------------------------------------


def test_lookup_tolerates_server_model_names():
    assert lookup_model_info("mistral-small-3.2-24b") is MODELS["mistral-small-3.2-24b"]
    # dated / quantized suffixes served by vLLM, Scaleway...
    assert lookup_model_info("mistral-small-3.2-24b-instruct-2506") is MODELS["mistral-small-3.2-24b"]
    assert lookup_model_info("qwen3.6-27b-instruct-awq") is MODELS["qwen3.6-27b"]
    # Hugging Face organisation prefix and capitals, in any combination
    assert lookup_model_info("Qwen/Qwen3.6-27B") is MODELS["qwen3.6-27b"]
    assert lookup_model_info("qwen/qwen3.6-35B-A3B") is MODELS["qwen3.6-35b-a3b"]
    assert lookup_model_info("QWEN/QWEN3.6-35B-A3B-INSTRUCT") is MODELS["qwen3.6-35b-a3b"]
    assert lookup_model_info("GPT-5-Mini") is MODELS["gpt-5-mini"]
    assert lookup_model_info(" mistralai/Mistral-Small-3.2-24B-Instruct-2506 ") is MODELS["mistral-small-3.2-24b"]
    # the newest ids resolve to their own entry, not to the shorter sibling
    assert lookup_model_info("claude-opus-5-5-20260922") is MODELS["claude-opus-5-5"]
    assert lookup_model_info("gpt-6.1-sol-2026-09-29") is MODELS["gpt-6.1-sol"]
    # dated builds need no entry of their own
    assert lookup_model_info("deepseek-v4-flash-0731") is MODELS["deepseek-v4-flash"]
    assert lookup_model_info("deepseek-v4.1-flash-1017") is MODELS["deepseek-v4.1-flash"]
    assert lookup_model_info("deepseek-v4-pro-0813") is MODELS["deepseek-v4-pro"]
    assert lookup_model_info("mistralai/Mistral-Large-4-0") is MODELS["mistral-large-4"]
    # the longest known prefix wins
    assert lookup_model_info("gpt-5-mini-2026-01-01") is MODELS["gpt-5-mini"]
    # a dot is a version, not a separator
    assert lookup_model_info("gpt-5.99") is None
    assert lookup_model_info("totally-unknown-model") is None


def test_normalize_model_name():
    assert normalize_model_name("Qwen/Qwen3.6-35B-A3B") == "qwen3.6-35b-a3b"
    assert normalize_model_name("org/sub/Model-X") == "model-x"
    assert normalize_model_name("gpt-5-mini") == "gpt-5-mini"


def test_models_table_keys_are_lowercase():
    """The lookup lowercases the request, so a key with capitals would be unreachable."""
    assert [k for k in MODELS if k != k.lower()] == []


def test_context_size_uses_tolerant_lookup():
    llm = ChatOpenAI(model="mistral-small-3.2-24b-instruct-2506", api_key="x")
    expected = int(MODELS["mistral-small-3.2-24b"]["context_window"] * CONTEXT_SAFETY_RATIO)
    assert llm.get_context_size() == expected


def test_explicit_context_window_overrides_the_table():
    llm = ChatOpenAI(model="gpt-5-mini", api_key="x", context_window=32768)
    assert llm.get_context_size() == int(32768 * CONTEXT_SAFETY_RATIO)


def test_unknown_model_falls_back_to_the_default():
    llm = ChatOpenAI(model="totally-unknown-model", api_key="x")
    assert llm.get_context_size() == int(DEFAULT_CONTEXT_SIZE * CONTEXT_SAFETY_RATIO)


def fake_models_client(models):
    listing = [SimpleNamespace(id=model_id, model_extra=extra) for model_id, extra in models]
    return SimpleNamespace(models=SimpleNamespace(list=lambda: listing))


def test_auto_context_window_reads_vllm_max_model_len():
    llm = ChatOpenAI(model="my-qwen", api_key="x", base_url="http://vllm:8000/v1", context_window="auto")
    llm.client = fake_models_client([("other", {"max_model_len": 1000}), ("my-qwen", {"max_model_len": 32768})])
    assert llm.get_context_size() == int(32768 * CONTEXT_SAFETY_RATIO)
    assert llm.context_window == 32768  # cached: the server is asked once


def test_auto_context_window_falls_back_when_the_server_is_silent():
    llm = ChatOpenAI(model="mistral-small-3.2-24b", api_key="x", context_window="auto")
    llm.client = fake_models_client([("mistral-small-3.2-24b", {})])
    assert llm.resolve_context_window() == MODELS["mistral-small-3.2-24b"]["context_window"]


def test_auto_context_window_survives_a_server_error():
    def boom():
        raise ConnectionError("down")

    llm = ChatOpenAI(model="totally-unknown-model", api_key="x", context_window="auto")
    llm.client = SimpleNamespace(models=SimpleNamespace(list=boom))
    assert llm.resolve_context_window() == DEFAULT_CONTEXT_SIZE


# ---------------------------------------------------------------------------
# Calibration against the provider's token count
# ---------------------------------------------------------------------------


def long_window(n_words: int = 400) -> list[Message]:
    return [Message(role="system", content="sys"), Message(role="user", content="word " * n_words)]


def test_underestimate_is_corrected_at_once():
    buf = ChatMemoryBuffer(token_limit=10_000, tokenizer_fn=words)
    window = long_window()
    estimated = buf._token_count_for_messages(window)
    buf.observe_prompt_tokens(window, int(estimated * 1.5))
    assert abs(buf.token_scale - 1.5) < 0.01
    assert buf._budget() < 10_000 / 1.4


def test_overestimate_is_relaxed_slowly_and_clamped():
    buf = ChatMemoryBuffer(token_limit=10_000, tokenizer_fn=words, token_scale=2.0)
    window = long_window()
    estimated = buf._token_count_for_messages(window)
    buf.observe_prompt_tokens(window, estimated)  # ratio 1.0
    assert 1.0 < buf.token_scale < 2.0
    buf.observe_prompt_tokens(window, estimated * 100)
    assert buf.token_scale == 4.0  # MAX_TOKEN_SCALE


def test_small_or_missing_counts_do_not_calibrate():
    buf = ChatMemoryBuffer(token_limit=10_000, tokenizer_fn=words)
    buf.observe_prompt_tokens([Message(role="user", content="hi")], 500)
    buf.observe_prompt_tokens(long_window(), 0)
    assert buf.token_scale == 1.0


def test_agent_feeds_the_real_prompt_tokens_back():
    agent = Agent(name="a", llm=ChatOpenAI(model="gpt-5-mini", api_key="x"))
    agent.history.tokenizer_fn = words
    window = long_window()
    estimated = agent.history._token_count_for_messages(window)
    agent._calibrate_history(window, SimpleNamespace(input_tokens=estimated * 2))
    assert abs(agent.history.token_scale - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Pass 0: lightening tool results
# ---------------------------------------------------------------------------


def tool_turn(call_id: str, name: str, arguments: dict, result_words: int) -> list[Message]:
    call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
    return [
        Message(role="assistant", content=None, tool_calls=[call]),
        Message(role="tool", tool_call_id=call_id, content=" ".join(["data"] * result_words)),
    ]


def agent_history(n_reads: int = 8, result_words: int = 500) -> list[Message]:
    messages = [Message(role="user", content="Analyse every file and write a report.")]
    for i in range(n_reads):
        messages += tool_turn(f"c{i}", "ReadFileTool", {"file_path": f"f{i}.csv"}, result_words)
    return messages


def contents(messages):
    return [m.content for m in messages if m.role == "tool"]


def test_old_tool_results_are_cleared_before_anything_is_dropped():
    buf = ChatMemoryBuffer(token_limit=2500, tokenizer_fn=words)
    buf.add_messages(agent_history())
    stored_before = [m.content for m in buf.messages]

    window = buf.get()

    # the task and every call are still there: nothing was pruned
    assert window[0].content == "Analyse every file and write a report."
    assert sum(1 for m in window if m.tool_calls) == 8
    results = contents(window)
    assert len(results) == 8
    # the oldest results became stubs naming the call, the 3 most recent are intact
    assert results[0].startswith("[Result of ReadFileTool(") and "f0.csv" in results[0]
    assert all(r.startswith("data") for r in results[-3:])
    # pure read: the stored history is untouched
    assert [m.content for m in buf.messages] == stored_before


def test_clearing_stops_as_soon_as_the_window_fits():
    buf = ChatMemoryBuffer(token_limit=3600, tokenizer_fn=words)
    buf.add_messages(agent_history())
    results = contents(buf.get())
    cleared = [r for r in results if r.startswith("[Result of")]
    assert 0 < len(cleared) < 5
    assert results.index(cleared[-1]) == len(cleared) - 1  # oldest first


def test_nothing_changes_when_the_history_fits():
    buf = ChatMemoryBuffer(token_limit=100_000, tokenizer_fn=words)
    buf.add_messages(agent_history())
    assert all(r.startswith("data") for r in contents(buf.get()))


def test_repeated_calls_keep_only_the_latest_result():
    messages = [Message(role="user", content="task")]
    messages += tool_turn("a", "ReadFileTool", {"file_path": "big.csv"}, 500)
    messages += tool_turn("b", "GrepTool", {"pattern": "x"}, 300)
    messages += tool_turn("c", "ReadFileTool", {"file_path": "big.csv"}, 500)
    buf = ChatMemoryBuffer(token_limit=1200, tokenizer_fn=words, keep_recent_tool_results=3)
    buf.add_messages(messages)
    results = contents(buf.get())
    assert results[0].startswith("[Same call as a later ReadFileTool(")
    assert results[1].startswith("data") and results[2].startswith("data")


def test_summary_buffer_does_not_call_the_llm_when_clearing_is_enough():
    calls = []

    class SpyLLM:
        def invoke(self, input):
            calls.append(input)
            return SimpleNamespace(content="summary", usage=None)

    buf = ChatSummaryMemoryBuffer(token_limit=2500, tokenizer_fn=words, llm=SpyLLM())
    buf.add_messages([Message(role="user", content="earlier question"), Message(role="assistant", content="answer")])
    buf.add_messages(agent_history())
    buf.compact()
    assert calls == []
    assert buf.summary == ""


# ---------------------------------------------------------------------------
# Long tool outputs set aside
# ---------------------------------------------------------------------------


def make_agent(**kwargs):
    def dump_logs() -> str:
        """Return the whole log file."""
        return " ".join(f"line{i}" for i in range(1000))

    agent = Agent(name="a", llm=ChatOpenAI(model="gpt-5-mini", api_key="x"), tools=[Tool(dump_logs)], **kwargs)
    agent.history.tokenizer_fn = words
    return agent


def test_long_tool_output_is_set_aside_and_readable():
    agent = make_agent(max_tool_output_tokens=50)
    assert "read_tool_output" in [t.name for t in agent.get_all_tools()]

    message = agent.run_tool(ToolCall(id="1", name="dump_logs", arguments="{}"))
    assert message.content.startswith("line0 line1")
    assert "Output truncated: 1000 tokens" in message.content
    ref = message.content.split("ref='")[1].split("'")[0]
    offset = int(message.content.split("offset=")[1].split(")")[0])

    page = agent.run_tool(
        ToolCall(id="2", name="read_tool_output", arguments=json.dumps({"ref": ref, "offset": offset}))
    )
    assert page.content.lstrip().startswith("line50")
    assert "continue with offset=" in page.content
    assert "Output truncated" not in page.content  # the reader is never truncated itself


def test_tool_outputs_are_kept_whole_by_default():
    agent = make_agent()
    assert "read_tool_output" not in [t.name for t in agent.get_all_tools()]
    message = agent.run_tool(ToolCall(id="1", name="dump_logs", arguments="{}"))
    assert message.content.endswith("line999")


def test_unknown_reference():
    agent = make_agent(max_tool_output_tokens=50)
    page = agent.run_tool(ToolCall(id="1", name="read_tool_output", arguments=json.dumps({"ref": "nope"})))
    assert page.content == "Unknown reference 'nope'."
