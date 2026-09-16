import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from gwenflow.llms.openai import ChatOpenAI
from gwenflow.types import Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_response(text="hello", tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(
        role="assistant",
        content=text,
        tool_calls=tool_calls,
        reasoning_content=None,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


def skip_if_no_real_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    return pytest.mark.skipif(key in [None, "", "test"], reason="OPENAI_API_KEY missing or fake value.")


# ---------------------------------------------------------------------------
# Unit tests (mocked client)
# ---------------------------------------------------------------------------


def test_invoke_returns_content(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_openai_response("hello from openai")
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    result = llm.invoke("Say hello")

    assert result.content == "hello from openai"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_invoke_with_message_list(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_openai_response("pong")
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    messages = [Message(role="user", content="ping")]
    result = llm.invoke(messages)

    assert result.content == "pong"


def test_invoke_with_tool_calls(monkeypatch):
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="my_tool", arguments='{"x": 1}'),
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_openai_response("", tool_calls=[tool_call])
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    result = llm.invoke("Use a tool")

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "my_tool"
    assert result.tool_calls[0].arguments == '{"x": 1}'


def test_invoke_no_usage_returns_none(monkeypatch):
    response = _make_openai_response("ok")
    response = SimpleNamespace(choices=response.choices, usage=None)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = response
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    result = llm.invoke("hi")

    assert result.usage is None


def test_invoke_raises_on_api_error(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = Exception("network error")
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    with pytest.raises(RuntimeError, match="Error in calling openai API"):
        llm.invoke("hello")


# ---------------------------------------------------------------------------
# Streaming tool calls
# ---------------------------------------------------------------------------


def _tool_call_delta(tool_id=None, name=None, arguments=None, index=0, function_unset=False):
    function = None if function_unset else SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=tool_id, index=index, type="function", function=function)


def _stream_chunk(tool_calls, finish_reason=None):
    delta = SimpleNamespace(content=None, tool_calls=tool_calls, audio=None, reasoning_content=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _vllm_tool_call_chunks():
    """Deltas as vLLM emits them: the call opens with name/arguments unset, where OpenAI sends ""."""
    return [
        _stream_chunk([_tool_call_delta(tool_id="chatcmpl-tool-8382", name="search_medias", arguments=None)]),
        _stream_chunk([_tool_call_delta(arguments='{"query": "')]),
        _stream_chunk([_tool_call_delta(arguments='paris"}')]),
        _stream_chunk(None, finish_reason="tool_calls"),
    ]


def _streamed_tool_calls(responses):
    return [part for part in responses[-1].parts if getattr(part, "kind", None) == "tool-call"]


def _last_tool_call(responses):
    tool_calls = _streamed_tool_calls(responses)
    assert len(tool_calls) == 1
    return tool_calls[0]


def _stream_tool_calls(monkeypatch, chunks):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(chunks)
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)
    return _streamed_tool_calls(list(ChatOpenAI(api_key="fake-key").stream("hello")))


def test_stream_keeps_tool_calls_apart_when_server_sends_no_id(monkeypatch):
    """Some OpenAI-compatible servers never send an id; `index` is what separates the calls."""
    tool_calls = _stream_tool_calls(
        monkeypatch,
        [
            _stream_chunk([_tool_call_delta(tool_id="", name="tool_a", arguments='{"a": 1}', index=0)]),
            _stream_chunk([_tool_call_delta(tool_id="", name="tool_b", arguments='{"b": 2}', index=1)]),
            _stream_chunk(None, finish_reason="tool_calls"),
        ],
    )

    assert [(tc.name, tc.arguments) for tc in tool_calls] == [("tool_a", '{"a": 1}'), ("tool_b", '{"b": 2}')]


def test_stream_does_not_duplicate_a_name_repeated_on_every_delta(monkeypatch):
    """Some servers repeat id and name on each fragment instead of only the first."""
    tool_calls = _stream_tool_calls(
        monkeypatch,
        [
            _stream_chunk([_tool_call_delta(tool_id="call_9", name="search_medias", arguments='{"query": "')]),
            _stream_chunk([_tool_call_delta(tool_id="call_9", name="search_medias", arguments='paris"}')]),
            _stream_chunk(None, finish_reason="tool_calls"),
        ],
    )

    assert [(tc.name, tc.arguments) for tc in tool_calls] == [("search_medias", '{"query": "paris"}')]


def test_stream_interleaves_parallel_tool_calls_by_index(monkeypatch):
    tool_calls = _stream_tool_calls(
        monkeypatch,
        [
            _stream_chunk([_tool_call_delta(tool_id="call_a", name="tool_a", arguments="", index=0)]),
            _stream_chunk([_tool_call_delta(tool_id="call_b", name="tool_b", arguments="", index=1)]),
            _stream_chunk(
                [_tool_call_delta(arguments='{"a": ', index=0), _tool_call_delta(arguments='{"b": ', index=1)]
            ),
            _stream_chunk([_tool_call_delta(arguments="1}", index=0), _tool_call_delta(arguments="2}", index=1)]),
            _stream_chunk(None, finish_reason="tool_calls"),
        ],
    )

    assert [(tc.id, tc.name, tc.arguments) for tc in tool_calls] == [
        ("call_a", "tool_a", '{"a": 1}'),
        ("call_b", "tool_b", '{"b": 2}'),
    ]


def test_stream_reassembles_a_name_split_across_deltas(monkeypatch):
    tool_calls = _stream_tool_calls(
        monkeypatch,
        [
            _stream_chunk([_tool_call_delta(tool_id="call_7", name="search_", arguments="")]),
            _stream_chunk([_tool_call_delta(name="medias", arguments='{"query": 1}')]),
            _stream_chunk(None, finish_reason="tool_calls"),
        ],
    )

    assert [(tc.name, tc.arguments) for tc in tool_calls] == [("search_medias", '{"query": 1}')]


def test_stream_accumulates_tool_call_opened_without_arguments(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(_vllm_tool_call_chunks())
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    tool_call = _last_tool_call(list(llm.stream("hello")))

    assert tool_call.id == "chatcmpl-tool-8382"
    assert tool_call.name == "search_medias"
    assert tool_call.arguments == '{"query": "paris"}'


def test_stream_tolerates_tool_call_delta_without_function(monkeypatch):
    """`ChoiceDeltaToolCall.function` is Optional in the SDK, so a delta may omit it."""
    chunks = [
        _stream_chunk([_tool_call_delta(tool_id="chatcmpl-tool-8382", function_unset=True)]),
        _stream_chunk([_tool_call_delta(name="search_medias", arguments='{"query": "paris"}')]),
        _stream_chunk(None, finish_reason="tool_calls"),
    ]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(chunks)
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    tool_call = _last_tool_call(list(llm.stream("hello")))

    assert tool_call.id == "chatcmpl-tool-8382"
    assert tool_call.name == "search_medias"
    assert tool_call.arguments == '{"query": "paris"}'


def test_astream_accumulates_tool_call_opened_without_arguments(monkeypatch):
    async def fake_create(*args, **kwargs):
        async def iterator():
            for chunk in _vllm_tool_call_chunks():
                yield chunk

        return iterator()

    fake_client = MagicMock()
    fake_client.chat.completions.create = fake_create
    monkeypatch.setattr(ChatOpenAI, "get_async_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")

    async def collect():
        return [response async for response in llm.astream("hello")]

    tool_call = _last_tool_call(asyncio.run(collect()))

    assert tool_call.id == "chatcmpl-tool-8382"
    assert tool_call.name == "search_medias"
    assert tool_call.arguments == '{"query": "paris"}'


# ---------------------------------------------------------------------------
# _format_response
# ---------------------------------------------------------------------------


def test_format_response_json_object():
    llm = ChatOpenAI(api_key="fake-key", response_format={"type": "json_object"})
    result = llm._format_response('{"key": "value"}', response_format={"type": "json_object"})
    assert result == {"key": "value"}


def test_format_response_pydantic_model():
    class MyModel(BaseModel):
        name: str

    llm = ChatOpenAI(api_key="fake-key")
    result = llm._format_response('{"name": "alice"}', response_format=MyModel)
    assert result == {"name": "alice"}


def test_format_response_plain_text():
    llm = ChatOpenAI(api_key="fake-key")
    result = llm._format_response("hello", response_format=None)
    assert result == "hello"


def test_format_response_none():
    llm = ChatOpenAI(api_key="fake-key")
    assert llm._format_response(None, response_format=None) is None


def test_format_response_invalid_json_falls_back():
    llm = ChatOpenAI(api_key="fake-key")
    result = llm._format_response("not json at all", response_format={"type": "json_object"})
    assert result == "not json at all"


# ---------------------------------------------------------------------------
# _model_params
# ---------------------------------------------------------------------------


def test_model_params_empty_by_default():
    llm = ChatOpenAI(api_key="fake-key")
    params = llm._model_params
    assert "temperature" not in params
    assert "tools" not in params
    assert "response_format" not in params


def test_model_params_includes_temperature():
    llm = ChatOpenAI(api_key="fake-key", temperature=0.7)
    assert llm._model_params["temperature"] == 0.7


def test_model_params_includes_response_format_pydantic():
    class MySchema(BaseModel):
        answer: str

    llm = ChatOpenAI(api_key="fake-key", response_format=MySchema)
    params = llm._model_params
    assert params["response_format"]["type"] == "json_schema"
    assert params["response_format"]["json_schema"]["name"] == "MySchema"
    assert params["response_format"]["json_schema"]["strict"] is True


# ---------------------------------------------------------------------------
# Client params
# ---------------------------------------------------------------------------


def test_get_client_params_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = ChatOpenAI()
    with pytest.raises(ValueError, match="api_key"):
        llm._get_client_params()


def test_get_client_params_uses_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    llm = ChatOpenAI()
    params = llm._get_client_params()
    assert params["api_key"] == "env-key"


# ---------------------------------------------------------------------------
# Integration test (real API)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@skip_if_no_real_api_key()
def test_invoke_real_api():
    llm = ChatOpenAI()
    result = llm.invoke("Reply with exactly one word: hello")

    assert result.content is not None
    assert len(result.content) > 0
    assert result.usage.input_tokens > 0


@pytest.mark.integration
@skip_if_no_real_api_key()
def test_stream_real_api():
    llm = ChatOpenAI()
    chunks = list(llm.stream("Reply with exactly one word: hello"))

    text_chunks = [c for c in chunks if c.content]
    assert len(text_chunks) > 0
    full_text = "".join(c.content for c in text_chunks)
    assert len(full_text) > 0
