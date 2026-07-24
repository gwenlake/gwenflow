import json
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


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


def _make_batch_completion_body(text="hello", custom_id="0"):
    return {
        "id": f"chatcmpl-{custom_id}",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_build_batch_line_shape():
    llm = ChatOpenAI(api_key="fake-key", model="gpt-4o-mini", temperature=0.5)
    line = llm._build_batch_line("custom-1", "hello")

    assert line["custom_id"] == "custom-1"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/chat/completions"
    assert line["body"]["model"] == "gpt-4o-mini"
    assert line["body"]["temperature"] == 0.5
    assert line["body"]["messages"] == [{"role": "user", "content": "hello"}]


def test_build_batch_file_rejects_empty_inputs():
    llm = ChatOpenAI(api_key="fake-key")
    with pytest.raises(ValueError, match="non-empty"):
        llm._build_batch_file([], None)


def test_build_batch_file_rejects_mismatched_custom_ids():
    llm = ChatOpenAI(api_key="fake-key")
    with pytest.raises(ValueError, match="same length"):
        llm._build_batch_file(["a", "b"], ["only-one"])


def test_build_batch_file_rejects_duplicate_custom_ids():
    llm = ChatOpenAI(api_key="fake-key")
    with pytest.raises(ValueError, match="unique"):
        llm._build_batch_file(["a", "b"], ["dup", "dup"])


def test_create_batch_uploads_file_and_submits_job(monkeypatch):
    fake_client = MagicMock()
    fake_client.files.create.return_value = SimpleNamespace(id="file-123")
    fake_client.batches.create.return_value = SimpleNamespace(id="batch-123", status="validating")
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    batch = llm.create_batch(["hello", "world"])

    assert batch.id == "batch-123"
    fake_client.files.create.assert_called_once()
    assert fake_client.files.create.call_args.kwargs["purpose"] == "batch"
    fake_client.batches.create.assert_called_once_with(
        input_file_id="file-123",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=None,
    )


def test_get_batch_results_matches_by_custom_id(monkeypatch):
    output_lines = "\n".join(
        [
            json.dumps(
                {
                    "custom_id": "0",
                    "response": {"status_code": 200, "body": _make_batch_completion_body("hi zero", "0")},
                    "error": None,
                }
            ),
            json.dumps(
                {
                    "custom_id": "1",
                    "response": {"status_code": 200, "body": _make_batch_completion_body("hi one", "1")},
                    "error": None,
                }
            ),
            json.dumps({"custom_id": "2", "response": None, "error": {"code": "server_error", "message": "boom"}}),
        ]
    )

    fake_client = MagicMock()
    fake_client.batches.retrieve.return_value = SimpleNamespace(
        status="completed", output_file_id="out-file", error_file_id=None
    )
    fake_client.files.content.return_value = SimpleNamespace(text=output_lines)
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    results = llm.get_batch_results("batch-123")

    assert results["0"].response.content == "hi zero"
    assert results["0"].error is None
    assert results["1"].response.content == "hi one"
    assert results["2"].response is None
    assert results["2"].error == {"code": "server_error", "message": "boom"}


def test_get_batch_results_raises_if_not_finished(monkeypatch):
    fake_client = MagicMock()
    fake_client.batches.retrieve.return_value = SimpleNamespace(
        status="in_progress", output_file_id=None, error_file_id=None
    )
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    with pytest.raises(RuntimeError, match="not finished yet"):
        llm.get_batch_results("batch-123")


def test_get_batch_results_surfaces_validation_errors_on_failed_status(monkeypatch):
    batch_error = SimpleNamespace(code="invalid_request", line=3, message="bad model name", param="model")
    fake_client = MagicMock()
    fake_client.batches.retrieve.return_value = SimpleNamespace(
        status="failed",
        output_file_id=None,
        error_file_id=None,
        errors=SimpleNamespace(data=[batch_error]),
    )
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)

    llm = ChatOpenAI(api_key="fake-key")
    results = llm.get_batch_results("batch-123")

    assert len(results) == 1
    item = next(iter(results.values()))
    assert item.response is None
    assert item.error == {"code": "invalid_request", "message": "bad model name", "param": "model", "line": 3}


def test_poll_batch_stops_at_terminal_status(monkeypatch):
    statuses = iter(["validating", "in_progress", "completed"])
    fake_client = MagicMock()
    fake_client.batches.retrieve.side_effect = lambda batch_id: SimpleNamespace(status=next(statuses))
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)
    monkeypatch.setattr("gwenflow.llms.openai.time.sleep", lambda _: None)

    llm = ChatOpenAI(api_key="fake-key")
    batch = llm.poll_batch("batch-123", poll_interval=0)

    assert batch.status == "completed"
    assert fake_client.batches.retrieve.call_count == 3


def test_poll_batch_times_out(monkeypatch):
    fake_client = MagicMock()
    fake_client.batches.retrieve.return_value = SimpleNamespace(status="in_progress")
    monkeypatch.setattr(ChatOpenAI, "get_client", lambda self: fake_client)
    monkeypatch.setattr("gwenflow.llms.openai.time.sleep", lambda _: None)

    times = iter([0.0, 0.0, 5.0])
    monkeypatch.setattr("gwenflow.llms.openai.time.monotonic", lambda: next(times))

    llm = ChatOpenAI(api_key="fake-key")
    with pytest.raises(TimeoutError):
        llm.poll_batch("batch-123", poll_interval=0, timeout=1.0)
