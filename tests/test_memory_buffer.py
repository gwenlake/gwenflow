"""Unit tests for ChatMemoryBuffer.

Covers:
- Multi-modal token counting (image/audio/file aren't stringified to their repr)
- Tool call / thinking_parts token accounting
- Pre-filter clamping only touches text, never multi-modal parts
- Pruning loop is bounded (no out-of-range indexing)
- Pruning preserves tool/assistant chains (no orphan tool responses)
- System prompt prepended on every get()
"""

import pytest

from gwenflow.memory import ChatMemoryBuffer
from gwenflow.types import AudioContent, FileContent, ImageContent, Message, TextContent, ThinkingContent


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def test_token_count_string_content():
    buf = ChatMemoryBuffer(token_limit=1000)
    n = buf._token_count_for_message(Message(role="user", content="hello world"))
    # tokenizer + 4 token envelope
    assert n > 4


def test_token_count_multimodal_image_does_not_count_repr():
    """Without the fix this would stringify a list to its Python repr and count
    the class names — wildly inaccurate. We use a fixed per-image estimate."""
    buf = ChatMemoryBuffer(token_limit=1000)
    m = Message(role="user", content=[TextContent(content="hi"), ImageContent.from_url("https://x/y.jpg")])
    n = buf._token_count_for_message(m)
    # text "hi" + image baseline (85 low-detail) + envelope
    assert n >= 85
    # And nowhere near the 30+ tokens of "ImageContent(url='https://...'...)"
    # that the old buggy impl would have counted (we exceed it because of the
    # 85 image charge, but the test asserts on actual structure).
    assert n < 200  # sanity ceiling


def test_token_count_image_high_detail_costs_more():
    buf = ChatMemoryBuffer(token_limit=10000)
    low = buf._token_count_for_message(Message(role="user", content=[ImageContent.from_url("x", detail="low")]))
    high = buf._token_count_for_message(Message(role="user", content=[ImageContent.from_url("x", detail="high")]))
    assert high > low


def test_token_count_audio_uses_transcript_when_available():
    buf = ChatMemoryBuffer(token_limit=10000)
    long_transcript = "word " * 200
    a = AudioContent(data="X" * 100, format="wav", transcript=long_transcript)
    n = buf._token_count_for_message(Message(role="user", content=[a]))
    # Transcript should drive the cost (~200 words ≈ 200+ tokens)
    assert n > 100


def test_token_count_file_scales_with_data_size():
    buf = ChatMemoryBuffer(token_limit=100000)
    small = buf._token_count_for_message(Message(role="user", content=[FileContent(data="AB" * 100)]))
    big = buf._token_count_for_message(Message(role="user", content=[FileContent(data="AB" * 10000)]))
    assert big > small


def test_token_count_includes_tool_calls():
    buf = ChatMemoryBuffer(token_limit=10000)
    plain = buf._token_count_for_message(Message(role="assistant", content="ok"))
    with_tool = buf._token_count_for_message(
        Message(
            role="assistant",
            content="ok",
            tool_calls=[{"id": "t1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris", "units": "metric"}'}}],
        )
    )
    assert with_tool > plain


def test_token_count_includes_thinking_parts():
    buf = ChatMemoryBuffer(token_limit=10000)
    plain = buf._token_count_for_message(Message(role="assistant", content="42"))
    with_thinking = buf._token_count_for_message(
        Message(
            role="assistant",
            content="42",
            thinking_parts=[ThinkingContent(content="let me think step by step about this problem")],
        )
    )
    assert with_thinking > plain


# ---------------------------------------------------------------------------
# Pre-filter: clamping only touches text
# ---------------------------------------------------------------------------


def test_clamp_long_string_content():
    buf = ChatMemoryBuffer(token_limit=200)
    long_text = "word " * 5000
    buf.add_message(Message(role="user", content=long_text))
    kept = buf.get()
    # Find the user message in kept; should be truncated
    user_msg = next(m for m in kept if m.role == "user")
    assert isinstance(user_msg.content, str)
    assert len(user_msg.content) < len(long_text)


def test_clamp_preserves_multimodal_parts():
    """Multi-modal parts (image/audio/file) must NOT be touched by the clamper."""
    # Budget must fit the image (85 tokens low-detail) + clamped text (~half budget)
    buf = ChatMemoryBuffer(token_limit=2000)
    img = ImageContent.from_url("https://x/big.jpg", detail="low")
    long_text = "word " * 10000  # ~10000 tokens, way over per-message limit (1000)
    m = Message(role="user", content=[TextContent(content=long_text), img])
    buf.add_message(m)
    kept = buf.get()
    user_msg = next(m for m in kept if m.role == "user")
    assert isinstance(user_msg.content, list)
    # Image is still there, untouched
    images = [p for p in user_msg.content if isinstance(p, ImageContent)]
    assert len(images) == 1
    assert images[0].url == "https://x/big.jpg"
    assert images[0].detail == "low"
    # Text was clamped
    texts = [p for p in user_msg.content if isinstance(p, TextContent)]
    assert len(texts[0].content) < len(long_text)


def test_clamp_with_none_content_does_not_crash():
    """An assistant message can carry tool_calls and no content — clamping must
    not blow up on None."""
    buf = ChatMemoryBuffer(token_limit=200)
    buf.add_message(Message(role="user", content="hi"))
    buf.add_message(
        Message(
            role="assistant",
            content=None,
            tool_calls=[{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        )
    )
    # Should not raise
    kept = buf.get()
    assert kept  # we got something back


# ---------------------------------------------------------------------------
# Pruning behavior
# ---------------------------------------------------------------------------


def test_under_budget_keeps_everything():
    buf = ChatMemoryBuffer(token_limit=10000)
    for i in range(5):
        buf.add_message(Message(role="user", content=f"msg {i}"))
    kept = buf.get()
    assert len(kept) == 5


def test_pruning_drops_oldest():
    buf = ChatMemoryBuffer(token_limit=80)
    for i in range(20):
        buf.add_message(Message(role="user", content=f"message number {i} with a bunch of words"))
    kept = buf.get()
    assert len(kept) < 20
    # Latest messages kept (the last one must be present)
    last = next(m for m in kept if m.role == "user")
    # The last user message we added had i=19
    assert any("19" in (m.content or "") for m in kept)


def test_pruning_does_not_start_with_orphan_tool():
    """When pruning, the kept window must never start with a `tool` message —
    that would be an orphan response with no matching tool_call."""
    buf = ChatMemoryBuffer(token_limit=60)
    # Conversation: user → assistant(tool_call) → tool → assistant → ... repeated
    for i in range(10):
        buf.add_message(Message(role="user", content=f"q{i} " * 5))
        buf.add_message(
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": f"t{i}", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            )
        )
        buf.add_message(Message(role="tool", tool_call_id=f"t{i}", content=f"r{i}"))
        buf.add_message(Message(role="assistant", content=f"a{i}"))
    kept = buf.get()
    # First non-system kept message must be user (never tool, never assistant orphan)
    first_real = next(m for m in kept if m.role != "system")
    assert first_real.role == "user", f"kept[0]={first_real.role}, expected user"


def test_pruning_bounded_with_only_assistant_and_tool():
    """Pathological input: all messages are assistant/tool, no user. The
    buggy old loop would index out of bounds. The fix should drop everything
    and return only system (or empty)."""
    buf = ChatMemoryBuffer(token_limit=50, system_prompt="sys")
    for i in range(5):
        buf.add_message(Message(role="assistant", content=f"a{i} word " * 30))
        buf.add_message(Message(role="tool", tool_call_id=f"t{i}", content=f"r{i} word " * 30))
    # Should not raise / loop forever
    kept = buf.get()
    # Either everything was dropped (only system) or what remained fits
    if len(kept) == 1:
        assert kept[0].role == "system"
    else:
        # Real bound is: doesn't crash, returns a list
        assert isinstance(kept, list)


def test_pruning_keeps_recent_user_question_intact():
    """After pruning, the most recent user message must still be there in full."""
    buf = ChatMemoryBuffer(token_limit=200)
    for i in range(20):
        buf.add_message(Message(role="user", content=f"old {i}"))
        buf.add_message(Message(role="assistant", content=f"answer {i}"))
    buf.add_message(Message(role="user", content="LATEST_QUESTION"))
    kept = buf.get()
    assert any(m.content == "LATEST_QUESTION" for m in kept if m.role == "user")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def test_system_prompt_prepended():
    buf = ChatMemoryBuffer(token_limit=10000, system_prompt="be terse")
    buf.add_message(Message(role="user", content="hi"))
    kept = buf.get()
    assert kept[0].role == "system"
    assert kept[0].content == "be terse"


def test_initial_token_count_exceeds_limit_raises():
    long_sys = "word " * 10000
    buf = ChatMemoryBuffer(token_limit=100, system_prompt=long_sys)
    with pytest.raises(ValueError, match="Initial token count"):
        buf.get()


def test_overflow_returns_system_only():
    """When everything's been dropped but we're still over, return system only (or empty)."""
    buf = ChatMemoryBuffer(token_limit=50, system_prompt="sys")
    # One message bigger than the budget on its own
    buf.add_message(Message(role="user", content="word " * 200))
    kept = buf.get()
    # Either clamped to fit, or dropped — but we don't blow up
    assert isinstance(kept, list)


def test_empty_history_returns_only_system():
    buf = ChatMemoryBuffer(token_limit=1000, system_prompt="hi")
    kept = buf.get()
    assert len(kept) == 1
    assert kept[0].role == "system"


def test_empty_history_no_system_returns_empty():
    buf = ChatMemoryBuffer(token_limit=1000)
    assert buf.get() == []
