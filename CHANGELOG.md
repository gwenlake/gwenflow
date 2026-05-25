# Changelog

All notable changes to gwenflow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-25

First stable release. The API surface is now committed and any breaking change
will warrant a 2.x. This release brings native SDK adoption for Anthropic /
Mistral / Google, end-to-end thinking handling across providers, and full
multi-modal input/output.

### ⚠️ Breaking changes — migration guide

| Before | After |
|---|---|
| `from gwenflow.llms import ChatGemini` | `from gwenflow import ChatGoogle` *(renamed for consistency with `ChatOpenAI`, `ChatAnthropic`, etc.)* |
| `Message(role="...", reasoning_content="...")` | Field removed — was never sent on the wire. Use `Message.thinking_parts` for echo-back where the provider requires it (Anthropic extended thinking). |
| `ModelResponse.to_openai()` | Removed from the type. Use `response_to_openai_dict(response)` from `gwenflow.llms.openai` (or the provider-specific equivalents in `deepseek.py`, `anthropic.py`, `google.py`). |
| `MessageContent = TextContent \| ThinkingContent \| ToolCall` | Split into `MessageContent` (input, no thinking) and `ResponsePart` (output, includes `ThinkingContent`). `Message.content` now uses `MessageContent`; `ModelResponse.parts` uses `ResponsePart`. |
| `gwenflow.llms.models.MODELS["deepseek-r1"]` | Renamed to `"deepseek-reasoner"` to match the actual DeepSeek API model name. |
| `ChatMistral` / `ChatGoogle` via the OpenAI-compat endpoint | Now use the native `mistralai` and `google-genai` SDKs. Same `ChatBase` API — translation happens internally. |

### Added

- **Multi-modal input** — `ImageContent`, `AudioContent`, `FileContent` content
  parts with `from_url` / `from_path` / `from_bytes` helpers. Each provider's
  adapter translates to its native wire format (`image_url` for OpenAI,
  `image` block for Anthropic, `inline_data` for Google).
- **Multi-modal output** — audio from `gpt-4o-audio-preview` surfaced via
  `response.audio` (an `AudioContent` aggregating streamed chunks).
  `response.images` exposes generated images.
- **Thinking / reasoning, uniform across providers**
  - OpenAI gpt-5 / o-series: `reasoning_effort` param on `ChatOpenAI`, full
    reasoning summaries via `ResponseOpenAI` (Responses API).
  - Anthropic: `thinking={"type": "enabled", "budget_tokens": N}` on
    `ChatAnthropic`, with signature preservation for extended-thinking +
    tool-use turns.
  - Mistral Magistral: `ThinkChunk` parsing.
  - Google Gemini 2.5: `thinking={"include_thoughts": True, "thinking_budget": N}`
    on `ChatGoogle`, with `thought_signature` echo-back.
  - DeepSeek-R1: `reasoning_content` parsed natively.
  - Local models (Ollama qwen3, deepseek-r1 distills, gemma3): inline
    `<think>...</think>` extracted via `ChatOllama`'s default
    `extract_think_tags=True` (handles tags split across stream chunks).
  - Agent-level: `AgentEventThinking` events on streams,
    `AgentResponse.reasoning_content` accumulates across turns.
- **Native SDKs** — `ChatMistral` uses `mistralai>=1.5`, `ChatGoogle` uses
  `google-genai>=1.0`. Both support: invoke / ainvoke / stream / astream,
  tool calls, structured output, multi-modal input, thinking.
- **Top-level exports** — `ChatGoogle`, `ChatMistral`, `ChatDeepSeek` now
  importable directly: `from gwenflow import ChatMistral`.
- **Provider response serializers** — `response_to_openai_dict`,
  `response_to_deepseek_dict`, `response_to_anthropic_dict`,
  `response_to_google_dict` for serializing a `ModelResponse` back to each
  provider's wire shape.
- **Tests** — 64 new pytest tests covering multi-modal input/output, native
  Mistral/Google adapters, audio output parsing, Skills regression, MCP
  client integration, and memory buffer behavior.

### Changed

- **`ChatMemoryBuffer`** — multi-modal-aware token counting (image/audio/file
  parts get proper per-type token estimates instead of being stringified to
  their Python `repr`). Includes `tool_calls`, `thinking_parts`, and `name`
  fields in the token total.
- **`Message.to_openai()`** — translates `list[MessageContent]` to the OpenAI
  multi-modal wire shape (`{"type": "text", ...}`, `{"type": "image_url", ...}`,
  `{"type": "input_audio", ...}`, `{"type": "file", ...}`).
- **`ChatAnthropic._format_messages`** — handles `ImageContent`, `FileContent`
  (`document` block), rejects `AudioContent` with `NotImplementedError`.
- **`Agent.run` / `arun` / `run_stream` / `arun_stream`** — emit
  `AgentEventThinking` for native model thinking, attach
  `_message.thinking_parts = self.llm.get_thinking_parts(response)` for
  echo-back, accumulate thinking across turns.

### Fixed

- **OpenAI `_parse_response`** — used to assign `block.message.content` to
  `ThinkingContent` instead of `reasoning_content`, and the `hasattr` check
  was always true. Now reads `reasoning_content` (DeepSeek/Mistral/Gemma) or
  `reasoning` (gpt-5/o-series) correctly.
- **OpenAI streaming** — thinking deltas (`delta.reasoning_content` /
  `delta.reasoning`) were silently dropped. Now emitted as `ThinkingContent`
  chunks.
- **Anthropic streaming** — `signature_delta` events now captured into
  `ThinkingContent.extra["signature"]` so extended thinking with tools works
  across multi-turn conversations.
- **`Skill.get_tools()` / `SkillsToolset.get_tools()`** — imported
  `FunctionTool` which doesn't exist. Now uses `Tool(callable)`.
- **`ChatMemoryBuffer.get()`** — inner pruning loop had no bound check, could
  decrement `message_count` below zero leading to wrong slicing. Now bounded.
- **`ChatMemoryBuffer`** — pre-filter clamping used to crash on multi-modal
  content (`list.split()` doesn't exist) and on `None` content. Now only
  clamps text portions.
- **`reason()` / `areason()`** — removed dead `<think>` regex (the LLM-level
  extractor handles tags now).

### Removed

- `Message.reasoning_content` — was never read by any provider, dead field.
- `ModelResponse.to_openai()` — provider-specific code didn't belong on a
  provider-neutral type.

[1.0.0]: https://github.com/gwenlake/gwenflow/releases/tag/v1.0.0
