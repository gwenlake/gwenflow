# Changelog

All notable changes to gwenflow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `ChatOpenAI._parse_response` (and so `ChatGwenlake`, `ChatDeepSeek`, `ChatOllama`...)
  crashed with `AttributeError: 'ReadFileTool' object has no attribute 'get'` whenever
  the model answered in plain text while tools that inherit `BaseTool` but not `Tool`
  were on offer — every `CodingAgent` run. The text-tool-call recovery now accepts any
  `BaseTool`. Only the OpenAI family had the bug (the other providers have no text
  recovery); every provider now types its tools as `BaseTool`, and
  `tests/test_llm_base_tools.py` pins schema building and plain-text parsing with
  BaseTool subclasses on offer for all nine LLM classes.
- The context window of a model served under another name was silently the 128k
  default: `lookup_model_info` now strips provider / organisation prefixes, ignores
  case and falls back on the longest known name followed by a separator
  (`mistral-small-3.2-24b-instruct-2506` -> `mistral-small-3.2-24b`). Matching ignores
  case end to end (`qwen/qwen3.6-35B-A3B` -> `qwen3.6-35b-a3b`), and a test pins the
  table keys to lowercase. An unknown model logs a warning once.
- `MODELS` refreshed (2026-10-23): GPT-6 Astra / Sol / Luna and GPT-6.1 Sol, Claude
  Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5 and Mythos 5.1, Gemini 4 Argon, 3.8 Flash,
  3.6 Flash and 3.5 Flash-Lite, DeepSeek `deepseek-flash` / `deepseek-v4.1-flash` (dated
  builds such as `deepseek-v4-flash-0731` resolve through the prefix rule), Grok 4.6 and
  4.7, Mistral Large 4 (`mistral-large-4`), the Mistral `-latest` aliases, GLM 5.3 / 5.3
  Flash / 5.3 FlashX (Z.ai and via Mistral),
  Qwen3.8 Max / Flash and the Qwen3.8 open weights (27B, Flash-Next, 2.4T-A95B). Mistral
  windows and reasoning flags now match the `/v1/models` listing: Medium 3.5, Small 4
  and Ministral 14B / 8B are 262,144 tokens, Medium 3.5 and Small 4 are reasoning models.

### Added

- `context_window` on every LLM: an int overrides the table (a vLLM server started
  with a smaller `--max-model-len`), `"auto"` reads it from the server's `/models`
  listing (`max_model_len`, `context_length` or `context_window`) on OpenAI-compatible
  clients and falls back on the table.
- Token calibration: after each call the agent hands the provider's real prompt token
  count to the memory buffer (`observe_prompt_tokens`), which scales its budget
  (`token_scale`). tiktoken's `cl100k_base` under-counts for Mistral, Qwen or Gemma,
  so the history overflowed before compaction kicked in.
- Pass 0 of `ChatMemoryBuffer`, model-free and only when the history does not fit:
  results of a call repeated later with the same arguments are replaced by a stub,
  then the oldest tool results beyond `keep_recent_tool_results` (3) are replaced by
  a one-line stub naming the call, oldest first, until the window fits. It also
  works inside the live turn, where the summary never applies, and it spares
  `ChatSummaryMemoryBuffer` an LLM call whenever clearing is enough.
- `Agent.max_tool_output_tokens`: a longer tool result is set aside; the model gets
  its beginning, a reference and a `read_tool_output(ref, offset, length)` tool to
  page through the rest. Off by default.
- Context measured on the provider's own counts. The memory buffer anchors its size on
  the provider's last reported usage (`observe_usage`: prompt + completion tokens,
  which cover the system prompt, tool schemas and files) and estimates only the
  messages stored since; `used_tokens()` / `context_usage()` expose the result with
  the window, the limit and whether the window is known. The agent anchors after
  every reply, on all four run loops.
- `ChatMemoryBuffer.max_fraction` (exclusive with `token_limit`): the budget as a
  share of the model's context window, resolved on every request against the LLM
  actually called. **Default behaviour change:** the buffer an agent creates for
  itself now uses `max_fraction=0.75` instead of a `token_limit` frozen at
  construction, so replacing `agent.llm` moves the budget. A buffer passed
  explicitly is untouched.
- `fallback_context_window` on every LLM: used only when the model is unknown
  (`context_window` still overrides everything). `ChatBase.context_window_info()`
  says where the value comes from and whether it is assumed.
- `gwenflow.utils.tokens.hf_tokenizer(model)`: a `tokenizer_fn` backed by the Hugging
  Face tokenizer of the served model (optional `transformers` dependency), for
  buffers that must count like a Mistral, Qwen or Gemma server does.
- Compaction in tiers (`gwenflow.memory.compaction`).
  `ChatMemoryBuffer.compact()` now rewrites the stored history, cheapest
  strategy first, re-measuring after each one and stopping once the history is back
  under `target()`: `DeduplicateReads` (latest read of a file wins; on every request),
  `ClampOversized` (head + tail of one oversized answer, tool result or tool argument),
  `ClearToolResults` (`keep_pairs`, `exclude_tools`, `clear_tool_inputs`,
  `min_clear_tokens`), `SummarizeOlder` (the summary tier of `ChatSummaryMemoryBuffer`,
  now inside the live turn too, with a recent tail kept verbatim) and `SlidingWindow`.
  Strategies never mutate the messages they receive; a tier that raises is skipped and
  the sliding window always closes the chain. `strategies=[...]` replaces the tiers,
  `default_strategies()` builds them.
  **Default behaviour changes:** compaction is persisted in `messages` (the originals
  of what was altered are kept in `compactions[-1].removed`), the plain
  `ChatMemoryBuffer` compacts too (without summary), and compaction works in batches:
  it triggers at `max_fraction` (0.85 of the window for the agent's default buffer)
  and stops at `target()` (70 % of the limit by default, `target_fraction` /
  `target_tokens` to set it), so the prefix sent to the model, and the provider's
  prompt cache, stay stable between two batches. `get()` is unchanged as a last
  resort when the stored history still does not fit.
- Tool call / result pairing is guaranteed by construction (groups, see
  `tool_groups`), checked by `validate_tool_pairs`. The first user message, the last
  one (the live task, also pinned by the agent for the run) and any message marked
  with `compaction.pin()` survive every strategy. The task now survives a
  schema-validation retry.
- `/compact` works on every buffer and reports the tiers applied, the message and
  token counts before and after.
- A better summary. The summarizer gets a `<message>`-tagged transcript where every
  tool result is capped (`tool_result_max_chars`, 500) and text imitating the tags is
  escaped; the previous summary is handed back as a `<previous-summary>` block to
  update (keep what is still true, drop what is stale, add the new); user messages can
  be kept verbatim beside the summary (`keep_user_messages`, each capped at
  `keep_user_messages_max_chars`); an empty summary is retried once, then the sliding
  window takes over for that batch. The summary message now says it is second-hand,
  and a sliding-window cut leaves a one-line receipt (`SlidingWindow(receipt=True)`)
  so the model does not assume continuity.
- `Agent(summarizer=...)`: a (cheaper) model for the summaries. **Default behaviour
  change:** the summaries' tokens are now folded into `AgentResponse.usage` and
  `Agent.total_usage` of the run that triggered them (`/cost` shows the compaction
  block as a breakdown, "included above"), and a delegated worker's compaction is no
  longer counted twice.
- Observability. `Agent.context_usage()` / `ChatMemoryBuffer.context_usage()` return a
  `ContextUsage` (tokens used, window, limit, fraction, whether the window is known,
  whether the count is anchored on the provider); `Agent(on_context_usage=...)` is
  called with it before every request. `Agent(warn_near_limits=True)` adds a
  request-scoped "Context notice" to the system prompt when `warning_threshold` of the
  budget or of `max_turns` is reached (URGENT) and when compaction is imminent or
  `critical_remaining_turns` are left (CRITICAL), so the model concludes instead of
  seeing its history rewritten; nothing is written in the history. Each compaction
  that triggers emits a `compact_messages` span with the strategies applied and the
  message and token counts before and after, never the content
  (`tracer.span(name)` is the generic helper behind it).

## [1.3.0] — 2026-10-12

### Added

- **Long-term markdown memory** (`gwenflow.memory`). Files live under a virtual
  `/memories` root in a `MemoryStore` (async protocol with a sync `read` hook,
  optimistic `expected_version` locking, path validation that rejects traversal);
  `InMemoryMemoryStore` and `LocalFileMemoryStore` are the reference backends.
  `MemoryToolset` mirrors `SkillsToolset`: always-loaded core files rendered in a
  `## Memory` block, an index of topic files (path + frontmatter description) read
  on demand with `read_memory`, and the editing tools `memory_edit`, `memory_write`,
  `write_topic_file` (frontmatter generated) and, for curators, `memory_move`.
  The toolset works on an in-process working copy (`aload()` before the run,
  `aflush()` after): tools stay sync, every dirty file is flushed independently
  and this run's edits are replayed on a fresh copy after a version conflict.
  `MemoryConsolidator` rewrites files from a turn's messages with one cheap LLM
  call and returns the entities seen (`merge_entities` keeps them additively);
  `MemoryCurator` runs an `Agent` that merges candidates, deduplicates and
  archives unread files. Prompt texts are grouped in `gwenflow.memory.long_term.prompts`. New `Agent.memory` field; the block is placed after the
  JSON schema and before knowledge/context so a stable prompt prefix stays
  cacheable. Not rendered when `system_prompt` is set (same as skills).
- **Automatic recall** (`gwenflow.memory.long_term.recall`): `MemoryToolset.aload(query=...)`
  matches the request against the topic index (file names, descriptions,
  identifiers) and renders the matching files in the prompt, without an LLM
  call; `read_memory` remains the fallback. Generic function words only are
  ignored; applications add their domain words with `recall_stopwords`.
  `MemoryToolset(tools=...)` exposes a subset of the tools (a conversational
  agent typically keeps `read_memory` and `memory_edit`), and the protocol
  follows what is exposed. `MemoryConsolidator` updates the recalled topic
  files and may create new ones under `creatable_prefixes` (never a file it was
  not shown, see `known_paths`); `ConsolidationResult.failed` tells a failed
  call from "nothing to change". A topic file's `type` defaults to its directory.
- Example: `examples/agent_long_term_memory.py` walks through the tools, an
  agent with memory, the consolidator and the curator on the same files.
- `gwenflow.utils.markdown.parse_frontmatter` / `render_frontmatter`; `Skill`
  delegates to it. `gwenflow.memory.base.render_messages_as_text` (lifted from
  `ChatSummaryMemoryBuffer`, with per-message truncation).

## [1.2.5] — 2026-10-12

### Fixed

- **Structured output + tools on OpenAI-compatible servers.** Sending
  `response_format` (JSON schema) together with `tools` works with OpenAI, but
  many OpenAI-compatible servers (vLLM, DeepSeek, Scaleway, Ollama...) apply the
  JSON grammar from the first token, so the model could never emit a tool call:
  it answered in JSON right away and hallucinated the data it should have
  fetched. The agent now withholds `response_format` while tools are on offer
  and instead exposes the schema as a `final_result` tool (pydantic-ai's
  approach): calling it ends the run with the validated arguments as
  `response.parsed`, invalid arguments are fed back to the model as the tool
  result, and other tool calls in the same turn are not executed. A model that
  "calls" the schema by its own title (Qwen, Llama...) is handled the same way.
  Plain JSON text is still accepted; when it fails validation the retry turn
  forces a `final_result` call through a named `tool_choice`, so
  `response_format` is never sent together with tools. Agents without tools
  are unchanged. A named `tool_choice` in the OpenAI shape
  (`{"type": "function", "function": {"name": ...}}`) is now translated for
  Anthropic, Google and the OpenAI Responses API; it was passed through
  verbatim before. `ChatAnthropic` also honours string `tool_choice` values
  (`"required"`/`"any"` → `{"type": "any"}`, `"none"` → `{"type": "none"}`)
  instead of always sending `{"type": "auto"}`.

## [1.2.4] — 2026-09-16

### Fixed

- **Streamed tool calls against non-OpenAI servers.** `ChatOpenAI.stream()` /
  `astream()` accumulated delta fragments onto a `ToolCall` built from the first
  fragment and matched fragments by `id`. vLLM (and other OpenAI-compatible
  servers) opens a tool call with `arguments=None` where OpenAI sends `""`, so
  the next fragment raised `TypeError: unsupported operand type(s) for +=:
  'NoneType' and 'str'` — surfaced as `RuntimeError: Error in calling openai
  API: ...`. Every request that triggered a tool failed. Fragments are now keyed
  by `index`, which is what the streaming protocol uses to tell parallel calls
  apart. This also fixes three related cases: a fragment with no `function`
  object (it is `Optional` in the SDK), a server that never sends an `id` (two
  calls were silently merged into one), and a server that repeats `id`+`name` on
  every fragment (the name was duplicated).
- **Dead `except` clauses in the Gwenlake embeddings and reranker.** Both caught
  `requests.exceptions.RequestException` around a call made with an
  `httpx.Client`, so a network failure propagated raw instead of becoming the
  intended `ValueError`. The exception wrapped by tenacity's `RetryError`
  therefore changes from `ConnectError` to `ValueError`.

### Changed

- **`import gwenflow` no longer loads every vendor SDK**, dropping resident
  memory from ~148 MiB to ~56 MiB per process. Backends, tools, readers and
  `CodingAgent` resolve lazily (PEP 562); `networkx` and the S3 helpers are
  imported where they are used. No dependency was removed and no public name
  moved — `from gwenflow import ChatAnthropic` works exactly as before, the SDK
  simply loads on first access. Note that `from gwenflow import *` resolves
  every name in `__all__` and therefore forfeits the saving; use explicit
  imports.

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
