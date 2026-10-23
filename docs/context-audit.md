# Context management audit (step 0)

State of `main` at commit 407c27a, i.e. after the first context-management
patch (tolerant model lookup, `context_window`, `token_scale` calibration,
pass 0 clearing, `max_tool_output_tokens`). Every finding has a test in
`tests/test_context_audit.py`: confirmed behaviours pass, defects are
`xfail(strict=True)` and name the step that fixes them.

## 1. Context window

| Case | Today | Verdict |
|---|---|---|
| Known model | `MODELS[...]["context_window"] * 0.75` | correct |
| Served under another name (`Qwen/Qwen3.6-27B-AWQ`, `mistral-small-3.2-24b-instruct-2506`) | tolerant, case-insensitive lookup | correct |
| Unknown model | 128,000 × 0.75 = 96,000, one warning | tolerable, but a vLLM server at 32k overflows silently |
| `context_window=<int>` | overrides the table | correct |
| `context_window="auto"` | reads `/v1/models` on OpenAI-compatible servers | correct |
| `fallback_context_window` | does not exist | missing (step 1) |
| `agent.llm = other` after construction | `history.token_limit` keeps the old value: the budget is frozen in `Agent.__post_init__` | defect (step 1) |

The default is 128k, with no fallback parameter. Any default is wrong for a
small local server: the fix is an explicit fallback, not a different default.

## 2. What the budget counts

Counted: system prompt, tool schemas (`reserved_tokens`, refreshed per run,
`final_result` included), the rolling summary message, tool-call arguments,
`tool_call_id`, thinking parts and their signatures, images / audio / files
(conservative estimates), validation retry messages (they are user messages).

Not counted: nothing structural. The weak spot is not an omission but the
estimate itself (see 3) and the fact that the output is only reserved through
the global 0.75 ratio, never explicitly.

## 3. Accuracy of the local estimate

Live measurement (2026-10-23) of tiktoken `cl100k_base` against the prompt
tokens Mistral billed, same messages, `max_tokens=1`:

| Content | local | mistral-small-latest | ratio | mistral-large-latest | ratio |
|---|---|---|---|---|---|
| French prose | 358 | 302 | 0.84 | 290 | 0.81 |
| Python code | 521 | 562 | 1.08 | 550 | 1.06 |
| JSON records | 1,421 | 1,682 | 1.18 | 1,670 | 1.18 |
| Mixed | 2,258 | 2,487 | 1.10 | 2,475 | 1.10 |

The error changes sign with the content type (−19 % on French, +18 % on
JSON), so a single global `token_scale` can only track the mix of the moment.
Anchoring on the provider's `input_tokens` for everything already sent, and
estimating only the messages added since, bounds the error to the last few
messages (step 1).

## 4. The live turn

`ChatSummaryMemoryBuffer` anchors on the last user message and never
summarizes anything after it. For one instruction followed by 50 tool calls:

1. pass 0 stubs results of repeated calls, then the oldest results beyond the
   3 most recent, until the window fits;
2. still over budget, the oldest messages of the turn are dropped outright:
   the assistant's intermediate notes and plans vanish with no trace, the
   summary stays empty and the summarizer is never called.

Confirmed by `test_live_turn_overflow_clears_results_then_drops_oldest_messages`;
wanted behaviour in `test_intermediate_notes_of_a_long_turn_survive_in_some_form`
(steps 2–3).

## 5. Tool call / result pairing

Correct today. The window never starts on a `tool` message (orphan skip in
`_prune`), stubs keep their `tool_call_id`, and the summary boundary is a
group boundary. Parallel groups of 3 calls survive every cut tried
(`test_pruning_keeps_parallel_groups_whole`). The invariant is now explicit
in `assert_tool_pairs`, reused by the later steps.

## 6. The initial instruction

Survives only while it is the *last* user message. A schema-validation retry
appends a user message, the anchor moves to it, and the task is pruned or
summarized away (`test_task_survives_a_validation_retry_message`, step 2:
pin the first user message).

## 7. Prefix stability

`get()` recomputes the window on every request and pass 0 clears results
oldest-first "until it fits": each new tool pair clears exactly one more
result, so the sent prefix changes at every request and the provider's prompt
cache is invalidated every time
(`test_prefix_currently_changes_on_every_request`). Wanted: clear in batches,
from a high threshold down to a lower target, and persist the rewrite
(`test_prefix_is_stable_between_two_compaction_batches`, step 2).

## 8. Cost and failure of the summary

- The summarizer's tokens are recorded in `history.usage` and shown by
  `/cost`, but neither `AgentResponse.usage` nor `Agent.total_usage` include
  them (`test_summary_tokens_are_part_of_the_run_usage`, step 3).
- A summarizer that raises never fails the run: the pointer does not advance
  and the window degrades to dropping. Correct.
- A summarizer that returns an empty summary is retried on every `compact()`
  with no cap: one wasted LLM call per turn for the rest of the session
  (`test_empty_summary_is_retried_a_bounded_number_of_times`, step 3).
