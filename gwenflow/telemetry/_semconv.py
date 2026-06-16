# OpenInference attribute keys
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
LLM_MODEL_NAME = "llm.model_name"
LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
LLM_TOKEN_COUNT_PROMPT_CACHE_READ = "llm.token_count.prompt_details.cache_read"
LLM_TOKEN_COUNT_PROMPT_CACHE_WRITE = "llm.token_count.prompt_details.cache_write"
TOOL_NAME = "tool.name"
SESSION_ID = "session.id"
USER_ID = "user.id"

# Span kind
SPAN_KIND_LLM = "LLM"
SPAN_KIND_AGENT = "AGENT"
SPAN_KIND_TOOL = "TOOL"
SPAN_KIND_CHAIN = "CHAIN"

# Custom keys.
AGENT_NAME = "gwenflow.agent.name"  # for multi-agent flows
LLM_TOOL_CALLS = "llm.tool_calls"
