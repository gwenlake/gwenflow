# OpenInference attribute keys
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
LLM_MODEL_NAME = "llm.model_name"
LLM_PROVIDER = "llm.provider"
LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
LLM_TOKEN_COUNT_PROMPT_CACHE_READ = "llm.token_count.prompt_details.cache_read"
LLM_TOKEN_COUNT_PROMPT_CACHE_WRITE = "llm.token_count.prompt_details.cache_write"
LLM_TOKEN_COUNT_PROMPT_AUDIO = "llm.token_count.prompt_details.audio"
LLM_TOKEN_COUNT_COMPLETION_AUDIO = "llm.token_count.completion_details.audio"
LLM_TOKEN_COUNT_COMPLETION_REASONING = "llm.token_count.completion_details.reasoning"
LLM_FINISH_REASON = "llm.finish_reason"
TOOL_NAME = "tool.name"
SESSION_ID = "session.id"
USER_ID = "user.id"

# Retriever / RAG
RETRIEVAL_DOCUMENTS = "retrieval.documents"

# Embedding
EMBEDDING_MODEL_NAME = "embedding.model_name"

# Reranker
RERANKING_MODEL_NAME = "reranking.model_name"
RERANKING_TOP_K = "reranking.top_k"
RERANKING_OUTPUT_DOCUMENTS = "reranking.output_documents"

# Span kind
SPAN_KIND_LLM = "LLM"
SPAN_KIND_AGENT = "AGENT"
SPAN_KIND_TOOL = "TOOL"
SPAN_KIND_CHAIN = "CHAIN"
SPAN_KIND_RETRIEVER = "RETRIEVER"
SPAN_KIND_EMBEDDING = "EMBEDDING"
SPAN_KIND_RERANKER = "RERANKER"

# Custom keys.
AGENT_NAME = "gwenflow.agent.name"  # for multi-agent flows
AGENT_LLM_REQUESTS = "gwenflow.agent.llm_requests"  # LLM calls made in one run
AGENT_TOOL_CALLS = "gwenflow.agent.tool_calls"  # tools executed in one run
LLM_TOOL_CALLS = "llm.tool_calls"
