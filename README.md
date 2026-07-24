<div align="center">

![Logo of Gwenflow](https://raw.githubusercontent.com/gwenlake/gwenflow/refs/heads/main/docs/images/gwenflow.png)

**A framework for orchestrating applications powered by autonomous AI agents and LLMs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/v/release/gwenlake/gwenflow)](https://github.com/gwenlake/gwenflow/releases)

</div>

## Why Gwenflow?

Gwenflow, a framework designed by [Gwenlake](https://gwenlake.com), streamlines the creation of customized, production-ready applications built around Agents and Large Language Models (LLMs). It provides developers with the tools to integrate LLMs and Agents into efficient, scalable solutions.

**Key capabilities:**
- **Multiple LLM providers** — OpenAI, Anthropic, Azure, Mistral, Google, Ollama, DeepSeek
- **Autonomous agents** — agentic loop with tool use, memory, and structured output
- **Multi-agent flows** — DAG-based pipelines defined in code or YAML
- **RAG pipeline** — document readers, vector stores, and retrieval-augmented generation
- **Streaming** — sync and async streaming for all providers
- **Telemetry** — built-in OpenTelemetry tracing

## Installation

```bash
pip install gwenflow
```

Install the latest from the main branch:

```bash
pip install -U git+https://github.com/gwenlake/gwenflow.git@main
```

## Quick Start

```python
from gwenflow import ChatOpenAI

llm = ChatOpenAI(model="gpt-5-mini")
response = llm.invoke("Describe Argentina in one sentence.")
print(response.text)
```

## Multiple LLM Providers

Swap providers with a single import — the agent API is identical across all of them.

```python
from gwenflow import (
    ChatOpenAI, ChatAnthropic, ChatGoogle, ChatMistral,
    ChatDeepSeek, ChatOllama, ChatAzureOpenAI, ChatGwenlake,
)

openai_llm    = ChatOpenAI(model="gpt-5-mini")
anthropic_llm = ChatAnthropic(model="claude-sonnet-4-6")
google_llm    = ChatGoogle(model="gemini-2.5-flash")
mistral_llm   = ChatMistral(model="mistral-small-2603")
local_llm     = ChatOllama(model="llama3.2", base_url="http://localhost:11434/v1")
```

Anthropic, Mistral, and Google use their native SDKs (`anthropic`, `mistralai`, `google-genai`). OpenAI, Azure, DeepSeek, Ollama, and Gwenlake share the OpenAI Chat-Completions wire format.

## Reasoning / Thinking

Models that expose their internal reasoning (OpenAI gpt-5/o-series, Anthropic extended thinking, Mistral Magistral, Google Gemini 2.5, DeepSeek-R1, local models via Ollama) all surface it the same way: `response.thinking` and `AgentEventThinking` events on the stream.

```python
from gwenflow import ChatAnthropic, ChatGoogle, ChatOpenAI
from gwenflow.llms.openai_response import ResponseOpenAI

# OpenAI Responses API
llm = ResponseOpenAI(model="gpt-5-mini", reasoning_effort="medium", reasoning_summary="auto")

# Anthropic extended thinking
llm = ChatAnthropic(model="claude-opus-4-5", thinking={"type": "enabled", "budget_tokens": 1024})

# Google Gemini 2.5
llm = ChatGoogle(model="gemini-2.5-flash", thinking={"include_thoughts": True, "thinking_budget": 1024})

response = llm.invoke("If a train leaves Paris at 9:00 going 120 km/h ...")
print("Reasoning:", response.thinking)
print("Answer:", response.text)
```

For local models that emit `<think>...</think>` inline (qwen3, deepseek-r1 distills, gemma3), `ChatOllama` extracts them automatically into `ThinkingContent` parts.

## Multi-modal Input

Send images, audio, or PDFs alongside text in a single `Message`. The same `Message` works across providers — each adapter translates to the right wire format (`image_url` for OpenAI, `image` block for Anthropic, `inline_data` for Google, etc.).

```python
from gwenflow import ChatOpenAI, ChatAnthropic
from gwenflow.types import Message, TextContent, ImageContent, AudioContent, FileContent

# Image from URL, file, or raw bytes
msg = Message(role="user", content=[
    TextContent(content="What's in this image?"),
    ImageContent.from_url("https://example.com/cat.jpg"),
    # Or: ImageContent.from_path("/tmp/photo.jpg")
    # Or: ImageContent.from_bytes(png_bytes, media_type="image/png")
])

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke([msg])
print(response.text)

# Audio (gpt-4o-audio-preview, Gemini 2.5)
audio_msg = Message(role="user", content=[
    TextContent(content="Transcribe this clip."),
    AudioContent.from_path("/tmp/recording.wav"),
])

# PDF / documents (gpt-4o, Claude, Gemini)
pdf_msg = Message(role="user", content=[
    TextContent(content="Summarise this report."),
    FileContent.from_path("/tmp/report.pdf"),
])
```

Audio output from `gpt-4o-audio-preview` is surfaced via `response.audio` (a base64 `AudioContent` with the transcript when available).

## Agents

An `Agent` runs an agentic loop: it calls the LLM, dispatches tool calls, appends results, and repeats until done.

Any Python function can become a tool via the `@agent.tool` decorator — the function name, docstring, and type annotations are automatically converted into the schema the LLM receives.

```python
import requests
import json

from gwenflow import ChatOpenAI, Agent


agent = Agent(
    name="Finance Agent",
    instructions=[
        "Help users with exchange rates and stock prices.",
        "Answer in one sentence and mention the date if available.",
    ],
    llm=ChatOpenAI(model="gpt-5-mini"),
)


@agent.tool
def get_exchange_rate(currency_iso: str) -> str:
    """Get the current exchange rate for a given currency. Currency MUST be in ISO format."""
    try:
        response = requests.get("http://www.floatrates.com/daily/usd.json").json()
        return json.dumps(response[currency_iso.lower()])
    except Exception:
        return "Currency not found"


@agent.tool
def get_stock_price(ticker: str) -> str:
    """Get the latest stock price for a given ticker symbol."""
    try:
        import yfinance as yf
        return str(yf.Ticker(ticker).fast_info["lastPrice"])
    except Exception:
        return "Ticker not found"


print(agent.run("What's the exchange rate of the Euro?").content)
# As of January 10, 2025, the exchange rate for the Euro (EUR) is approximately
# 0.9709 EUR per 1 USD (last updated at 15:55 GMT).
```

You can also pass tools explicitly at construction time by wrapping the function with `Tool`:

```python
from gwenflow import Tool

agent = Agent(
    name="Finance Agent",
    llm=ChatOpenAI(model="gpt-5-mini"),
    tools=[Tool(get_exchange_rate)],
)
```

## Agents with Built-in Tools

```python
from gwenflow import Agent, ChatOpenAI
from gwenflow.tools import WikipediaTool

agent = Agent(
    name="Research Assistant",
    instructions=["Answer questions concisely using your tools when needed."],
    llm=ChatOpenAI(model="gpt-5-mini"),
    tools=[WikipediaTool()],
)

response = agent.run("Summarize the Wikipedia page about Winston Churchill.")
print(response.content)
```

## Coding Agent

`CodingAgent` is an `Agent` preset that ships with a sandboxed bundle of file, shell, and web-reader tools (`ReadFile`, `EditFile`, `WriteFile`, `Grep`, `Find`, `Ls`, `Shell`, `LocalFileWrite`, `WebsiteReader`). All file and shell operations are scoped to `base_dir`.

```python
from gwenflow import ChatOpenAI
from gwenflow.agents import CodingAgent

agent = CodingAgent(
    llm=ChatOpenAI(model="gpt-5-mini"),
    base_dir="./my_project",
)

response = agent.run(
    "Add a `greet(name)` function to utils.py that returns 'Hello, <name>!', "
    "then write a pytest test for it and run the test suite."
)
print(response.content)
```

The agent will list files, read the relevant ones, make targeted edits, run the tests via the shell, and report back. Extra tools passed via `tools=` are appended to the bundled set.

## Structured Output

Pass a Pydantic model as `response_model` and the agent returns a parsed object via `response.parsed`.

```python
from typing import List
from pydantic import BaseModel
from gwenflow import Agent, ChatOpenAI


class MovieReview(BaseModel):
    title: str
    year: int
    rating: float
    summary: str
    pros: List[str]
    cons: List[str]


agent = Agent(
    name="Movie Critic",
    instructions="You are a film critic. Analyse movies and return structured reviews.",
    llm=ChatOpenAI(model="gpt-5-mini"),
    response_model=MovieReview,
)

response = agent.run("Review the movie: Inception (2010)")
review: MovieReview = response.parsed

print(f"{review.title} ({review.year}) — {review.rating}/10")
print(f"  {review.summary}")
print(f"  Pros: {', '.join(review.pros)}")
print(f"  Cons: {', '.join(review.cons)}")
```

## Async Streaming

```python
import asyncio
from gwenflow import Agent, ChatOpenAI
from gwenflow.tools import WikipediaTool


async def main():
    agent = Agent(
        name="Research Assistant",
        instructions="Answer questions concisely using your tools when needed.",
        llm=ChatOpenAI(model="gpt-5-mini"),
        tools=[WikipediaTool()],
    )

    async for chunk in agent.arun_stream("Who invented the World Wide Web?"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


asyncio.run(main())
```

## Skills

Skills are reusable bundles of domain-specific instructions, optionally with companion Python scripts and resource files. The agent loads the compact skill list into its system prompt and only fetches the full instructions on demand via a `load_skill` tool — keeping token usage low when you have dozens of skills.

Layout on disk:

```
my_skills/
├── weather/
│   ├── SKILL.md        # YAML frontmatter + markdown instructions
│   ├── scripts/        # optional — Python functions auto-wrapped as Tools
│   │   └── tools.py
│   └── resources/      # optional — files readable via read_skill_resource
└── ...
```

```python
from gwenflow import Agent, ChatOpenAI
from gwenflow.skills import SkillsDirectory

skills = SkillsDirectory("./my_skills")
agent = Agent(
    name="Assistant",
    llm=ChatOpenAI(model="gpt-5-mini"),
    skills=skills.skills,
)
print(agent.run("What's the weather like in Paris?").content)
```

The agent automatically gets three management tools: `list_skills`, `load_skill`, and `read_skill_resource`. Any Python function defined in a skill's `scripts/` directory (with a docstring and type annotations) is auto-registered as a callable Tool.

## Multi-Agent Orchestration

Give an agent a `team` of specialist agents and it becomes an orchestrator. Each teammate is exposed to the orchestrator's LLM as a handoff tool named `ask_<slug>`, with the teammate's `description` as the tool description. The orchestrator decides who to delegate to, calls them, and synthesises the final answer.

```python
from gwenflow import Agent, ChatOpenAI, Tool


def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny and 22C in {city}"


def get_population(city: str) -> str:
    """Return the population of a city."""
    return {"Paris": "2.1 million", "Tokyo": "13.9 million"}.get(city, "unknown")


weather_agent = Agent(
    name="weather_agent",
    description="Knows the current weather in any city.",
    llm=ChatOpenAI(model="gpt-5-mini"),
    tools=[Tool(get_weather)],
)

demographics_agent = Agent(
    name="demographics_agent",
    description="Knows the population of cities.",
    llm=ChatOpenAI(model="gpt-5-mini"),
    tools=[Tool(get_population)],
)

orchestrator = Agent(
    name="orchestrator",
    instructions=[
        "You manage a team of specialist agents.",
        "Delegate sub-tasks to the right teammate via the ask_* tools.",
        "Then synthesise a final answer for the user.",
    ],
    llm=ChatOpenAI(model="gpt-5-mini"),
    team=[weather_agent, demographics_agent],
)

response = orchestrator.run(
    "One-sentence travel briefing for Paris: weather and population."
)
print(response.content)
# Travel briefing — Paris: currently sunny and about 22°C,
# and the city proper has roughly 2.1 million residents.
```

Each teammate's `name` becomes the handoff tool slug and its `description` tells the orchestrator when to delegate to it — write descriptions like a job blurb.

## Multi-Agent Flows

Chain agents together in a DAG. Each node receives the output of its upstream nodes as input.

**Code-based:**

```python
from gwenflow import Agent, ChatOpenAI

llm = ChatOpenAI(model="gpt-5-mini")

joker = Agent(
    name="Joker",
    instructions="You are a comedian. Tell short, funny jokes.",
    llm=llm,
)

explainer = Agent(
    name="Explainer",
    instructions="You are a professor. Explain why jokes are funny, citing humour theory.",
    llm=llm,
)

joke = joker.run("Tell me a joke about programmers.").content
explanation = explainer.run(f"Explain why this joke is funny:\n\n{joke}").content

print(f"Joke:\n{joke}\n")
print(f"Why it's funny:\n{explanation}")
```

**YAML-based pipeline:**

```yaml
# flows/news_summary.yaml
name: News Summary Pipeline

nodes:
  - name: fetch_news
    type: gwenflow.httpRequest
    parameters:
      url: https://hacker-news.firebaseio.com/v0/topstories.json
      method: GET

  - name: summarize
    type: gwenflow.Agent
    parameters:
      model: openai/gpt-5-mini
      instructions: You are a tech journalist who writes concise summaries.
      task: "Here are today's top Hacker News story IDs:\n\n{fetch_news}\n\nWhat topics are trending?"

connections:
  fetch_news:
    - - node: summarize
        type: main
        index: 0
```

```python
from gwenflow import FlowRunner

runner = FlowRunner("flows/news_summary.yaml")
runner.run()
```

## Telemetry & Observability

Gwenflow ships with optional, standards-based tracing built on [OpenTelemetry](https://opentelemetry.io/) and the [OpenInference](https://github.com/Arize-ai/openinference) semantic conventions, so traces flow into **any** OTLP-compatible backend (Jaeger, Grafana Tempo, Arize Phoenix, Langfuse, Honeycomb, …).

It is **off by default and zero-overhead** until you enable it: when the telemetry packages aren't installed (or no `Telemetry()` is configured), every agent / LLM / tool / flow call is a transparent pass-through.

Install the extra and turn it on:

```bash
uv add "gwenflow[telemetry]"
```

```python
from gwenflow import Agent, ChatOpenAI, Telemetry

# One line enables OTLP export. Defaults to http://localhost:4318 (OTLP/HTTP).
Telemetry(organization="gwenflow")

agent = Agent(name="Researcher", llm=ChatOpenAI(model="gpt-4o-mini"))
agent.run("What is the capital of France?")
```

Traces are organized so a backend can drill down from a whole organization to a single operation — **organization → project → session → trace → span**:

- **Organization** — `Telemetry(organization=...)`, reported as the standard OTel resource attribute `service.name` (used natively by Jaeger, Phoenix Arize). Fixed per process.
- **Per-request attributes** — attach anything to every span in a block with `tracer.context(metadata={...})`: a project, an environment, a session, … Nothing is privileged or required.
- **Trace** — one top-level call (`agent.run`, `flow.run`); spans nest automatically into an agentic tree (`Flow → Agent → LLM / Tool`) capturing model, token usage, latency, tool calls and status.

```python
from gwenflow.telemetry import tracer

# Fully flexible — every key is yours; nothing is assumed or required.
with tracer.context(metadata={"project.id": "support-bot", "session.id": "conversation-123"}):
    agent.run("...")

# Opt-in for the OpenInference session.id / user.id conventions (all args optional):
with tracer.session("conversation-123", user_id="user-42", metadata={"project.id": "support-bot"}):
    agent.run("...")
```

`metadata` keys are plain span attributes (blocks nest, inner keys win), so they stay compatible with any backend: your own tooling can build an `organization → project → session → trace → span` hierarchy from them, Jaeger shows them as filterable tags, and OpenInference tools (Phoenix Arize) recognise `session.id` / `user.id` if you choose to use those conventional keys. `session.id` is not a core OTLP concept — it is offered as opt-in, never imposed. (Resource-level fixed attributes can also be added via the standard `OTEL_RESOURCE_ATTRIBUTES` env var.)

## More Examples

Explore practical implementations and advanced scenarios in the [`examples/`](https://github.com/gwenlake/gwenflow/tree/main/examples) directory.

## Contributing to Gwenflow

We are very open to the community's contributions - be it a quick fix of a typo, or a completely new feature! You don't need to be a Gwenflow expert to provide meaningful improvements.

## Compliance & Licensing

To ensure **gwenflow** remains commercially friendly and safe for enterprise exploitation, we strictly monitor our dependency tree. We primarily allow permissive licenses (MIT, Apache-2.0, BSD) and systematically avoid "Strong Copyleft" licenses (such as AGPL or GPL) that could impact your source code.

### License Audit

We use `licensecheck` to automate this verification. You can audit the current dependencies locally by running:

```bash
uv run licensecheck --recursive
```
