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
from gwenflow import ChatOpenAI, ChatAnthropic, ChatOllama

openai_llm   = ChatOpenAI(model="gpt-5-mini")
anthropic_llm = ChatAnthropic(model="claude-sonnet-4-6")
local_llm    = ChatOllama(model="llama3.2")
```

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

You can also pass tools explicitly at construction time using `FunctionTool.from_function`:

```python
from gwenflow import FunctionTool

agent = Agent(
    name="Finance Agent",
    llm=ChatOpenAI(model="gpt-5-mini"),
    tools=[FunctionTool.from_function(get_exchange_rate)],
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
