# Agent - Examples and Use Cases

This page demonstrates various examples of using agents in **Gwenflow** to interact with different tools, including local functions, web search, finance data, and memory management.

This Markdown shows various examples; the full code lives in the individual Python files in this directory.

> **Note:** the outputs in this document are illustrative. Actual numbers (prices, weights, headlines) will reflect live market data at the time you run the examples.

---

## ▶️ Running the examples

All examples are runnable via `uv run`. From the project root:

```bash
# 1. Make sure dependencies are installed
uv sync --all-groups --frozen

# 2. Some finance examples need yfinance (not in default deps)
uv pip install yfinance

# 3. Provide an API key for whichever provider the example uses
#    (most finance examples below use Anthropic):
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
# or:  export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run any example
uv run python examples/agent_finance_portfolio.py
```

> **Tip:** prefer `uv pip install yfinance` over `uv run --with yfinance ...` — `--with` can create an isolated overlay that doesn't see the editable `gwenflow` install.

---

## 📦 Function Agents

You can provide agents with specific tools that perform targeted tasks. Use the `@agent.tool` decorator to register any Python function — the name, docstring, and type annotations are automatically converted into the schema the LLM receives.

### Example: Bike Availability Tool — [`agent_bikes.py`](agent_bikes.py)

```python
import requests
import json

from gwenflow import Agent, ChatOpenAI

agent = Agent(
    name="City Data Agent",
    llm=ChatOpenAI(model="gpt-5-mini"),
    instructions=[
        "Provide bike data for specified stations.",
        "Respond in JSON format only."
    ],
)

@agent.tool
def get_bike_availability(station_name: str = None) -> str:
    """Get bike availability from Rennes Métropole stations. Omit station_name to list all stations."""
    url_station = f"https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-des-stations-le-velo-star-en-temps-reel/records?where=nom=%22{station_name}%22&limit=1"
    url_general = "https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-des-stations-le-velo-star-en-temps-reel/records"
    try:
        response = requests.get(url_station if station_name else url_general)
        response.raise_for_status()
        return json.dumps(response.json().get('results', []))
    except Exception as e:
        return json.dumps({"error": str(e)})

queries = [
    "What is the bike availability at Oberthur station?",
    "List all bike docking stations."
]

for query in queries:
    response = agent.run(query)
    print(response.content)
```

---

## 🌐 Web Search Agents

Web search agents can explore and extract data from the internet, making them ideal for content generation and data analysis.

### Example: SEO Content Generator — [`agent_websearch_seo.py`](agent_websearch_seo.py)

```python
from gwenflow import Agent, ChatOpenAI
from gwenflow.tools import TavilyWebSearchTool, WebsiteReaderTool, PDFReaderTool

task = """
Create an outline for an article that will be 2,000 words on the keyword 'Best SEO prompts' for a company working in the sector '{sector}', based on the top 10 results from Google.
Include every relevant heading possible, FAQs, and LSI keywords. Provide recommended external links with anchor text.
Split the outline into part 1 and part 2.
"""

agent = Agent(
    name="SEO-Prompt",
    llm=ChatOpenAI(model="gpt-5-mini"),
    instructions=[
        "You are a content strategist.",
        "Write the content in markdown format.",
        "Ensure to structure the outline clearly into sections and subsections."
    ],
    tools=[TavilyWebSearchTool(), WebsiteReaderTool(), PDFReaderTool()]
)

response = agent.run(task.format(sector="Data Analytics"))
print(response.content)
```

---

## 🧠 Memory Management

Agents can be initialized with memory to maintain context across interactions, allowing for more natural and context-aware conversations.

### Example: Medical Assistant with Memory — [`agent_memory.py`](agent_memory.py)

```python
from gwenflow import Agent, ChatOpenAI
from gwenflow.memory import ChatMemoryBuffer

messages = [
    {"role": "user", "content": "Tell me about Alzheimer"},
    {"role": "assistant", "content": "Alzheimer is a neurodegenerative disease."}
]

memory = ChatMemoryBuffer(token_limit=300)
memory.add_messages(messages)

agent = Agent(
    name="Medical Assistant",
    llm=ChatOpenAI(model="gpt-5-mini"),
    instructions=[
        "Provide medical explanations in simple language.",
        "Maintain context across multiple queries."
    ],
    history=memory
)

response = agent.run("What did we discuss previously?")
print(response.content)
```

---

## 💼 Finance & Asset Management

Five examples that build up from a simple market-data Q&A agent to multi-agent investment workflows. All examples use the Yahoo Finance tool suite (`YahooFinanceSearch`, `YahooFinanceStock`, `YahooFinanceNews`, `YahooFinanceScreen`) and `ChatAnthropic` (Claude) as the LLM. Swap the LLM for any other `ChatBase` subclass if you prefer (`ChatOpenAI`, `ResponseOpenAI`, `ChatGwenlake`, …).

### 1. Live market Q&A — [`agent_finance.py`](agent_finance.py)

A single agent answers ad-hoc questions about quotes, news, and comparisons.

```python
from gwenflow import Agent, ChatOpenAI
from gwenflow.tools.yahoofinance import YahooFinanceNews, YahooFinanceSearch, YahooFinanceStock

agent = Agent(
    name="Finance Analyst",
    instructions=[
        "You are a financial analyst.",
        "Use your tools to retrieve up-to-date stock data and news.",
        "Always mention the ticker symbol and the data source date when available.",
    ],
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[YahooFinanceSearch(), YahooFinanceStock(), YahooFinanceNews()],
)

print(agent.run("What is the current stock price of Apple?").content)
```

Illustrative output:
```
Q: What is the current stock price of Apple?
A: Apple Inc. (AAPL) is currently trading at $189.84 as of the latest Yahoo Finance quote.
   Day change: +1.27%. The 52-week range is $164.08 – $199.62.
```

---

### 2. Portfolio snapshot (structured output) — [`agent_finance_portfolio.py`](agent_finance_portfolio.py)

Pass the agent a list of holdings; it calls `YahooFinanceStock` for each, computes weights, and returns a typed `PortfolioSnapshot` via `response_model`.

```python
class PortfolioSnapshot(BaseModel):
    total_value: float
    positions: List[Position]
    top_winner: str
    top_loser: str
    commentary: str

agent = Agent(
    name="Portfolio Analyst",
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[YahooFinanceStock()],
    response_model=PortfolioSnapshot,
)

response = agent.run("Build a portfolio snapshot for: 50 AAPL, 30 MSFT, 20 GOOGL, 15 NVDA, 40 JPM")
snapshot: PortfolioSnapshot = response.parsed
```

Illustrative output:
```
Total portfolio value: $59,420.50

Ticker     Shares      Price          Value   Weight    Day %
AAPL           50  $   189.84  $    9,492.00   15.97%   +1.27%
MSFT           30  $   421.18  $   12,635.40   21.27%   -0.42%
GOOGL          20  $   178.95  $    3,579.00    6.02%   +0.81%
NVDA           15  $   903.45  $   13,551.75   22.81%   +2.94%
JPM            40  $   200.40  $    8,016.00   13.49%   -0.13%

Top winner: NVDA   Top loser: MSFT

The portfolio leans heavily toward US large-cap technology (NVDA + MSFT + AAPL + GOOGL ≈ 66%),
with a single financial-sector hedge (JPM). NVDA leads the day on continued AI-driven momentum;
MSFT is modestly weaker. Consider trimming NVDA if its weight is uncomfortable post-rally.
```

---

### 3. Investment research team (multi-agent handoff) — [`agent_finance_research_team.py`](agent_finance_research_team.py)

A `PortfolioManager` orchestrator delegates to two specialists via the `team=[...]` parameter. Gwenflow auto-generates handoff tools (`ask_equityanalyst`, `ask_newsanalyst`) so the manager calls them like any other tool.

```python
equity_analyst = Agent(
    name="EquityAnalyst",
    description="Equity research analyst. Use for fundamentals: price, P/E, margins, growth, sector.",
    llm=ChatAnthropic(model="claude-haiku-4-5"),
    tools=[YahooFinanceSearch(), YahooFinanceStock()],
)
news_analyst = Agent(
    name="NewsAnalyst",
    description="News & catalyst analyst. Use for recent headlines, sentiment, and near-term catalysts.",
    llm=ChatAnthropic(model="claude-haiku-4-5"),
    tools=[YahooFinanceNews()],
)
portfolio_manager = Agent(
    name="PortfolioManager",
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    team=[equity_analyst, news_analyst],
)

response = portfolio_manager.run(
    "Should we add NVDA to a balanced US equity portfolio? "
    "Get fundamentals from the equity analyst and recent news from the news analyst, then decide."
)
print(response.content)
```

Illustrative output:
```
======================================================================
Evaluating NVDA
======================================================================
Recommendation: HOLD

Thesis: NVIDIA remains the dominant supplier of AI accelerators with ~80% data-center
GPU share. Fundamentals are strong (gross margins ~73%, FY revenue growth >100% YoY,
forward P/E ~36 on consensus). Near-term news flow is constructive — major hyperscaler
capex commitments, but a recent China export-control headline introduces overhang.
For a balanced portfolio already meaningfully exposed to US large-cap tech, adding more
NVDA at current valuation increases concentration risk without a clear margin of safety.

Top risks:
  1. Regulatory: tighter US/China semiconductor controls could compress addressable market.
  2. Cyclicality: hyperscaler AI capex is at record levels; any pause in 2026 orders
     would hit revenue and multiple simultaneously.
```

The orchestrator made two tool calls (one to each teammate), the teammates ran their own Yahoo Finance tools, and the manager synthesised the final answer — visible in `response.messages` and `response.usage.tool_calls`.

---

### 4. Stock screener with streaming — [`agent_finance_screener.py`](agent_finance_screener.py)

Composes two tools (`YahooFinanceScreen` for the screen, `YahooFinanceStock` for enrichment) and streams the analyst's narrative chunk-by-chunk with `agent.run_stream`.

```python
agent = Agent(
    name="Screener Analyst",
    instructions=[
        "Step 1: call YahooFinanceScreen with the requested operator and values to get a ticker list.",
        "Step 2: pick the top 5 tickers from the screen and call YahooFinanceStock on each.",
        "Step 3: write a concise markdown table with: Ticker | Name | Sector | Mkt Cap | P/E | 52w High.",
    ],
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[YahooFinanceScreen(), YahooFinanceStock()],
)

for chunk in agent.run_stream(
    "Run a large-cap US equity screen: market cap above $200B. "
    "Operator='GT', values=['intradaymarketcap', 200000000000]. "
    "Then enrich the top 5 with fundamentals and produce the report."
):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

Illustrative streamed output:
```
[streaming]: Screen returned 28 tickers above the $200B threshold. Pulling fundamentals
for the top 5 by average daily volume…

| Ticker | Name             | Sector             | Mkt Cap   | P/E   | 52w High |
|--------|------------------|--------------------|-----------|-------|----------|
| AAPL   | Apple Inc.       | Technology         | $2.94T    | 31.2  | $199.62  |
| MSFT   | Microsoft Corp.  | Technology         | $3.13T    | 36.4  | $468.35  |
| NVDA   | NVIDIA Corp.     | Technology         | $2.22T    | 36.0  | $974.00  |
| AMZN   | Amazon.com Inc.  | Consumer Cyclical  | $1.82T    | 52.7  | $189.77  |
| GOOGL  | Alphabet Inc.    | Communication      | $2.20T    | 25.1  | $182.49  |

- Heavy mega-cap technology tilt — 4 of the top 5 sit in the Information Technology
  or Communication Services sectors.
- Valuations are stretched relative to historical norms (AMZN P/E above 50), reflecting
  AI-driven growth expectations.
- Only GOOGL trades below the S&P 500's average forward P/E, the cheapest name on the list.
```

---

### 5. Portfolio risk memo (structured + tool-heavy) — [`agent_finance_risk.py`](agent_finance_risk.py)

For each holding the agent pulls sector + country + price via `YahooFinanceStock`, then aggregates exposures and returns a typed `RiskMemo`. Useful as a starting point for compliance-style reporting.

```python
class RiskMemo(BaseModel):
    portfolio_value: float
    overall_risk_score: int        # 1 = very conservative, 10 = very aggressive
    sector_breakdown: List[SectorExposure]
    country_breakdown: List[CountryExposure]
    top_concentrations: List[Concentration]   # positions over 15% weight
    diversification_comment: str
    recommendation: str

agent = Agent(
    name="Risk Officer",
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[YahooFinanceStock()],
    response_model=RiskMemo,
)
memo: RiskMemo = agent.run(prompt).parsed
```

Illustrative output:
```
Portfolio value: $112,438.20
Overall risk score: 8/10

Sector breakdown:
  Technology                    71.42%
  Financial Services            10.96%
  Energy                         8.04%
  Semiconductors                 5.84%
  Communication Services         3.74%

Country breakdown:
  United States                 84.21%
  Taiwan                         9.31%
  Netherlands                    6.48%

Top concentrations (>15%):
  AAPL    16.84%  — single largest position; tied to consumer hardware cycle.
  NVDA    18.12%  — concentrated AI-accelerator exposure on top of MSFT/GOOGL semis tilt.

Diversification: The book is heavily concentrated in US-listed large-cap technology
(71% sector, 84% country). Semiconductor exposure is layered (NVDA + TSM + ASML) and
correlates highly across cycles.

Recommendation: Trim NVDA or AAPL toward 10–12% each, redeploy 5–8% into a
non-correlated sleeve (international developed equities or short-duration credit).
Keep XOM as the current commodity/inflation hedge but do not expand it without
broadening sector mix elsewhere.
```

---

## 🧰 Other examples in this directory

| File | What it shows |
|---|---|
| [`agent_async.py`](agent_async.py) | `agent.arun(...)` / async iteration |
| [`agent_fx.py`](agent_fx.py) | Foreign-exchange rate lookups via a custom tool |
| [`agent_news.py`](agent_news.py) | News aggregation pattern |
| [`agent_structured_output.py`](agent_structured_output.py) | Pydantic `response_model` pattern (non-finance domain) |
| [`agent_web.py`](agent_web.py), [`agent_websearch_*.py`](.) | Tavily + website readers |
| [`agent_wikipedia.py`](agent_wikipedia.py) | Wikipedia tool |
| [`demo_flow_sequential.py`](demo_flow_sequential.py), [`demo_flow_yaml.py`](demo_flow_yaml.py) | `Flow` DAG examples |
| [`demo_telemetry.py`](demo_telemetry.py) | OpenTelemetry tracing |
| [`demo_tools_app.py`](demo_tools_app.py) | Tool catalogue / inspector |
| [`llm_openai_batch.py`](llm_openai_batch.py) | `ChatOpenAI` Batch API: `create_batch` / `poll_batch` / `get_batch_results` |

---

## 💡 Propose Your Examples

We are always open to community contributions! If you have interesting examples, don't hesitate to share them with us.
