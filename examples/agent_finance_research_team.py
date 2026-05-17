"""Investment research team — multi-agent handoff.

A Portfolio Manager orchestrates two specialist agents:
  - Equity Analyst (fundamentals via YahooFinanceStock / YahooFinanceSearch)
  - News Analyst (recent catalysts via YahooFinanceNews)

The PM exposes each teammate as a callable tool (via `team=[...]`) and synthesises
a Buy / Hold / Avoid recommendation grounded in their findings.
"""

import dotenv

from gwenflow import Agent, ChatAnthropic
from gwenflow.tools.yahoofinance import YahooFinanceNews, YahooFinanceSearch, YahooFinanceStock

dotenv.load_dotenv(override=True)


equity_analyst = Agent(
    name="EquityAnalyst",
    description="Equity research analyst. Use for fundamentals: price, P/E, margins, growth, sector.",
    instructions=[
        "You are an equity research analyst.",
        "Pull the latest fundamentals via YahooFinanceStock / YahooFinanceSearch.",
        "Return a tight 4-6 line summary: price, market cap, sector, P/E, revenue growth, margins.",
    ],
    llm=ChatAnthropic(model="claude-haiku-4-5"),
    tools=[YahooFinanceSearch(), YahooFinanceStock()],
)

news_analyst = Agent(
    name="NewsAnalyst",
    description="News & catalyst analyst. Use for recent headlines, sentiment, and near-term catalysts.",
    instructions=[
        "You are a sell-side news analyst.",
        "Use YahooFinanceNews to surface the 3-5 most recent material headlines.",
        "Return a short bullet list with headline + a one-line takeaway each.",
    ],
    llm=ChatAnthropic(model="claude-haiku-4-5"),
    tools=[YahooFinanceNews()],
)

portfolio_manager = Agent(
    name="PortfolioManager",
    instructions=[
        "You are a portfolio manager evaluating whether to add a stock to a balanced equity portfolio.",
        "Delegate fundamentals questions to ask_equityanalyst and news/catalyst questions to ask_newsanalyst.",
        "Do NOT call Yahoo tools yourself; the specialists handle that.",
        "After gathering both views, return a structured recommendation: Buy / Hold / Avoid, "
        "a 1-paragraph thesis, and the top 2 risks.",
    ],
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    team=[equity_analyst, news_analyst],
)

candidates = ["NVDA", "INTC", "PLTR"]

for ticker in candidates:
    print(f"\n{'=' * 70}\nEvaluating {ticker}\n{'=' * 70}")
    response = portfolio_manager.run(
        f"Should we add {ticker} to a balanced US equity portfolio? Get fundamentals "
        f"from the equity analyst and recent news from the news analyst, then decide."
    )
    print(response.content)
