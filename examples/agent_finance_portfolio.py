"""Portfolio snapshot — structured output.

Given a list of holdings (ticker + shares), the agent fetches live prices via
YahooFinanceStock and returns a typed PortfolioSnapshot with total value,
position weights, and the day's top winners/losers.
"""

from typing import List

import dotenv
from pydantic import BaseModel, Field

from gwenflow import Agent, ChatAnthropic
from gwenflow.tools.yahoofinance import YahooFinanceStock

dotenv.load_dotenv(override=True)


class Position(BaseModel):
    ticker: str
    shares: float
    last_price: float
    market_value: float
    weight_pct: float = Field(description="Position weight as percent of total portfolio value.")
    day_change_pct: float


class PortfolioSnapshot(BaseModel):
    total_value: float
    positions: List[Position]
    top_winner: str = Field(description="Ticker with the largest positive day_change_pct.")
    top_loser: str = Field(description="Ticker with the largest negative day_change_pct.")
    commentary: str = Field(description="One short paragraph on portfolio composition and the day's moves.")


HOLDINGS = [
    {"ticker": "AAPL", "shares": 50},
    {"ticker": "MSFT", "shares": 30},
    {"ticker": "GOOGL", "shares": 20},
    {"ticker": "NVDA", "shares": 15},
    {"ticker": "JPM", "shares": 40},
]


agent = Agent(
    name="Portfolio Analyst",
    instructions=[
        "You are a portfolio analyst.",
        "For each holding, call YahooFinanceStock to get currentPrice and regularMarketChangePercent.",
        "Compute market_value = shares * currentPrice, then per-position weight_pct and the totals.",
        "Return the data in the required schema. Be precise with numbers; round to 2 decimals.",
    ],
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[YahooFinanceStock()],
    response_model=PortfolioSnapshot,
)

prompt = "Build a portfolio snapshot for these holdings: " + ", ".join(
    f"{h['shares']} {h['ticker']}" for h in HOLDINGS
)

response = agent.run(prompt)
snapshot: PortfolioSnapshot = response.parsed

print(f"Total portfolio value: ${snapshot.total_value:,.2f}")
print(f"\n{'Ticker':<8} {'Shares':>8} {'Price':>10} {'Value':>14} {'Weight':>8} {'Day %':>8}")
for p in snapshot.positions:
    print(
        f"{p.ticker:<8} {p.shares:>8.0f} ${p.last_price:>9.2f} ${p.market_value:>13,.2f} "
        f"{p.weight_pct:>7.2f}% {p.day_change_pct:>+7.2f}%"
    )
print(f"\nTop winner: {snapshot.top_winner}   Top loser: {snapshot.top_loser}")
print(f"\n{snapshot.commentary}")
