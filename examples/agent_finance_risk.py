"""Portfolio risk memo — structured output + tool-heavy.

Given a portfolio of holdings, the agent pulls per-ticker fundamentals from
YahooFinanceStock, computes sector / region concentration, and emits a typed
RiskMemo suitable for compliance-style review.
"""

from typing import List

import dotenv
from pydantic import BaseModel, Field

from gwenflow import Agent, ChatAnthropic
from gwenflow.tools.yahoofinance import YahooFinanceStock

dotenv.load_dotenv(override=True)


class SectorExposure(BaseModel):
    sector: str
    weight_pct: float


class CountryExposure(BaseModel):
    country: str
    weight_pct: float


class Concentration(BaseModel):
    ticker: str
    weight_pct: float
    note: str = Field(description="One short line on why this concentration is notable.")


class RiskMemo(BaseModel):
    portfolio_value: float
    overall_risk_score: int = Field(ge=1, le=10, description="1 = very conservative, 10 = very aggressive.")
    sector_breakdown: List[SectorExposure]
    country_breakdown: List[CountryExposure]
    top_concentrations: List[Concentration] = Field(description="Positions over 15% weight.")
    diversification_comment: str
    recommendation: str = Field(description="2-3 sentence action item for the PM.")


HOLDINGS = [
    {"ticker": "AAPL", "shares": 80},
    {"ticker": "MSFT", "shares": 60},
    {"ticker": "NVDA", "shares": 40},
    {"ticker": "GOOGL", "shares": 25},
    {"ticker": "META", "shares": 25},
    {"ticker": "TSM", "shares": 30},
    {"ticker": "ASML", "shares": 8},
    {"ticker": "JPM", "shares": 50},
    {"ticker": "XOM", "shares": 60},
]


agent = Agent(
    name="Risk Officer",
    instructions=[
        "You are a portfolio risk officer.",
        "For each holding, call YahooFinanceStock and extract: currentPrice, sector, country.",
        "Compute market_value per position (shares * currentPrice) and total portfolio_value.",
        "Aggregate weights by sector and by country (in percent).",
        "Flag any position whose weight exceeds 15% as a top_concentration.",
        "Assign overall_risk_score 1-10 based on concentration, sector tilt, and geographic mix.",
        "Be objective and specific; the recommendation should be actionable.",
    ],
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[YahooFinanceStock()],
    response_model=RiskMemo,
)

prompt = "Produce a risk memo for these holdings: " + ", ".join(
    f"{h['shares']} {h['ticker']}" for h in HOLDINGS
)

response = agent.run(prompt)
memo: RiskMemo = response.parsed

print(f"Portfolio value: ${memo.portfolio_value:,.2f}")
print(f"Overall risk score: {memo.overall_risk_score}/10")

print("\nSector breakdown:")
for s in sorted(memo.sector_breakdown, key=lambda x: -x.weight_pct):
    print(f"  {s.sector:<28} {s.weight_pct:>6.2f}%")

print("\nCountry breakdown:")
for c in sorted(memo.country_breakdown, key=lambda x: -x.weight_pct):
    print(f"  {c.country:<28} {c.weight_pct:>6.2f}%")

if memo.top_concentrations:
    print("\nTop concentrations (>15%):")
    for c in memo.top_concentrations:
        print(f"  {c.ticker:<6} {c.weight_pct:>6.2f}%  — {c.note}")

print(f"\nDiversification: {memo.diversification_comment}")
print(f"\nRecommendation: {memo.recommendation}")
