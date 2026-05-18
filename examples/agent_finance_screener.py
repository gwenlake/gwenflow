"""Stock screener with streaming.

Uses YahooFinanceScreen to find tickers matching a market-cap floor, then enriches
each hit with YahooFinanceStock. The agent's running commentary is streamed
chunk-by-chunk via Agent.run_stream so you see the analysis as it's produced.
"""

import dotenv

from gwenflow import Agent, ChatAnthropic
from gwenflow.tools.yahoofinance import YahooFinanceScreen, YahooFinanceStock

dotenv.load_dotenv(override=True)


agent = Agent(
    name="Screener Analyst",
    instructions=[
        "You are a quantitative equity screener.",
        "Step 1: call YahooFinanceScreen with the requested operator and values to get a ticker list.",
        "Step 2: pick the top 5 tickers from the screen and call YahooFinanceStock on each.",
        "Step 3: write a concise markdown table with: Ticker | Name | Sector | Mkt Cap | P/E | 52w High.",
        "Then add 2-3 short bullet points on what the screen surfaces (themes, sector tilt, notable names).",
    ],
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[YahooFinanceScreen(), YahooFinanceStock()],
)

prompt = (
    "Run a large-cap US equity screen: market cap above $200B. "
    "Operator='GT', values=['intradaymarketcap', 200000000000]. "
    "Then enrich the top 5 with fundamentals and produce the report."
)

print("[streaming]: ", end="", flush=True)
for chunk in agent.run_stream(prompt):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()
