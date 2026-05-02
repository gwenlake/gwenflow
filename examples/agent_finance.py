import dotenv

from gwenflow import Agent, ChatOpenAI
from gwenflow.tools.yahoofinance import YahooFinanceNews, YahooFinanceSearch, YahooFinanceStock

dotenv.load_dotenv(override=True)

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

queries = [
    "What is the current stock price of Apple?",
    "Give me a summary of recent news about Tesla.",
    "Compare the performance of Microsoft and Google stocks.",
]

for query in queries:
    print(f"\nQ: {query}")
    print(f"A: {agent.run(query).content}")
