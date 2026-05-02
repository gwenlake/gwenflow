from dataclasses import dataclass
from typing import Annotated, Union

from pydantic import Field

from gwenflow.tools.tool import Tool


@dataclass(kw_only=True)
class YahooFinanceSearch(Tool):
    name: str = "YahooFinanceSearch"
    description: str = "Search for a stock on Yahoo Finance."

    def _run(self, query: str = Field(description="The stock to search for.")):
        try:
            import yfinance as yf
            return yf.Search(query).quotes
        except ImportError:
            return "yfinance is not installed. Please install it with `pip install yfinance`."


@dataclass(kw_only=True)
class YahooFinancePick(Tool):
    name: str = "YahooFinancePick"
    description: str = "Search for a stock on Yahoo Finance."

    def _run(self, query: str = Field(description="The stock to search for.")):
        try:
            import yfinance as yf
            return yf.Search(query).quotes
        except ImportError:
            return "yfinance is not installed. Please install it with `pip install yfinance`."


@dataclass(kw_only=True)
class YahooFinanceStock(Tool):
    name: str = "YahooFinanceStock"
    description: str = "Retrieve stock data from Yahoo Finance."

    def _run(self, ticker: str = Field(description="The ticker stock to search for.")):
        try:
            import yfinance as yf
            return yf.Ticker(ticker).info
        except ImportError:
            return "yfinance is not installed. Please install it with `pip install yfinance`."


@dataclass(kw_only=True)
class YahooFinanceNews(Tool):
    name: str = "YahooFinanceNews"
    description: str = (
        "Search for news on company on Yahoo Finance."
        "Search for news on Yahoo Finance."
        "Input should be a search query."
        "This tool will return the latest news"
    )

    def _run(self, query: str = Field(description="The query to search for.")):
        try:
            import yfinance as yf
            stocks = yf.Search(query, enable_fuzzy_query=True, news_count=15, max_results=0)
            return {news["title"]: news["link"] for news in stocks.news}
        except ImportError:
            return "yfinance is not installed. Please install it with `pip install yfinance`."


@dataclass(kw_only=True)
class YahooFinanceScreen(Tool):
    name: str = "YahooFinanceScreen"
    description: str = (
        "Screen for stocks on Yahoo Finance.This tool will return a list of stocks that meet the criteria."
    )

    def _run(
        self,
        operator: Annotated[str, Field(description="The queries support operators: GT, LT, BTWN, EQ, AND, OR.")],
        values: Annotated[
            list[Union[str, int]], Field(description="The values to search for. Ex: ['intradaymarketcap', 1000000000]")
        ],
    ):
        try:
            import yfinance as yf

            screener = yf.Screener()
            try:
                query = yf.EquityQuery(operator, eval(values))
            except Exception:
                return "Query not valid. Please check the Yahoo Finance screener documentation."

            try:
                screener.set_body(
                    {
                        "size": 250,
                        "offset": 0,
                        "sortField": "avgdailyvol3m",
                        "sortType": "DESC",
                        "quoteType": "EQUITY",
                        "query": query.to_dict(),
                        "userId": "",
                        "userIdType": "guid",
                    }
                )
                return [quote["symbol"] for quote in screener.response["quotes"]]
            except Exception:
                return "Query not valid. Please check the Yahoo Finance screener documentation."

        except ImportError:
            return "yfinance is not installed. Please install it with `pip install yfinance`."
