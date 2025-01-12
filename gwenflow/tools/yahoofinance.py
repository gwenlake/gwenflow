from typing import Any

from pydantic import Field

from gwenflow.tools import BaseTool


class YahooFinanceSearch(BaseTool):
    name: str = "YahooFinanceSearch"
    description: str = "Search for a stock on Yahoo Finance."

    def _run(self, query: str = Field(description="The stock to search for.")) -> str:
        try:
            import yfinance as yf

            stocks = yf.Search(query)

            print(stocks.quotes)
        except ImportError:
            return "yfinance is not installed. Please install it with `pip install yfinance`."


class YahooFinancePick(BaseTool):
    name: str = "YahooFinancePick"
    description: str = "Search for a stock on Yahoo Finance."

    def _run(self, query: str = Field(description="The stock to search for.")) -> str:
        try:
            import yfinance as yf

            stocks = yf.Search(query)

            print(stocks.quotes)
        except ImportError:
            return "yfinance is not installed. Please install it with `pip install yfinance`."


class YahooFinanceStock(BaseTool):
    name: str = "YahooFinanceStock"
    description: str = "Retrieve stock data from Yahoo Finance."

    def _run(
        self, ticker: str = Field(description="The ticker stock to search for.")
    ) -> dict | str:
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)

            return stock.info

        except ImportError:
            return "yfinance is not installed. Please install it with `pip install yfinance`."


class YahooFinanceNews(BaseTool):
    name: str = "YahooFinanceNews"
    description: str = (
        "Search for news on company on Yahoo Finance."
        "Search for news on Yahoo Finance."
        "Input should be a search query."
        "This tool will return the latest news"
    )

    def _run(
        self, query: str = Field(description="The query to search for.")
    ) -> dict[Any, Any] | str:
        try:
            import yfinance as yf

            stocks = yf.Search(
                query, enable_fuzzy_query=True, news_count=15, max_results=0
            )
            # create a dict of news with title and link
            news_dict = {}
            for news in stocks.news:
                news_dict[news["title"]] = news["link"]

            return news_dict
        except ImportError:
            return "yfinance is not installed. Please install it with `pip install yfinance`."


class YahooFinanceScreen(BaseTool):
    name: str = "YahooFinanceScreen"
    description: str = (
        "Screen for stocks on Yahoo Finance."
        "This tool will return a list of stocks that meet the criteria."
    )

    def _run(
        self,
        operator: str = Field(
            description="The queries support operators: GT (greater than), LT (less than), BTWN (between), EQ (equals), and logical operators AND and OR for combining multiple conditions."
        ),
        values: list[str | int] = Field(
            description="The values to search for. For example, ['intradaymarketcap', 1000000000] for market cap greater than 1 billion.",
        ),
    ) -> list[Any] | str:
        try:
            import yfinance as yf

            screener = yf.Screener()

            try:

                query = yf.EquityQuery(operator, eval(values))

            except Exception:
                return "Query not valid. Please check the Yahoo Finance screener documentation."

            print(query.to_dict())

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

                result = screener.response
                symbols = [quote["symbol"] for quote in result["quotes"]]

                print(symbols)

                return symbols
            except Exception:
                return "Query not valid. Please check the Yahoo Finance screener documentation."

        except ImportError:
            return "yfinance is not installed. Please install it with `pip install yfinance`."
