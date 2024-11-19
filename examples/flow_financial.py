import dotenv
from gwenflow import ChatOpenAI, Tool, AutoFlow

import langchain.agents
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper

# Load API key from .env file
dotenv.load_dotenv(override=True)

# Initialize Python REPL tool
python_repl = PythonREPL()
python_repl_tool = langchain.agents.Tool(
    name="python_repl",
    description="This tool can execute python code and shell commands (pip commands to modules installation) Use with caution",
    func=python_repl.run,
)
_alpha = AlphaVantageAPIWrapper()

def alpha_vantage (symbol):
    """Make a request to the AlphaVantage API to get the
            latest price and volume information."""
    return _alpha._get_quote_endpoint(symbol)

# Initialize Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=5000)
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)

# Wrap tools
tool_python = Tool.from_langchain(python_repl_tool)
tool_wikipedia = Tool.from_langchain(wikipedia)
tool_afinance = Tool.from_function(alpha_vantage)

# Set up language model
llm = ChatOpenAI(model="gpt-4o")

# Define the flow with a financial objective
flow = AutoFlow(llm=llm, tools=[tool_python, tool_wikipedia, tool_afinance])
flow.generate_tasks(objective="Create a detailed financial analysis of Apple Inc., including stock performance, revenue trends, and market competition, with charts and tables saved as PNGs and compiled into a professional PowerPoint presentation named financial_analysis.pptx")
flow.run()