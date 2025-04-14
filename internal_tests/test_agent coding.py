import context

from gwenflow import ChatOpenAI, Agent, Task, Tool

import langchain.agents
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


python_repl = PythonREPL()
python_repl_tool = langchain.agents.Tool(
    name="python_repl",
    description="This tool can execute python code and shell commands (pip commands to modules installation) Use with caution",
    func=python_repl.run,
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=5000)
wikipedia   = WikipediaQueryRun(api_wrapper=api_wrapper)


tool_python    = Tool.from_langchain( python_repl_tool )
tool_wikipedia = Tool.from_langchain( wikipedia )

# Agent
llm = ChatOpenAI(model="gpt-4o-mini")

# agent = Agent(
#     role="Biographer",
#     instructions="Grab some information from Wikipedia about the person being analysed, and then prepare a detailed biography in a PPTX. Please, proceed step by step.",
#     llm=llm,
#     tools=[tool_python, tool_wikipedia],
# )

# task = Task(
#     description="Biography of Marilyn Monroe",
#     expected_output="Produce a pptx called biography.pptx",
#     agent=agent
# )

# print(task.run())


# Agent

instructions = """
Create an Excel file (in pptx format) on the topic given. \
Please use Wikipedia to get some information if needed. \
Please use the tool to generate and execute Python code, using the package yfinance if you need to get recent prices.
Please create only one xlsx file and stop the process after that.
"""

agent = Agent(
    role="CFA Financial Analyst",
    instructions=instructions.strip(),
    llm=llm,
    tools=[tool_python, tool_wikipedia],
)

task = Task(
    description="Create a diversified portfolio with the top 10 capitalizations of the SP500. I need the number of shares in the portfolio, the closing price, name and sector",
    expected_output="Produce a xlsx called portfolio.xlsx",
    agent=agent
)

print(task.run())
