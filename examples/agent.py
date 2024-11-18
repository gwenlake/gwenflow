from gwenflow import ChatOpenAI, Agent, Task, Tool

import langchain.agents
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import dotenv

# load you api key
dotenv.load_dotenv(override=True)

python_repl = PythonREPL()
python_repl_tool = langchain.agents.Tool(
    name="python_repl",
    description="This tool can execute python code and shell commands (pip commands to modules installation) Use with caution. Write script in markdown block",
    func=python_repl.run,
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=5000)
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)

tool_python = Tool.from_langchain(python_repl_tool)
tool_wikipedia = Tool.from_langchain(wikipedia)

# Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

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
Think step by step about your objective. Write script in MARKDOWN block, and it will be executed. if package installation is needed, use pip command in a separate block but always send the python script after the installation block.
Always save figures to file in the current directory. Do not use plt.show()
"""

agent = Agent(
    role="CFA Financial Analyst",
    instructions=instructions.strip(),
    llm=llm,
    tools=[tool_python, tool_wikipedia],
)

task = Task(
    description="Create a python plot of NVIDA vs TSLA with MA YTD from 2024-01-01 to 01-08-2024, Write script in markdown block",
    expected_output="Produce a xlsx called portfolio.xlsx",
    agent=agent
)

print(task.run())

# if __name__ == "__main__":
#
#
#     output = runner.run_python_command(command)
#     print(output)