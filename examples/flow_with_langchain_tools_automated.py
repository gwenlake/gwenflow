import dotenv
from gwenflow import ChatOpenAI, Tool, AutoFlow

import langchain.agents
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# Load API key from .env file
dotenv.load_dotenv(override=True)

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


llm = ChatOpenAI(model="gpt-4o-mini")

flow = AutoFlow(llm=llm, tools=[tool_python, tool_wikipedia])
flow.generate_tasks(objective="Tell me about the biography of Kamala Harris and produce a pptx name biography_auto.pptx")
flow.run()
