import context

from gwenflow import ChatOpenAI, Agent, Tool, Flow
from gwenflow import set_log_level_to_debug

set_log_level_to_debug()

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


llm = ChatOpenAI(model="gpt-4o-mini")


agent_bio = Agent(
    name="Biographer",
    tools=[tool_wikipedia],
)

agent_summarizer = Agent(
    name="Summarizer",
)

# agent_pptx = Agent(
#     name="Powerpoint Analyst",
#     description="Prepare a beautiful and professionnal Powerpoint file (PPTX format) summarizing the information given.",
#     llm=llm,
#     tools=[tool_python],
# )

flow = Flow(
    llm = ChatOpenAI(model="gpt-4o-mini"),
    steps=[
        {
            "agent": agent_bio,
        },
        {
            "agent": agent_summarizer,
            "task": "Summarize the biography given in the context in an ascii table.",
            "depends_on": ["Biographer"],
        }
    ]
)

response = flow.run("Write the biography of Emmanuel Macron.")
print(response["Summarizer"].content)