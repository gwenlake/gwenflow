import dotenv
from gwenflow import ChatOpenAI, Agent, Task, Tool, Flow

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

llm = ChatOpenAI(model="gpt-4o")

agent_bio = Agent(
    role="Biographer",
    instructions="Grab some information from Wikipedia about the person being analysed. Please, proceed step by step.",
    llm=llm,
    tools=[tool_wikipedia],
)

task = Task(
    description="Biography of Marilyn Monroe",
    expected_output="Two pages of markdown text.",
    agent=agent_bio
)

agent_bio = Agent(
    role="Biographer",
    instructions="Grab some information from Wikipedia about the person being analysed. Please, proceed step by step.",
    llm=llm,
    tools=[tool_wikipedia],
)

task_bio = Task(
    description="Biography of Marilyn Monroe",
    expected_output="Two pages of markdown text.",
    agent=agent_bio
)

agent_pptx = Agent(
    role="Powerpoint Analyst",
    instructions="You prepare Powerpoint on any topic.",
    llm=llm,
    tools=[tool_python],
)

task_pptx = Task(
    description="Prepare a professionnal Powerpoint summarizing the information given",
    expected_output="A pptx file called biography.pptx.",
    agent=agent_pptx
)

flow = Flow(tasks=[task_bio, task_pptx])
flow.run()
