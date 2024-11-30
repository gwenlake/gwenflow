import dotenv
from datetime import date

from gwenflow import ChatOpenAI, Agent, Task, Tool, AutoFlow, Flow

import langchain.agents
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL

# Load API key from .env file
dotenv.load_dotenv(override=True)

today = date.today().strftime("%Y-%m-%d")

# Initialize DuckDuckGo tool
search = DuckDuckGoSearchRun()

tool_search = Tool.from_langchain(search)

# Set up language model
llm = ChatOpenAI(model="gpt-4o-mini")

python_repl = PythonREPL()


python_repl_tool = langchain.agents.Tool(
    name="python_repl",
    description="This tool can execute python code and shell commands (pip commands to modules installation) Use with caution",
    func=python_repl.run,
)

tool_python = Tool.from_langchain(python_repl_tool)

agent_search = Agent(
    role="Web retriever",
    instructions="Collect information from internet. Please, proceed step by step. Give structured answer. Be precise.",
    llm=llm,
    tools=[tool_search],
)

agent_coordinates = Agent(
    role="Map builder",
    instructions="build map using python code",
    llm=llm,
    tools=[tool_python, tool_search],
)

agent_map = Agent(
    role="Map builder",
    instructions="execute and save leaflet map using python code into a html file",
    llm=llm,
    tools=[tool_python],
)

task_city = task_meteo = Task(
    description="Get top 5 most populated French britanny cities.",
    expected_output="A list.",
    agent=agent_search
)

task_meteo = Task(
    description=f"Get latest temperature, in celcius degree and meteo status for the date : {today} for a all cities given.",
    expected_output="A table format, with cities in row, temperature and meteo status in columns",
    agent=agent_search
)

task_cooridnates = Task(
    description="From the given table, add for each city, their coordinates, in a column lat and long",
    expected_output="A table format, with cities in row, temperature, meteo status, lat and long in column.",
    agent=agent_coordinates
)

task_map = Task(
    description="From the given python code with coordinates, output a html geographic map file of latest temperature with cities. Use a different folium icon for the marker, depending the weather status.",
    expected_output="A html leaflet map of the cities called map.html.",
    agent=agent_map
)

flow = Flow(tasks=[task_city, task_meteo, task_cooridnates, task_map])
flow.run()
