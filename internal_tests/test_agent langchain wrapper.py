import context
import json
from gwenflow import ChatOpenAI, Agent, Task, Tool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.utils.function_calling import convert_to_openai_tool

# A wrapper for wikipedia
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia   = WikipediaQueryRun(api_wrapper=api_wrapper)

def tool_wrapper(text: str) -> str:
    """Cette fonction récupère des informations sur la météo"""
    print("\n\nMeteo invoked:", text)
    return "Il fait 20 degrés."

def tool_wikipedia(text: str) -> str:
    """Cette fonction récupère des informations sur wikipedia"""
    print("\n\nWikipedia invoked:", text)
    return wikipedia.invoke(text)

# attention, très forte importance du docstring de langchain, beaucoup mieux fait!
# tool_wikipedia.__doc__ = convert_to_openai_tool(wikipedia)["function"]["description"]
# print(json.dumps(convert_to_openai_tool(wikipedia), indent=4))

# Gwenflow Agents
llm = ChatOpenAI(model="gpt-4o-mini")
agent = Agent(
    role="Analyst",
    llm=llm,
    tools=[tool_wrapper, tool_wikipedia],
)

# query = "Summarize the wikipedia's page about Emmanuel Macron and please some info about his wife."
query = "Summarize the wikipedia's page about Emmanuel Macron and tell me about a recent fact. And please give me some info about his wife."
# query = "Trouve moi 3 liens (avec les url), que tu présentes sous forme de documents JSON, sur Emmanuel Macron."
# query = "Quelle est la météo aujourd'hui à Rennes?"
# query = "Quelle est la solution de 3+2?"
task = Task(
    description=query,
    expected_output="Answer in one paragraph.",
    agent=agent
)

print("Q:", query)
print("A:", task.run())