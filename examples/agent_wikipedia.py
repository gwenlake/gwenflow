import dotenv

from gwenflow import Agent, ChatOpenAI
from gwenflow.tools import DuckDuckGoSearchTool, WikipediaTool

dotenv.load_dotenv(override=True)

agent = Agent(
    name="Research Assistant",
    instructions=[
        "You are an expert researcher.",
        "Use Wikipedia for factual background and DuckDuckGo for recent information.",
        "Cite your sources briefly in your answer.",
    ],
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[WikipediaTool(), DuckDuckGoSearchTool()],
)

queries = [
    "What is quantum computing and what are its current applications?",
    "Who was Alan Turing and why is he important?",
    "What is the James Webb Space Telescope?",
]

for query in queries:
    print(f"\nQ: {query}")
    print(f"A: {agent.run(query).content}")
