import requests
import json
import dotenv

from gwenflow import ChatOpenAI, Agent, Task, Tool

# load you api key
dotenv.load_dotenv(override=True)

# tool to get fx
def get_exchange_rate(currency_iso: str) -> str:
    """Get the current exchange rate for a given currency. Currency MUST be in iso format."""
    try:
        response = requests.get("http://www.floatrates.com/daily/usd.json").json()
        data = response[currency_iso.lower()]
        return json.dumps(data)
    except Exception as e:
        print(f"Currency not found: {currency_iso}")
    return "Currency not found"

tool_get_exchange_rate = Tool.from_function(get_exchange_rate)

# llm, agent and task
llm = ChatOpenAI(model="gpt-4o-mini")

agentfx = Agent(
    role="Fx Analyst",
    instructions="Get recent exchange rates data.",
    llm=llm,
    tools=[tool_get_exchange_rate],
)

queries = [
    "Find the capital city of France?",
    "What's the exchange rate of the Brazilian real?",
    "What's the exchange rate of the Euro?",
    "What's the exchange rate of the Chine Renminbi?",
    "What's the exchange rate of the Chinese Yuan?",
    "What's the exchange rate of the Tonga?"
]

for query in queries:
    task = Task(
        description=query,
        expected_output="Answer in one sentence and if there is a date, mention this date.",
        agent=agentfx
    )
    print("")
    print("Q:", query)
    print("A:", task.run())