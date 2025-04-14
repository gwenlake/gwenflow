import context
import requests
import json

from gwenflow import ChatOpenAI, Agent, FunctionTool


llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatGwenlake(model="meta/llama-3.1-8b-instruct")


# test 1
# def get_weather(location) -> str:
#     return "{'temp':67, 'unit':'F'}"

# agent = Agent(
#     name="Agent",
#     instructions="You are a helpful agent.",
#     functions=[get_weather],
# )

# messages = [{"role": "user", "content": "What's the weather in NYC?"}]

# response = client.run(agent=agent, messages=messages)
# print(response.messages[-1]["content"])


# test 2
def get_exchange_rate(currency_iso: str) -> str:
    """Get the current exchange rate for a given currency. Currency MUST be in iso format."""
    try:
        response = requests.get("http://www.floatrates.com/daily/usd.json").json()
        data = response[currency_iso.lower()]
        return json.dumps(data)
    except Exception as e:
        print(f"Currency not found: {currency_iso}")
    return "Currency not found"

tool_get_exchange_rate = FunctionTool.from_function(get_exchange_rate)


agent = Agent(
    name="AgentFX",
    instructions=[
        "Your role is to get exchange rates data.",
        "Answer in one sentence and if there is a date, mention this date.",
    ],
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
    print("")
    print("Q:", query)
    print("A:", agent.run(query).content)

    # stream = task.stream()
    # for s in stream:
    #     print(s)
