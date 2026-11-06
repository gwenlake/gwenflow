import dotenv

from gwenflow import Agent, ChatOpenAI

dotenv.load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4o-mini")

joker = Agent(
    name="Joker",
    instructions="You are a comedian. Tell short, funny jokes.",
    llm=llm,
)

explainer = Agent(
    name="Explainer",
    instructions="You are a professor. Explain why jokes are funny in a scientific way, citing humour theory.",
    llm=llm,
)

joke_response = joker.run("Tell me a joke about programmers.")
print(f"Joke:\n{joke_response.content}\n")

explanation = explainer.run(f"Explain why this joke is funny:\n\n{joke_response.content}")
print(f"Why it's funny:\n{explanation.content}")
