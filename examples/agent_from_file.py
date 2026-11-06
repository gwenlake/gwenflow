"""An agent described by one file: examples/agents/recipe_writer.yaml.

The file holds the model, the instructions, the model settings, the capabilities and the
output schema, in the Pydantic AI agent spec format. The code only brings the connection,
the variables of the instructions and the question.
"""

import dotenv

from gwenflow import Agent, ChatOpenAI

dotenv.load_dotenv(override=True)

# The file names the model; the code gives the connection it runs on (key, base URL, headers).
agent = Agent.from_file(
    "examples/agents/recipe_writer.yaml",
    llm=ChatOpenAI(),
    variables={"servings": 4, "diet": "vegetarian"},
)

print(agent.get_system_prompt(task=""))  # the instructions, variables filled
print()

response = agent.run("A quick weeknight pasta.")
recipe = response.parsed  # the output_schema of the file, as an object
print(f"{recipe.title} — {recipe.minutes} min")
for ingredient in recipe.ingredients:
    print(f"  - {ingredient}")
for number, step in enumerate(recipe.steps, start=1):
    print(f"  {number}. {step}")

# What the code passes wins over the file: other variables, another model, other settings.
chef = Agent.from_file(
    "examples/agents/recipe_writer.yaml",
    llm=ChatOpenAI(),
    variables={"servings": 2, "diet": "vegan"},
    model_settings={"temperature": 0.9},
)
print()
print(chef.run("A dessert with chocolate.").parsed.title)
