import os
from gwenflow.flows.base import Flow
from gwenflow.agents import Agent 
from gwenflow.llms import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "sk-proj-................................"

print("Initialisation du LLM OpenAI (GPT-4o-mini)...")
llm = ChatOpenAI(model="gpt-4o-mini")

print("Création manuelle des Agents...")

# --- Agent 1 ---
joker_agent = Agent(
    name="Joker",
    description="Tu es un humoriste. Tu inventes des blagues courtes.",
    response_model=None,
    tools=[],
    llm=llm,        
    depends_on=[] 
)

# --- Agent 2 ---
explainer_agent = Agent(
    name="Explainer",
    description="Tu es un professeur sérieux. Tu expliques pourquoi une blague est drôle de manière scientifique.",
    response_model=None,
    tools=[],
    llm=llm,
    depends_on=["Joker"]
)

print("Construction du Flow...")

steps = [
    {
        "agent": explainer_agent,
        "depends_on": ["Joker"]
    },
    {
        "agent": joker_agent,
        "depends_on": [] 
    }
]

flow = Flow(steps=steps, tools=[], llm=llm)

print("\n--- Structure de l'équipe ---")
flow.describe()

print("\nLancement de la mission...")
try:
    results = flow.run("Fais ton travail.")
    
    print("\nRésultats :")
    for agent_name, response in results.items():
        print(f"\n[Agent: {agent_name}]")
        if hasattr(response, 'choices'):
            print(f"> {response.choices[0].message.content}")
        elif hasattr(response, 'content'):
            print(f"> {response.content}")
        else:
            print(f"> {response}")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nErreur : {e}")