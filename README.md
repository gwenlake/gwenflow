# Gwenflow

[![PyPI - License](https://img.shields.io/pypi/l/langchain-core?style=flat-square)](https://opensource.org/licenses/MIT)


**Gwenflow** is a Python framework designed by [Gwenlake](https://gwenlake.com) to facilitate the creation of customized, production-ready applications built around Agents and
Large Language Models (LLMs). It provides developers with the tools necessary to integrate LLMs and Agents, enabling efficient and scalable solutions tailored to specific business or user needs. With Gwenflow, developers can leverage the power of advanced language models within robust, deployable applications.

## Installation

Install from the main branch to try the newest features:

```bash
pip install -U git+https://github.com/gwenlake/gwenflow.git@main
```

## Usage

```python
import requests
import json
import dotenv

from gwenflow.llms.openai import ChatOpenAI
from gwenflow.agents import Agent, Task


# --- load you api key

dotenv.load_dotenv(override=True)


# --- tool to get fx

def getfx(currency_iso: str) -> str:
    """Get the current exchange rate for a given currency. Currency MUST be in iso format."""
    try:
        response = requests.get("http://www.floatrates.com/daily/usd.json").json()
        data = response[currency_iso.lower()]
        return json.dumps(data)
    except Exception as e:
        print(f"Currency not found: {currency_iso}")
    return "Currency not found"


# --- llm, agent and task

llm = ChatOpenAI(model="gpt-4o-mini")

agentfx = Agent(
    role="Fx Analyst",
    instructions="Get recent exchange rates data.",
    llm=llm,
    tools=[getfx],
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
    print("A:", task.run().output)
```

```
Q: Find the capital city of France?
A: The capital city of France is Paris.

Q: What's the exchange rate of the Brazilian real?
A: The exchange rate of the Brazilian real (BRL) is approximately 5.76, as of November 12, 2024.

Q: What's the exchange rate of the Euro?
A: The exchange rate of the Euro (EUR) is 0.9409 as of November 12, 2024.

Q: What's the exchange rate of the Chine Renminbi?
A: The exchange rate of the Chinese Renminbi (CNY) is 7.23 as of November 12, 2024.

Q: What's the exchange rate of the Chinese Yuan?
A: The exchange rate of the Chinese Yuan (CNY) is 7.23 as of November 12, 2024.

Q: What's the exchange rate of the Tonga?
A: The current exchange rate for the Tongan paʻanga (TOP) is 2.3662, as of November 12, 2024.
```

## Contributing to Gwenflow

We are very open to the community's contributions - be it a quick fix of a typo, or a completely new feature! You don't need to be a Gwenflow expert to provide meaningful improvements.
