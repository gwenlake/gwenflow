<div align="center">

![Logo of Gwenflow](./docs/images/gwenflow.png)

**A framework for orchestrating applications powered by autonomous AI agents and LLMs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/v/release/gwenlake/gwenflow)](https://github.com/your-username/gwenflow/releases)


</div>


## Why Gwenflow?

Gwenflow, a framework designed by [Gwenlake](https://gwenlake.com), 
streamlines the creation of customized, production-ready applications built around Agents and
Large Language Models (LLMs). It provides developers with the tools necessary
to integrate LLMs and Agents, enabling efficient and
scalable solutions tailored to specific business or user needs.

## Installation

Install from the main branch to try the newest features:

```bash
pip install -U git+https://github.com/gwenlake/gwenflow.git@main
```

## Usage

Load your OpenAI api key from an environment variable:

```python
import os
from gwenflow import ChatOpenAI


llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)
```

or load your api key from a local .env file:

```python
import os
import dotenv
from gwenflow import ChatOpenAI

dotenv.load_dotenv(override=True) # load you api key from .env

llm = ChatOpenAI()
```

## Chat

```python
import os
from gwenflow import ChatOpenAI

dotenv.load_dotenv(override=True) # load you api key from .env

messages = [
    {
        "role": "user",
        "content": "Describe Argentina in one sentence."
    }
]

llm = ChatOpenAI(model="gpt-4o-mini")
print( llm.invoke(messages=messages) )
```

## Agent

```python
import os
from gwenflow import Agent

dotenv.load_dotenv(override=True) # load you OpenAI api key from .env

# automatically use gpt-4o-mini for the Agent
agent = Agent(
    role="Agent",
    description="You are a helpful agent.",
)

response = agent.run("how are you?")
print(response.content)
```

## Agents, Tasks and Tools

```python
import requests
import json
import dotenv

from gwenflow import ChatOpenAI, Agent, Task, Tool


dotenv.load_dotenv(override=True) # load you api key from .env

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

# Loop on a list of tasks
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

## Automated Flows, Agents with Langchain Tools

Run an agent with Langchain tools. Requires langchain and pip install langchain-experimental
```
pip install langchain
pip install langchain-experimental
```

> [!CAUTION]  
> Python REPL can execute arbitrary code on the host machine (e.g., delete files, make network requests). Use with caution.
> For more information general security guidelines, please see https://python.langchain.com/docs/security/.

```python
import dotenv
from gwenflow import ChatOpenAI, Tool, AutoFlow

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

flow = AutoFlow(llm=llm, tools=[tool_python, tool_wikipedia])
flow.generate_tasks(objective="Tell me about the biography of Kamala Harris and produce a pptx name biography_auto.pptx")
flow.run()
```

## Contributing to Gwenflow

We are very open to the community's contributions - be it a quick fix of a typo, or a completely new feature! You don't need to be a Gwenflow expert to provide meaningful improvements.
