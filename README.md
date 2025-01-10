<div align="center">

![Logo of Gwenflow](https://raw.githubusercontent.com/gwenlake/gwenflow/refs/heads/main/docs/images/gwenflow.png)

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

Install from PyPI:

```bash
pip install gwenflow
```

Install from the main branch to try the newest features:

```bash
# install from Github
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

dotenv.load_dotenv(override=True)  # load you api key from .env

llm = ChatOpenAI()
```

## Chat

```python
import os
from gwenflow import ChatOpenAI

dotenv.load_dotenv(override=True)  # load you api key from .env

messages = [
    {
        "role": "user",
        "content": "Describe Argentina in one sentence."
    }
]

llm = ChatOpenAI(model="gpt-4o-mini")
print(llm.invoke(messages=messages))
```

## Agent

```python
import os
from gwenflow import Agent

dotenv.load_dotenv(override=True)  # load you OpenAI api key from .env

# automatically use gpt-4o-mini for the Agent
agent = Agent(
    name="Agent",
    role="You are a helpful agent.",
)

response = agent.run("how are you?")
print(response.content)
```

## Agents and Tools

```python
import requests
import json
import dotenv

from gwenflow import ChatOpenAI, Agent, Tool

dotenv.load_dotenv(override=True)


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

llm = ChatOpenAI(model="gpt-4o-mini")

agent = Agent(
    name="AgentFX",
    role="Get recent exchange rates data.",
    instructions="Answer in one sentence and if there is a date, mention this date.",
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
```

```
Q: Find the capital city of France?
A: The capital city of France is Paris, which is not only the largest city in the country but also a major global center for art, fashion, and culture. Located in the north-central part of France along the Seine River, Paris is renowned for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city has a rich history dating back to its founding in the 3rd century BC as a small fishing village, and over the centuries, it has evolved into a vibrant metropolis known for its stylish boulevards, world-class cuisine, and significant historical events. Paris also serves as the political and administrative hub of France, housing key government institutions, including the Élysée Palace, where the French President resides.

Q: What's the exchange rate of the Brazilian real?
A: As of January 10, 2025, the exchange rate for the Brazilian Real (BRL) is approximately 6.0604 BRL to 1 USD, with an inverse rate of about 0.1650 USD for 1 BRL. This means that for every Brazilian Real, you can exchange it for about 16.5 cents in US dollars.

Q: What's the exchange rate of the Euro?
A: As of January 10, 2025, the exchange rate for the Euro (EUR) is approximately 0.9709 against the US Dollar, meaning 1 Euro can be exchanged for about 0.9709 USD; conversely, the inverse rate indicates that 1 USD is equivalent to about 1.0300 Euros.

Q: What's the exchange rate of the Chine Renminbi?
A: As of January 10, 2025, the exchange rate for the Chinese Renminbi (CNY), also known as the Chinese Yuan, is approximately 7.33 CNY per 1 USD. Additionally, the inverse rate indicates that 1 CNY is equivalent to about 0.1364 USD. This rate is important for various financial transactions, including trade, investments, and tourism related to China, reflecting the currency's strength in the global market as observed at 15:55 GMT on the same date.

Q: What's the exchange rate of the Chinese Yuan?
A: As of January 10, 2025, the exchange rate for the Chinese Yuan (CNY) is approximately 7.33 CNY to 1 USD, with an inverse rate of about 0.14 USD for each CNY, reflecting its value in the global market. This data indicates the strength of the yuan against the US dollar, and it is essential for understanding its purchasing power and economic interactions, especially for businesses and individuals engaging in international transactions.

Q: What's the exchange rate of the Tonga?
A: As of January 10, 2025, the current exchange rate for the Tongan paʻanga (ISO code: TOP) is approximately 2.42 TOP per 1 USD, while the inverse rate is around 0.41 USD per 1 TOP. The Tongan paʻanga is denoted by the numeric code 776, and the latest exchange data indicates that the rate was last updated at 15:55:14 GMT on the same day.
```

## Contributing to Gwenflow

We are very open to the community's contributions - be it a quick fix of a typo, or a completely new feature! You don't
need to be a Gwenflow expert to provide meaningful improvements.
