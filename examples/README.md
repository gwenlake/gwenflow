# Agent - Examples and Use Cases

This page demonstrates various examples of using agents in **Gwenflow** to interact with different tools, including local functions, web search, and memory management.

This Markdown shows various examples, they are more detailed in their specific Python files.

## 📦 Function Agents

You can provide agents with specific tools that perform targeted tasks. Use the `@agent.tool` decorator to register any Python function — the name, docstring, and type annotations are automatically converted into the schema the LLM receives.

### Example: Bike Availability Tool

```python
import requests
import json

from gwenflow import Agent, ChatOpenAI

agent = Agent(
    name="City Data Agent",
    llm=ChatOpenAI(model="gpt-5-mini"),
    instructions=[
        "Provide bike data for specified stations.",
        "Respond in JSON format only."
    ],
)

@agent.tool
def get_bike_availability(station_name: str = None) -> str:
    """Get bike availability from Rennes Métropole stations. Omit station_name to list all stations."""
    url_station = f"https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-des-stations-le-velo-star-en-temps-reel/records?where=nom=%22{station_name}%22&limit=1"
    url_general = "https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-des-stations-le-velo-star-en-temps-reel/records"
    try:
        response = requests.get(url_station if station_name else url_general)
        response.raise_for_status()
        return json.dumps(response.json().get('results', []))
    except Exception as e:
        return json.dumps({"error": str(e)})

queries = [
    "What is the bike availability at Oberthur station?",
    "List all bike docking stations."
]

for query in queries:
    response = agent.run(query)
    print(response.content)
```

---

## 🌐 Web Search Agents

Web search agents can explore and extract data from the internet, making them ideal for content generation and data analysis.

### Example: SEO Content Generator

```python
from gwenflow import Agent, ChatOpenAI
from gwenflow.tools import TavilyWebSearchTool, WebsiteReaderTool, PDFReaderTool

task = """
Create an outline for an article that will be 2,000 words on the keyword 'Best SEO prompts' for a company working in the sector '{sector}', based on the top 10 results from Google.
Include every relevant heading possible, FAQs, and LSI keywords. Provide recommended external links with anchor text.
Split the outline into part 1 and part 2.
"""

agent = Agent(
    name="SEO-Prompt",
    llm=ChatOpenAI(model="gpt-5-mini"),
    instructions=[
        "You are a content strategist.",
        "Write the content in markdown format.",
        "Ensure to structure the outline clearly into sections and subsections."
    ],
    tools=[TavilyWebSearchTool(), WebsiteReaderTool(), PDFReaderTool()]
)

response = agent.run(task.format(sector="Data Analytics"))
print(response.content)
```

---

## 🧠 Memory Management

Agents can be initialized with memory to maintain context across interactions, allowing for more natural and context-aware conversations.

### Example: Medical Assistant with Memory

```python
from gwenflow import Agent, ChatOpenAI
from gwenflow.memory import ChatMemoryBuffer

messages = [
    {"role": "user", "content": "Tell me about Alzheimer"},
    {"role": "assistant", "content": "Alzheimer is a neurodegenerative disease."}
]

memory = ChatMemoryBuffer(token_limit=300)
memory.add_messages(messages)

agent = Agent(
    name="Medical Assistant",
    llm=ChatOpenAI(model="gpt-5-mini"),
    instructions=[
        "Provide medical explanations in simple language.",
        "Maintain context across multiple queries."
    ],
    history=memory
)

response = agent.run("What did we discuss previously?")
print(response.content)
```

---

## 💡 Propose Your Examples

We are always open to community contributions! If you have interesting examples, don't hesitate to share them with us.
