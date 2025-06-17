# Agent - Examples and Use Cases

This page demonstrates various examples of using agents in **Gwenflow** to interact with different tools, including local functions, web search, and memory management.

This Markdown shows various examples, they are more detailed in their specific Python files.

## 📦 Function Agents

You can provide agents with specific tools that perform targeted tasks. These tools can be local functions converted into actionable tools for agents.

### Example: Bike Availability Tool

This example demonstrates how to create a simple tool that retrieves bike availability data from a public API and integrates it into an agent:

```python
import requests
import json
from gwenflow.tools import FunctionTool
from gwenflow.llms import ChatOpenAI
from gwenflow.agent import Agent

def get_bike_availability(station_name: str = None) -> str:
    url_station = f"https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-des-stations-le-velo-star-en-temps-reel/records?where=nom=%22{station_name}%22&limit=1"
    url_general = "https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-des-stations-le-velo-star-en-temps-reel/records"
    try:
        response = requests.get(url_station if station_name else url_general)
        response.raise_for_status()
        data = response.json()
        return json.dumps(data.get('results', []))
    except Exception as e:
        return json.dumps({"error": str(e)})

# Convert the function into a tool
bike_tool = FunctionTool.from_function(get_bike_availability)

# Initialize the agent
agent = Agent(
    name="City Data Agent",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    instructions=[
        "Provide bike data for specified stations.",
        "Respond in JSON format only."
    ],
    tools=[bike_tool]
)

# Example queries
queries = [
    "What is the bike availability at Oberthur station?",
    "List all bike docking stations."
]

for query in queries:
    response = agent.run(query)
    print(response)
```

---

## 🌐 Web Search Agents

Web search agents can explore and extract data from the internet, making them ideal for content generation and data analysis.

### Example: SEO Content Generator

```python
from gwenflow.tools import TavilyWebSearchTool, WebsiteReaderTool, PDFReaderTool
from gwenflow.llms import ChatOpenAI
from gwenflow.agent import Agent

# Task description
task = """
Create an outline for an article that will be 2,000 words on the keyword 'Best SEO prompts' for a company working in the sector '{sector}', based on the top 10 results from Google.
Include every relevant heading possible, FAQs, and LSI keywords. Provide recommended external links with anchor text.
Split the outline into part 1 and part 2.
"""

# Agent initialization with detailed configuration
agent = Agent(
    name="SEO-Prompt",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    instructions=[
        "You are a content strategist.",
        "Write the content in markdown format.",
        "Ensure to structure the outline clearly into sections and subsections."
    ],
    tools=[TavilyWebSearchTool(), WebsiteReaderTool(), PDFReaderTool()]
)

# Execute the task
response = agent.run(task.format(sector="Data Analytics"))
print(response.content)
```

---

## 🧠 Memory Management

Agents can be initialized with memory to maintain context across interactions, allowing for more natural and context-aware conversations.

### Example: Medical Assistant with Memory

```python
from gwenflow.memory import ChatMemoryBuffer
from gwenflow.llms import ChatOpenAI
from gwenflow.agent import Agent

# Initial context messages
messages = [
    {"role": "user", "content": "Tell me about Alzheimer"},
    {"role": "assistant", "content": "Alzheimer is a neurodegenerative disease."}
]

# Initialize memory with a token limit
memory = ChatMemoryBuffer(token_limit=300)
memory.add_messages(messages)

# Create the agent with memory
agent = Agent(
    name="Medical Assistant",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    instructions=[
        "Provide medical explanations in simple language.",
        "Maintain context across multiple queries."
    ],
    history=memory
)

# Query the agent
response = agent.run("What did we discuss previously?")
print(response)
```

---

## 💡 Propose Your Examples

We are always open to community contributions! If you have interesting examples, don't hesitate to share them with us.
