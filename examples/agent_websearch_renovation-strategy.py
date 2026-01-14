import dotenv

from gwenflow import ChatOpenAI, Agent
from gwenflow.tools import TavilyWebSearchTool, WebsiteReaderTool, PDFReaderTool
from gwenflow import set_log_level_to_debug


set_log_level_to_debug()

dotenv.load_dotenv(override=True)

"""

Exemple pour montrer que l'on peut imposer


"""


EXAMPLE = [
    {
        "strategy": "Triple Glazed Window Installation",
        "cost": "Estimated cost range: $5,000 - $10,000",
        "ROI": "5-7 years",
        "details": "Replacing standard windows with triple glazed windows to reduce heat loss by up to 30%. Enhances insulation and reduces energy bills.",
        "resources": [
            {
                "title": "Energy Saving Home Improvement Ideas - NYSERDA",
                "url": "https://www.nyserda.ny.gov/Featured-Stories/Energy-Saving-Improvement-Ideas",
            }
        ],
    },
    {
        "strategy": "Attic Insulation Upgrade",
        "cost": "Estimated cost range: $1,500 - $3,000",
        "ROI": "3-5 years",
        "details": "Installing high-performance insulation in the attic to prevent heat loss and improve thermal regulation, reducing heating and cooling costs by 20%.",
        "resources": [
            {
                "title": "Insulation Options for Homes - Energy.gov",
                "url": "https://www.energy.gov/energysaver/weatherize/insulation",
            }
        ],
    },
    {
        "strategy": "Solar Panel Installation",
        "cost": "Estimated cost range: $10,000 - $25,000",
        "ROI": "8-10 years",
        "details": "Installing photovoltaic solar panels to generate clean, renewable energy, offsetting electricity bills and reducing carbon footprint.",
        "resources": [{"title": "Solar Energy Basics - SEIA", "url": "https://www.seia.org/initiatives/solar-energy"}],
    },
]


task = """
You are an energy renovation consultant specializing in sustainable home upgrades for residential properties. 
Your task is to identify the top 5 most effective energy renovation strategies based on the top 10 Google search results 
and the 3 most relevant PDF documents found on Google. 
Each strategy must be practical, actionable, and aimed 
at maximizing energy efficiency while considering cost-effectiveness and ROI. 
You must provide the output EXCLUSIVELY in JSON format, adhering strictly to the following structure without any additional text, context, or formatting:

{EXAMPLE}

Ensure that each strategy includes specific metrics, technical specifications, or case study references where applicable. 
Focus on strategies that balance cost, ROI, and environmental impact. 
Do not include any introductory or concluding text, only the JSON outputs of each strategy.
"""

agent = Agent(
    name="Energy Renovation Strategist",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    instructions=[
        "You analyze data from the top 10 Google search results and the 3 most relevant PDF documents.",
        "You must respect the given JSON structure.",
    ],
    tools=[TavilyWebSearchTool(), WebsiteReaderTool(), PDFReaderTool()],
)

response = agent.run(task)
print(response.content)
