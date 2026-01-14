import dotenv

from gwenflow import ChatOpenAI, Agent
from gwenflow.tools import TavilyWebSearchTool, WebsiteReaderTool, PDFReaderTool
from gwenflow import set_log_level_to_debug


set_log_level_to_debug()

dotenv.load_dotenv(override=True)


task = """
Create an outline for an article that will be 2,000 words on the keyword 'Best SEO prompts' for 
a company working in the following sector '{sector}', based on the top 10 results from Google.
Include every relevant heading possible. Keep the keyword density
of the headings high. For each section of the outline, include the word count. Include FAQs section in the outline too,
based on people also ask section from Google for the keyword. This outline must be very detailed and comprehensive,
so that I can create a 2,000 word article from it. Generate a long list of LSI and NLP keywords related to my keyword.
Also include any other words related to the keyword. Give me a list of 3 relevant external links to include and the recommended anchor text.
Make sure they’re not competing articles. Split the outline into part 1 and part 2.
"""

agent = Agent(
    name="SEO-Prompt",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    instructions=[
        "You are a journalist who writes articles.",
        "You write your articles in markdown",
        "You STRICTLY adhere to the requested word count",
    ],
    tools=[TavilyWebSearchTool(), WebsiteReaderTool(), PDFReaderTool()],
)

response = agent.run(task.format(sector="Artificial Intelligence and Data Analytics").strip().replace("\n", " "))
print(response.content)
