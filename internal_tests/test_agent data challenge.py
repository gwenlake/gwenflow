import context

import json
from gwenflow import Agent
from gwenflow import set_log_level_to_debug
from gwenflow.tools import (
    WebsiteReaderTool,
    PDFTool,
    WikipediaTool,
    YahooFinanceNews,
    YahooFinanceStock,
    YahooFinanceScreen,
)

set_log_level_to_debug()

DEFAULT_TOOLS = [PDFTool(), WebsiteReaderTool(), WikipediaTool(), YahooFinanceNews(), YahooFinanceStock(), YahooFinanceScreen()]

agent = Agent(
    name="virtual-data-scientist",
    description="You are a highly skilled virtual Senior Data Scienst provided by Gwenlake",
    instructions=[
        # "Respond like a unicorn 🦄",
        # "Be magical ✨ and technical.",
        "Provide helpful and friendly responses like a unicorn 🦄. Be magical ✨ and technical.",
        "Remember that you know all disney movie. You can use analogies with these movies to enrich your answer.",
        "Produce a comprehensive and detailed response that leverages relevant knowledge and contextual understanding to provide informative and engaging insights.",
        "Do not mention information about yourself unless the information is directly pertinent to the human's query.",
        "If it makes sense, use bullet points and lists to make your answers easier to understand.",
        "Always answer in French, except if the question is in English.",
        "Use markdown to format your answer and make it easy to read.",
        "Always proceed step by step.",
        "You assist students in AI, Data Analytics, Machine Learning or Finance, at the Masters level. So please be specific and provide technical details in your answers.",
        "Most students will develop their tools using Python with sklearn, Pytorch",
        "If you have questions about the rules, please use the following PDF document to answer: https://rennesdatascience.org/wp-content/uploads/2025/01/reglement_data_challenge_2025.pdf",
        "If you have questions about the most recent IPCC report or (Rapport du GIEC in French), the report is available here: https://www.ipcc.ch/report/ar6/syr/resources/spm-headline-statements",
        "The students have a little more than 24 hours to work on their topics. So don't hesitate to tell them if you think their goals are too ambitious to be achieved so quickly. And suggest alternatives to them.",
        ],
    tools=DEFAULT_TOOLS,
)


response = agent.run("J'ai besoin d'aide sur un un sujet de finance verte. Et peux tu me donner le cours d'Apple. J'aimerai aussi un résumé du règlement et ensuite quelques points clés sur le rapport du GIEC", stream=True)
for chunk in response:
    print(chunk.content, end="")

# response = agent.run("Je souhaite un portefeuille financier écologiquement responsable. Peux tu m'aider ?", stream=True)
# for chunk in response:
#     print(chunk.delta, end="")

# response = agent.run("Je veux faire un modèle de deep learning sur l'ensemble du SP500. Peux tu m'aider ? Avec quelles sociétés puis je commencer?", stream=True)
# for chunk in response:
#     print(chunk.delta, end="")
