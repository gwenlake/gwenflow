import asyncio

from gwenflow import ChatOpenAI
from gwenflow.tools import WebsiteTool
from gwenflow.utils import set_log_level_to_debug

set_log_level_to_debug()


async def main() -> None:
    messages = [
        {
            "role": "user",
            "content": "Tell me something all pages on this site https://www.labiotech.eu/",
        },
        {
            "role": "user",
            "content": "Tell me something all pages on this site https://www.biocentury.com/analysis/articles",
        },
        {
            "role": "user",
            "content": "tell me something all pages on this site https://bioengineer.org/",
        },
        {
            "role": "user",
            "content": "tell me something all pages on this site https://www.fiercebiotech.com/",
        },
        {
            "role": "user",
            "content": "tell me something all pages on this site https://www.fiercepharma.com/",
        },
        {
            "role": "user",
            "content": "tell me something all pages on this site https://www.fda.gov/",
        },
        {
            "role": "user",
            "content": "tell me something all pages on this site https://biotechinfo.fr/",
        },
        {
            "role": "user",
            "content": "tell me something all pages on this site https://www.biopharmadive.com/",
        },
        {
            "role": "user",
            "content": "tell me something all pages on this site https://www.biospace.com/latest-news-press-releases",
        },
        {
            "role": "user",
            "content": "tell me something all pages on this site https://www.wsj.com/tech/biotech",
        },
        {
            "role": "user",
            "content": "tell me something all pages on this site https://www.businesswire.com/portal/site/home/news/industry/?vnsId=31053",
        },
    ]
    tasks = []
    for message in messages:
        llm = ChatOpenAI(model="gpt-4o-mini", tools=[WebsiteTool()])
        task = asyncio.create_task(asyncio.to_thread(llm.invoke, [message]))
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        print(i, result[:50])


asyncio.run(main())
