import context
import asyncio
import random
from gwenflow import ChatAzureOpenAI
from gwenflow.tools import TavilyWebSearchTool, WebsiteReaderTool, DuckDuckGoNewsTool
from gwenflow import set_log_level_to_debug

set_log_level_to_debug()


async def main() -> None:

    messages = [
        {
            "role": "user",
            "content": "Retrieve the care pathway for AML in the USA and Europe. Where available, share link to the clinical guidelines",
        },
        {
            "role": "user",
            "content": "Retrieve the care pathway for AML in Europe. Where available, share link to the clinical guidelines",
        },
        {
            "role": "user",
            "content": "Retrieve the care pathway for AML in the USA. Where available, share link to the clinical guidelines",
        }
    ]

    # llm = ChatAzureOpenAI(model="gpt-4o-mini", tools=[TavilyWebSearchTool()])
    llm = ChatAzureOpenAI(model="gpt-4o-mini")

    tasks = []
    for i in range(2):
        random_index = random.randint(0, 2)
        task = asyncio.create_task(llm.ainvoke([messages[random_index]]))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        print(i+1, result.choices[0].message.content[:50])
        # print(i+1, result.content[:50])

    # stream
    # tasks = []
    # for i in range(5):
    #     random_index = random.randint(0, 2)
    #     task = asyncio.create_task(llm.astream([messages[random_index]]))
    #     tasks.append(task)

    # results = await asyncio.gather(*tasks)
    # for i, stream in enumerate(results):
    #     for chunk in stream:
    #         print(i+1, chunk)

asyncio.run(main())