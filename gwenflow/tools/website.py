from dataclasses import dataclass

from pydantic import Field

from gwenflow.readers.website import WebsiteReader
from gwenflow.tools.tool import BaseTool


@dataclass(kw_only=True)
class WebsiteReaderTool(BaseTool):
    name: str = "WebsiteReaderTool"
    description: str = "Fetches and returns the content of a given URL."
    max_depth: int = 1

    def _run(self, url: str = Field(description="The url of the website to read.")):
        return [
            {"content": doc.content, "url": doc.metadata["url"]}
            for doc in WebsiteReader(max_depth=self.max_depth).read(url)
        ]
