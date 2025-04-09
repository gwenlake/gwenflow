import json
from pydantic import Field

from gwenflow.tools import BaseTool
from gwenflow.readers.website import WebsiteReader


class WebsiteReaderTool(BaseTool):

    name: str = "WebsiteReaderTool"
    description: str = "Fetches and returns the content of a given URL."
    max_depth: int = 1

    def _run(self, url: str = Field(description="The url of the website to read.")):
        clean_documents = []
        for doc in WebsiteReader(max_depth=self.max_depth).read(url):
            clean_documents.append({
                "content": doc.content,
                "url": doc.metadata["url"]
            })
        return clean_documents
