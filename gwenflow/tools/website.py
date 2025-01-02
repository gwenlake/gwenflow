import json
from pydantic import Field

from gwenflow.tools import BaseTool
from gwenflow.readers.website import WebsiteReader


class WebsiteTool(BaseTool):

    name: str = "WebsiteTool"
    description: str = "Fetches and returns the content of a given URL."

    def _run(self, url: str = Field(description="The url of the website to read.")) -> str:
        reader = WebsiteReader(max_depth=1)
        documents = reader.read(url)
        return json.dumps([doc.to_dict() for doc in documents])
