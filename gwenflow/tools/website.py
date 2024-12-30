import json
from pydantic import Field

from gwenflow.tools import BaseTool
from gwenflow.readers.website import WebsiteReader


class WebsiteTool(BaseTool):

    name: str = "website"
    description: str = "This function reads a url and returns the content."

    def _run(self, url: str = Field(description="The url of the website to read.")):
        reader = WebsiteReader(max_depth=1)
        documents = reader.read(url)
        return json.dumps([doc.to_dict() for doc in documents])
