import json
from pydantic import Field

from gwenflow.tools import BaseTool
from gwenflow.readers.pdf import PDFReader


class PDFTool(BaseTool):

    name: str = "PDFTool"
    description: str = "This function reads a PDF and returns its content."

    def _run(self, file: str = Field(description="The path of the PDf file to read.")) -> str:
        reader = PDFReader()
        documents = reader.read(file)
        return json.dumps([doc.to_dict() for doc in documents])
