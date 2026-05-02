from dataclasses import dataclass

from pydantic import Field

from gwenflow.readers.pdf import PDFReader
from gwenflow.tools.tool import Tool


@dataclass(kw_only=True)
class PDFReaderTool(Tool):
    name: str = "PDFReaderTool"
    description: str = "This function reads a PDF from a file or an url and returns its content."

    def _run(self, file: str = Field(description="The path of the PDF file to read.")):
        clean_documents = []
        for doc in PDFReader().read(file):
            meta = doc.metadata
            meta["content"] = doc.content
            clean_documents.append(meta)
        return clean_documents
