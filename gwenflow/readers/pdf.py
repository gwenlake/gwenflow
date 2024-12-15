
import fitz
from abc import ABC
from typing import List

from gwenflow.documents import Document



class PDFReader(ABC):

    def load_data(self, file: str) -> List[Document]:

        documents = []

        doc = fitz.open(file, filetype="pdf")
        for page in doc:
            text = page.get_text()
            safe_text = text.encode('utf-8', errors='ignore').decode('utf-8')
            tables = []
            for table in page.find_tables():
                tables.append(table.extract())
            metadata = dict(filename=str(file), page=page.number+1, tables=tables, images=[])
            doc = Document(content=safe_text, metadata=metadata)
            documents.append(doc)

        return documents