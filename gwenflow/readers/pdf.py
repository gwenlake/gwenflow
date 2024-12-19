
import re
import fitz
from abc import ABC
from typing import List

from gwenflow.documents import Document
from gwenflow.utils.aws import aws_s3_read_file, aws_s3_uri_to_bucket_key


class PDFReader(ABC):

    def load_data(self, file: str) -> List[Document]:

        documents = []

        aws = False
        if file.startswith("s3://"):
            aws = True

        try:
            if aws:
                bucket, key = aws_s3_uri_to_bucket_key(file)
                data = aws_s3_read_file(bucket, key)
                doc = fitz.open(stream=data, filetype="pdf")
            else:
                doc = fitz.open(file, filetype="pdf")

        except Exception as e:
            print(repr(e))
            return []

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
