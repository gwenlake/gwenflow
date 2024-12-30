
import json
from typing import List
from pathlib import Path

from gwenflow.types import Document
from gwenflow.readers.base import Reader
from gwenflow.utils import logger


class JSONReader(Reader):

    def read(self, file: Path) -> List[Document]:

        try:

            filename = self.get_file_name(file)
            content  = self.get_file_content(file, text_mode=True)

            json_content = json.loads(content)

            if isinstance(json_content, dict):
                json_content = [json_content]

            documents = [
                Document(
                    id=f"{filename}_{page_num}",
                    content=json.dumps(page_content),
                    metadata={"filename": filename, "page": page_num},
                )
                for page_num, page_content in enumerate(json_content, start=1)
            ]
    
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return []

        return documents
