
import json
from abc import ABC
from typing import List

from gwenflow.documents import Document


class JSONReader(ABC):

    def load_data(self, file: str) -> List[Document]:

        documents = []

        with open(file, encoding="utf-8") as f:

            metadata = json.load(f)

            if "content" not in metadata:
                raise ValueError(f"JSON file without content: { str(file) }.")

            content = metadata.pop("content")
            metadata["filename"] = str(file)

            documents.append(
                Document(
                    content=content,
                    metadata=metadata
                )
            )

        return documents