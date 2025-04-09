import hashlib
import re
import langchain_text_splitters
from pydantic import BaseModel
from tqdm import tqdm

from gwenflow.types.document import Document


class TokenTextSplitter(BaseModel):

    chunk_size: int = 500
    chunk_overlap: int = 100
    encoding_name: str = "cl100k_base"

    def split_text(self, text: str, metadata: dict = {}):
        text_splitter = langchain_text_splitters.TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=self.encoding_name,
        )
        pages = [p.page_content for p in text_splitter.create_documents([text])]

        chunks = []
        for i, chunk in enumerate(pages):
            _id = hashlib.md5("|".join([chunk, str(i)]).encode(), usedforsecurity=False).hexdigest()
            if "id" in metadata:
                _id = hashlib.md5("|".join([metadata["id"], str(i)]).encode(), usedforsecurity=False).hexdigest()
            chunks.append(Document(id=_id, content=chunk, metadata=metadata))

        return chunks
    
    def split_documents(self, documents: list, chunk_fields: list = [], meta_fields: list = [], clean_text: bool = False):
    
        chunks = []
        for document in tqdm(documents):

            if isinstance(document, Document):
                document = document.model_dump()
            
            # content
            if document.get("content"):
                content = document.pop("content")
            else:
                content = []
                if chunk_fields:
                    content = [ f"{f.upper()}: {document.get(f)}" for f in chunk_fields if document.get(f) ]
                else:
                    content = [ f"{k.upper()}: {v}" for k, v in document.items() if v ]
                content = ", ".join(content)

            # clean
            if clean_text:
                content = content.replace("\n", " ").replace("\r", " ")
                content = re.sub(' +', ' ', content)

            # meta
            metadata = document
            if meta_fields:
                metadata = {}
                for f in meta_fields:
                    metadata[f] = document.get(f)

            # split
            splitted_documents = self.split_text(text=content, metadata=metadata)
            chunks.extend(splitted_documents)

        return chunks
