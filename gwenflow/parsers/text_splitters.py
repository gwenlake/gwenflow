import re
import hashlib
from pydantic import BaseModel
from typing import List, Callable
from tqdm import tqdm

from gwenflow.types.document import Document
from gwenflow.logger import logger

try:
    import tiktoken
except ImportError:
    raise ImportError("`tiktoken` is not installed. Please install it with `pip install tiktoken`.")


class TokenTextSplitter(BaseModel):

    chunk_size: int = 500
    chunk_overlap: int = 100
    encoding_name: str = "cl100k_base"
    strip_whitespace: bool = False
    normalize_text: bool = False

    def split_text(self, text: str) -> List[str]:

        _tokenizer = tiktoken.get_encoding(self.encoding_name)
        input_ids = _tokenizer.encode(text)

        if self.normalize_text:
            text = text.replace("\n", " ").replace("\r", " ")
            text = re.sub(' +', ' ', text)        

        splits: List[str] = []
        start_idx = 0
        cur_idx = min(start_idx + self.chunk_size, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        while start_idx < len(input_ids):
            splits.append(_tokenizer.decode(chunk_ids))
            if cur_idx == len(input_ids):
                break
            start_idx += self.chunk_size - self.chunk_overlap
            cur_idx = min(start_idx + self.chunk_size, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]

        if self.strip_whitespace:
            splits = [s.strip() for s in splits]
        
        return splits
    
    def split_documents(self, documents: list[Document], chunk_fields: list = [], metadata_fields: list = []) -> list[Document]:
    
        chunks = []
        for document in tqdm(documents):

            if not document.id:
                logger.warning(f"Missing id on document: { document['content'] }. Skipping document.")
                continue

            # content
            content = ""
            if document.content:
                content = document.content
            elif chunk_fields:
                content = [ f"{f.upper()}: {document.metadata.get(f)}" for f in chunk_fields if document.metadata.get(f) ]
                content = ", ".join(content)
            else:
                content = [ f"{k.upper()}: {v}" for k, v in document.metadata.items() if v ]
                content = ", ".join(content)
            
            if not content:
                logger.warning(f"Missing content on document id: { document.id }. Skipping document.")
                continue

            # meta
            metadata = document.metadata
            if metadata_fields:
                metadata = {}
                for f in metadata_fields:
                    metadata[f] = document.metadata.get(f)
            metadata["document_id"] = document.id # keep original doc id

            # split
            splitted_text = self.split_text(text=content)
            for i, chunk in enumerate(splitted_text):
                metadata["chunk_id"] = f"chunk_{i}"
                _id = hashlib.md5("-".join([document.id, str(i)]).encode(), usedforsecurity=False).hexdigest()
                chunks.append(Document(id=_id, content=chunk, metadata=metadata))

        return chunks
