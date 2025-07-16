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
    
    def split_documents(self, documents: list[Document], chunk_fields: list = [], metadata_fields: list = [], min_last_chunk_size: int = 150) -> list[Document]:
    
        chunks = []
        tokenizer = tiktoken.get_encoding(self.encoding_name)

        for i, document in enumerate(tqdm(documents), start=0):

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
            
            if not document.content:
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

            extra_page = None

            # if next is the next page from the same doc
            if (
                i + 1 < len(documents) and
                len(splitted_text) > 0 and
                documents[i].metadata.get("filename") == documents[i + 1].metadata.get("filename") and
                documents[i + 1].metadata.get("page") == documents[i].metadata.get("page") + 1
            ):
                last_chunk = splitted_text[-1]
                last_chunk_tokens = tokenizer.encode(last_chunk)
                last_chunk_len = len(last_chunk_tokens)
                
                # if the last chunk is too short, we fusion it with the first from the next page
                if last_chunk_len < min_last_chunk_size:
                    next_doc = documents[i + 1]
                    next_text = next_doc.content or ""
                    next_chunks = self.split_text(next_text)
                    extra_page = next_doc.metadata.get("page")

                    if next_chunks:
                        # encode first chunk
                        next_chunk_tokens = tokenizer.encode(next_chunks[0])

                        # how much we can add
                        max_to_add = self.chunk_size - last_chunk_len

                        # take allowed part
                        to_add_tokens = next_chunk_tokens[:max_to_add]
                        to_add = tokenizer.decode(to_add_tokens)
                        splitted_text[-1] = last_chunk + " " + to_add

                        # remove from next doc
                        remaining_tokens = next_chunk_tokens[max_to_add:]
                        remaining_text = tokenizer.decode(remaining_tokens).strip()

                        # if something left, keep it
                        if remaining_text:
                            next_chunks[0] = remaining_text
                            documents[i + 1].content = " ".join(next_chunks).strip()
                        else:
                            documents[i + 1].content = ""  # force empty so it gets skipped later


            for j, chunk in enumerate(splitted_text):
                chunk_meta = metadata.copy() 
                chunk_meta["chunk_id"] = f"chunk_{i}"
                page = documents[i].metadata.get("page")
                chunk_meta["page"] = [page] if j < len(splitted_text) - 1 or not extra_page else [page, extra_page]

                _id = hashlib.md5("-".join([document.id, str(j)]).encode(), usedforsecurity=False).hexdigest()
                chunks.append(Document(id=_id, content=chunk.strip(), metadata=chunk_meta))

        return chunks
