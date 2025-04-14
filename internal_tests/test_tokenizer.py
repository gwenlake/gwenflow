import context

from gwenflow import Document
from gwenflow.utils import num_tokens_from_string
from gwenflow.parsers import TokenTextSplitter

# print(num_tokens_from_string("test de tokens"))

text = """
Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua.

Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua.

Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua.
"""

splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=2, normalize_text=False)
splitted_text = splitter.split_text(text)
print(splitted_text)

splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=2, normalize_text=True)
splitted_text = splitter.split_text(text)
print(splitted_text)


documents = [
    Document(id="1", content="Hello1"),
    Document(id="2", content="Lorem ipsum dolor sit amet, sed do eiusmod tempor incididunt", metadata={"source": "gwenlake", "year": "2024"}),
]
splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=2, normalize_text=True)
splitted_docs = splitter.split_documents(documents)
print("")
for doc in splitted_docs:
    print(doc)

documents = [
    Document(id="1", content="test de content"),
    Document(id="2", content=""),
    Document(id="3", content="", metadata={"source": "gwenlake", "year": "2024"}),
]
splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=2, normalize_text=True)
splitted_docs = splitter.split_documents(documents, chunk_fields=["source", "year"], metadata_fields=["source"])
print("")
for doc in splitted_docs:
    print(doc)
