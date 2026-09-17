"""Document readers.

Every reader is resolved lazily (PEP 562): each one pulls its own parsing stack
(pypdf, python-docx, openpyxl, beautifulsoup4, ...) and a deployment typically
uses one or none of them. They stay importable from this package as before --
the dependency is only loaded on first attribute access.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gwenflow.readers.csv import CSVReader
    from gwenflow.readers.directory import SimpleDirectoryReader
    from gwenflow.readers.docx import DocxReader
    from gwenflow.readers.excel import ExcelReader
    from gwenflow.readers.json import JSONReader
    from gwenflow.readers.pdf import PDFReader
    from gwenflow.readers.pptx import PptxReader
    from gwenflow.readers.text import TextReader
    from gwenflow.readers.website import WebsiteReader
    from gwenflow.readers.xml import XmlReader

_LAZY_READERS = {
    "CSVReader": "gwenflow.readers.csv",
    "DocxReader": "gwenflow.readers.docx",
    "ExcelReader": "gwenflow.readers.excel",
    "JSONReader": "gwenflow.readers.json",
    "PDFReader": "gwenflow.readers.pdf",
    "PptxReader": "gwenflow.readers.pptx",
    "SimpleDirectoryReader": "gwenflow.readers.directory",
    "TextReader": "gwenflow.readers.text",
    "WebsiteReader": "gwenflow.readers.website",
    "XmlReader": "gwenflow.readers.xml",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_READERS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache it so __getattr__ runs only once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_READERS))


__all__ = [
    "SimpleDirectoryReader",
    "TextReader",
    "JSONReader",
    "PDFReader",
    "WebsiteReader",
    "DocxReader",
    "ExcelReader",
    "CSVReader",
    "PptxReader",
    "XmlReader",
]
