import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Optional

import requests
from pydantic import Field

from gwenflow.logger import logger
from gwenflow.tools.tool import BaseTool

PUBMED_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


@dataclass(kw_only=True)
class PubMedBase(BaseTool):
    api_key: Optional[str] = None
    email: Optional[str] = None
    tool_name: str = "gwenflow"
    top_k_results: int = 5
    timeout: int = 30

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("NCBI_API_KEY")
        if self.email is None:
            self.email = os.getenv("NCBI_EMAIL")
        super().__post_init__()

    def _params(self, **extra: Any) -> dict[str, Any]:
        params: dict[str, Any] = {"tool": self.tool_name}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        params.update(extra)
        return params

    def _esearch(self, query: str, retmax: int) -> list[str]:
        url = f"{PUBMED_EUTILS_BASE}/esearch.fcgi"
        params = self._params(db="pubmed", term=query, retmax=retmax, retmode="json")
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _efetch(self, pmids: list[str]) -> ET.Element:
        url = f"{PUBMED_EUTILS_BASE}/efetch.fcgi"
        params = self._params(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return ET.fromstring(response.content)

    @staticmethod
    def _text(node: Optional[ET.Element]) -> str:
        if node is None:
            return ""
        return "".join(node.itertext()).strip()

    @classmethod
    def _parse_article(cls, article: ET.Element) -> dict[str, Any]:
        pmid = cls._text(article.find(".//PMID"))
        title = cls._text(article.find(".//Article/ArticleTitle"))

        abstract_parts: list[str] = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            text = cls._text(ab)
            if not text:
                continue
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(abstract_parts)

        authors: list[str] = []
        for author in article.findall(".//AuthorList/Author"):
            last = cls._text(author.find("LastName"))
            initials = cls._text(author.find("Initials"))
            collective = cls._text(author.find("CollectiveName"))
            if last:
                authors.append(f"{last} {initials}".strip())
            elif collective:
                authors.append(collective)

        journal = cls._text(article.find(".//Journal/Title"))
        year = cls._text(article.find(".//Journal/JournalIssue/PubDate/Year"))
        if not year:
            medline_date = cls._text(article.find(".//Journal/JournalIssue/PubDate/MedlineDate"))
            year = medline_date[:4] if medline_date else ""

        doi = ""
        for article_id in article.findall(".//ArticleIdList/ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = cls._text(article_id)
                break

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "year": year,
            "doi": doi,
            "url": PUBMED_ARTICLE_URL.format(pmid=pmid) if pmid else "",
        }


@dataclass(kw_only=True)
class PubMedTool(PubMedBase):
    name: str = "PubMedTool"
    description: str = (
        "Search PubMed for biomedical and life-sciences literature. "
        "Returns title, abstract, authors, journal, year, DOI, and PubMed URL "
        "for the top matching articles. Input should be a search query "
        "(e.g. 'CRISPR base editing 2024' or 'metformin AND diabetes')."
    )

    def _run(self, query: str = Field(description="PubMed search query")):
        pmids = self._esearch(query=query, retmax=self.top_k_results)
        if not pmids:
            raise ValueError(f"No PubMed results found for query: {query!r}")

        try:
            root = self._efetch(pmids)
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML response: {e}")
            raise

        articles = [self._parse_article(article) for article in root.findall(".//PubmedArticle")]
        if not articles:
            raise ValueError(f"PubMed returned PMIDs but no parseable articles for query: {query!r}")
        return articles
