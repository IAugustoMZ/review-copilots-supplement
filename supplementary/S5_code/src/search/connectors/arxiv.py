"""arXiv connector -- fulltext-access fallback ONLY (Phase 14), never a
general search backbone. See CLAUDE.md: "Google Scholar is supplementary...
arXiv" plays the same non-primary role here -- it exists to resolve an
ACCESS problem for an already-discovered, already-peer-reviewed record with
no other legal OA location, never to discover new corpus candidates.
https://export.arxiv.org/api/query -- no API key, Atom XML response.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

from review_pipeline.src.search.connectors.http_client import get_with_retry

BASE_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


@dataclass
class ArxivCandidate:
    arxiv_id: str
    title: str
    author_surnames: list[str] = field(default_factory=list)
    year: int | None = None
    journal_ref: str | None = None
    doi: str | None = None
    pdf_url: str | None = None
    abstract: str | None = None


def _parse_entry(entry: ET.Element) -> ArxivCandidate:
    entry_id = (entry.findtext(f"{_ATOM_NS}id") or "").strip()
    arxiv_id = entry_id.rsplit("/abs/", 1)[-1] if "/abs/" in entry_id else entry_id
    title = " ".join((entry.findtext(f"{_ATOM_NS}title") or "").split())
    summary = " ".join((entry.findtext(f"{_ATOM_NS}summary") or "").split()) or None

    authors = []
    for author_el in entry.findall(f"{_ATOM_NS}author"):
        name = (author_el.findtext(f"{_ATOM_NS}name") or "").strip()
        if name:
            authors.append(name.split()[-1])

    published = entry.findtext(f"{_ATOM_NS}published") or ""
    year = int(published[:4]) if published[:4].isdigit() else None

    journal_ref = entry.findtext(f"{_ARXIV_NS}journal_ref")
    doi = entry.findtext(f"{_ARXIV_NS}doi")

    pdf_url = None
    for link in entry.findall(f"{_ATOM_NS}link"):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href")
            break

    return ArxivCandidate(
        arxiv_id=arxiv_id,
        title=title,
        author_surnames=authors,
        year=year,
        journal_ref=journal_ref,
        doi=doi,
        pdf_url=pdf_url,
        abstract=summary,
    )


def search_arxiv_by_title(client: httpx.Client, title: str, max_results: int = 5) -> list[ArxivCandidate]:
    """Title-based candidate search. Returns [] on any error/zero-results --
    callers treat "no arXiv candidate" as a normal, expected outcome, not a
    failure worth surfacing separately."""
    query = f'ti:"{title}"'
    result = get_with_retry(client, BASE_URL, {"search_query": query, "max_results": max_results})
    if result.response is None or result.response.status_code != 200:
        return []
    try:
        root = ET.fromstring(result.response.text)
    except ET.ParseError:
        return []
    return [_parse_entry(entry) for entry in root.findall(f"{_ATOM_NS}entry")]
