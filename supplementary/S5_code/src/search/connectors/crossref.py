"""Crossref connector. https://api.crossref.org/works -- no API key required;
a contact email in the User-Agent/params opts into Crossref's "polite pool"
(higher, more reliable rate limits) per Crossref's own guidance.
"""

from __future__ import annotations

import re

import httpx

from review_pipeline.src.common.text_normalize import clean_text
from review_pipeline.src.search.connectors.base import (
    AuthorStub,
    ConnectorResponse,
    QueryFilters,
    RawRecord,
    SearchConnector,
)
from review_pipeline.src.search.connectors.http_client import build_client, get_with_retry

BASE_URL = "https://api.crossref.org/works"
MAX_ROWS = 1000  # Crossref's documented hard cap on `rows`

_PEER_REVIEWED_TYPES = {
    "journal-article": "peer_reviewed_journal",
    "proceedings-article": "peer_reviewed_conference",
    "book-chapter": "book_chapter_peer_reviewed",
}

# Crossref work types that are administrative/process artifacts, not scholarly
# works a literature review could ever cite -- some journals register peer
# review reports, decision letters, and author responses as their own citable
# DOIs. Discovered live: a single manuscript's review-process trail (title
# "Review for \"...\"", "Decision letter for \"...\"", "Author response for
# \"...\"") produced dozens of these per query, since they share the parent
# article's keywords/abstract text and match the same broad search terms.
# Excluded at the source rather than relying on dedup to clean them up after
# the fact -- dedup would only collapse them to ONE survivor, which would
# still be a peer-review artifact, not the actual paper.
_EXCLUDED_TYPES = {"peer-review", "grant"}


def _extract_abstract(item: dict) -> str | None:
    raw = item.get("abstract")
    if not raw:
        return None
    # Crossref abstracts are JATS-XML tagged (e.g. <jats:p>...</jats:p>); strip tags crudely
    # rather than pulling in a full XML parser for a field we treat as best-effort text.
    stripped = re.sub(r"<[^>]+>", " ", raw)
    return clean_text(stripped) or None


def _extract_year(item: dict) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        date_parts = item.get(key, {}).get("date-parts")
        if date_parts and date_parts[0] and date_parts[0][0]:
            return int(date_parts[0][0])
    return None


def _extract_authors(item: dict) -> list[AuthorStub]:
    authors = []
    for position, author in enumerate(item.get("author", []), start=1):
        given = author.get("given", "")
        family = author.get("family", "")
        if not family:
            continue
        full_name = f"{given} {family}".strip()
        authors.append(AuthorStub(full_name=full_name, surname=family, position=position))
    return authors


def _to_raw_record(item: dict) -> RawRecord:
    titles = item.get("title") or []
    title = titles[0] if titles else "(untitled)"
    container = item.get("container-title") or []
    doi = item.get("DOI")
    issn_list = item.get("ISSN") or []
    crossref_type = item.get("type")

    return RawRecord(
        doi=doi,
        title=clean_text(title),
        year=_extract_year(item),
        venue=clean_text(container[0]) if container else None,
        publication_type=_PEER_REVIEWED_TYPES.get(crossref_type, crossref_type),
        authors=_extract_authors(item),
        abstract=_extract_abstract(item),
        abstract_source="crossref" if item.get("abstract") else None,
        keywords=item.get("subject", []) or [],
        publisher=clean_text(item.get("publisher")),
        issn=issn_list[0] if issn_list else None,
        citation_count=item.get("is-referenced-by-count"),
        raw_source_id=doi or title,
        raw_metadata=item,
    )


def fetch_by_doi(client: httpx.Client, doi: str, contact_email: str) -> RawRecord | None:
    """Phase 7 enrichment lookup: GET /works/{doi} -- a single work, not a search."""
    result = get_with_retry(client, f"{BASE_URL}/{doi}", {"mailto": contact_email})
    if result.response is None or result.response.status_code != 200:
        return None
    item = result.response.json().get("message")
    if not item:
        return None
    return _to_raw_record(item)


class CrossrefConnector(SearchConnector):
    name = "crossref"

    def __init__(self, contact_email: str, client: httpx.Client | None = None) -> None:
        self.contact_email = contact_email
        self._client = client or build_client(contact_email)

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        params: dict = {
            "query.bibliographic": query,
            "rows": min(filters.rows, MAX_ROWS),
            "mailto": self.contact_email,
        }
        date_filters = []
        if filters.from_year:
            date_filters.append(f"from-pub-date:{filters.from_year}-01-01")
        if filters.to_year:
            date_filters.append(f"until-pub-date:{filters.to_year}-12-31")
        if date_filters:
            params["filter"] = ",".join(date_filters)
        params.update(filters.extra)

        result = get_with_retry(self._client, BASE_URL, params)
        if result.response is None:
            return ConnectorResponse(status="error", error_detail=result.error_detail, rate_limit_events=result.rate_limit_events)
        if result.error_detail and result.response.status_code >= 400:
            return ConnectorResponse(
                status="error",
                error_detail=result.error_detail,
                rate_limit_events=result.rate_limit_events,
            )

        payload = result.response.json()
        items = payload.get("message", {}).get("items", [])
        items = [item for item in items if item.get("type") not in _EXCLUDED_TYPES]
        records = [_to_raw_record(item) for item in items]
        status = "ok" if records else "zero_results"
        return ConnectorResponse(
            status=status,
            records=records,
            page_count=1,
            api_version=payload.get("message-version"),
            rate_limit_events=result.rate_limit_events,
        )
