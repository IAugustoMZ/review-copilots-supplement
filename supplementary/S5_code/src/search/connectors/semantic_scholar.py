"""Semantic Scholar connector. https://api.semanticscholar.org/graph/v1/paper/search
An API key is optional (raises rate limits only); pass via `api_key`.
"""

from __future__ import annotations

import httpx

from review_pipeline.src.search.connectors.base import (
    AuthorStub,
    ConnectorResponse,
    QueryFilters,
    RawRecord,
    SearchConnector,
)
from review_pipeline.src.search.connectors.http_client import (
    ANONYMOUS_POOL_RETRY,
    RetryPolicy,
    build_client,
    get_with_retry,
)


def _retry_policy(client: httpx.Client) -> RetryPolicy | None:
    """Wait out contention on the shared anonymous pool, but not on a keyed
    budget: with `x-api-key` set, a 429 means this client really is over its
    own 1 req/s allowance, and retrying harder just burns the next second's
    budget too. Keyed callers therefore keep the default policy.
    """
    return None if client.headers.get("x-api-key") else ANONYMOUS_POOL_RETRY


BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,venue,publicationTypes,externalIds,authors,openAccessPdf,citationCount,publicationVenue"
MAX_LIMIT = 100  # Semantic Scholar's /paper/search hard-caps `limit` at 100

_PEER_REVIEWED_TYPES = {
    "JournalArticle": "peer_reviewed_journal",
    "Conference": "peer_reviewed_conference",
}


def _extract_authors(paper: dict) -> list[AuthorStub]:
    authors = []
    for position, author in enumerate(paper.get("authors", []), start=1):
        full_name = author.get("name")
        if not full_name:
            continue
        surname = full_name.split()[-1]
        authors.append(AuthorStub(full_name=full_name, surname=surname, position=position))
    return authors


def _to_raw_record(paper: dict) -> RawRecord:
    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI")
    pub_types = paper.get("publicationTypes") or []
    mapped_type = next((_PEER_REVIEWED_TYPES[t] for t in pub_types if t in _PEER_REVIEWED_TYPES), None)
    open_access_pdf = paper.get("openAccessPdf") or {}
    abstract = paper.get("abstract")

    return RawRecord(
        doi=doi,
        title=paper.get("title") or "(untitled)",
        year=paper.get("year"),
        venue=paper.get("venue"),
        publication_type=mapped_type or (pub_types[0] if pub_types else None),
        authors=_extract_authors(paper),
        abstract=abstract,
        abstract_source="semantic_scholar" if abstract else None,
        citation_count=paper.get("citationCount"),
        oa_url=open_access_pdf.get("url"),
        oa_status="open" if open_access_pdf.get("url") else None,
        # `.get(key, default)` only falls through when the key is absent --
        # S2's /references and /citations endpoints commonly return an
        # unmatched reference as {"paperId": None, ...} (present, explicitly
        # null), which `.get` would pass straight through and fail
        # RawRecord's str-typed field. `or`-chain instead so None values fall
        # through too. Confirmed live: this is common enough in real
        # reference lists to crash a multi-seed snowball run on the first hit.
        raw_source_id=paper.get("paperId") or doi or paper.get("title") or "",
        raw_metadata=paper,
    )


DOI_LOOKUP_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"


def fetch_by_doi(client: httpx.Client, doi: str, contact_email: str) -> RawRecord | None:
    """Phase 7 enrichment lookup: GET /paper/DOI:{doi} -- a single work, not a search.

    contact_email is unused (S2's DOI lookup needs no mailto param) but kept in the
    signature so all three fetch_by_doi functions share one call shape for the enricher.
    """
    result = get_with_retry(client, DOI_LOOKUP_URL.format(doi=doi), {"fields": FIELDS}, policy=_retry_policy(client))
    if result.response is None or result.response.status_code != 200:
        return None
    paper = result.response.json()
    if not paper or "paperId" not in paper:
        return None
    return _to_raw_record(paper)


# Citation-graph snowballing (Phase 19). Confirmed live: /references wraps each
# item as {"citedPaper": {...requested citedPaper.* fields}}, /citations as
# {"citingPaper": {...}} -- same field names as a normal /paper/search result,
# just field-prefixed per S2's API, so _to_raw_record works unmodified on the
# nested dict.
_REFERENCE_FIELD_NAMES = ["title", "abstract", "year", "venue", "publicationTypes", "externalIds", "authors", "openAccessPdf", "citationCount"]
REFERENCES_FIELDS = ",".join(f"citedPaper.{f}" for f in _REFERENCE_FIELD_NAMES)
CITATIONS_FIELDS = ",".join(f"citingPaper.{f}" for f in _REFERENCE_FIELD_NAMES)
REFERENCES_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/references"
CITATIONS_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/citations"


def fetch_references(client: httpx.Client, doi: str, limit: int = MAX_LIMIT) -> ConnectorResponse:
    """Backward snowballing: the papers `doi` cites."""
    result = get_with_retry(
        client,
        REFERENCES_URL.format(doi=doi),
        {"fields": REFERENCES_FIELDS, "limit": min(limit, MAX_LIMIT)},
        policy=_retry_policy(client),
    )
    if result.response is None or (result.error_detail and result.response.status_code >= 400):
        return ConnectorResponse(status="error", error_detail=result.error_detail, rate_limit_events=result.rate_limit_events)
    # `or []`, not the `.get(key, default)` default alone -- confirmed live
    # that S2 can return {"data": null} for a paper with no references.
    data = result.response.json().get("data") or []
    records = [_to_raw_record(item["citedPaper"]) for item in data if item.get("citedPaper")]
    status = "ok" if records else "zero_results"
    return ConnectorResponse(status=status, records=records, page_count=1, rate_limit_events=result.rate_limit_events)


def fetch_citations(client: httpx.Client, doi: str, limit: int = MAX_LIMIT) -> ConnectorResponse:
    """Forward snowballing: papers that cite `doi`. One page of up to MAX_LIMIT --
    a recall-improving pass, not an exhaustive one (see openalex.fetch_citing_works)."""
    result = get_with_retry(
        client,
        CITATIONS_URL.format(doi=doi),
        {"fields": CITATIONS_FIELDS, "limit": min(limit, MAX_LIMIT)},
        policy=_retry_policy(client),
    )
    if result.response is None or (result.error_detail and result.response.status_code >= 400):
        return ConnectorResponse(status="error", error_detail=result.error_detail, rate_limit_events=result.rate_limit_events)
    data = result.response.json().get("data") or []
    records = [_to_raw_record(item["citingPaper"]) for item in data if item.get("citingPaper")]
    status = "ok" if records else "zero_results"
    return ConnectorResponse(status=status, records=records, page_count=1, rate_limit_events=result.rate_limit_events)


class SemanticScholarConnector(SearchConnector):
    name = "semantic_scholar"

    def __init__(self, contact_email: str, api_key: str | None = None, client: httpx.Client | None = None) -> None:
        self.contact_email = contact_email
        extra_headers = {"x-api-key": api_key} if api_key else None
        self._client = client or build_client(contact_email, extra_headers=extra_headers)

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        params: dict = {"query": query, "limit": min(filters.rows, MAX_LIMIT), "fields": FIELDS}
        if filters.from_year or filters.to_year:
            lo = filters.from_year or ""
            hi = filters.to_year or ""
            params["year"] = f"{lo}-{hi}"
        params.update(filters.extra)

        result = get_with_retry(self._client, BASE_URL, params, policy=_retry_policy(self._client))
        if result.response is None or (result.error_detail and result.response.status_code >= 400):
            return ConnectorResponse(status="error", error_detail=result.error_detail, rate_limit_events=result.rate_limit_events)

        payload = result.response.json()
        # `or []`, not the `.get(key, default)` default alone -- S2 can
        # return {"data": null} (confirmed live via /references and
        # /citations; the same shape is a live risk here too), which `.get`
        # would pass through as None and crash the list comprehension below.
        data = payload.get("data") or []
        records = [_to_raw_record(paper) for paper in data]
        status = "ok" if records else "zero_results"
        return ConnectorResponse(
            status=status,
            records=records,
            page_count=1,
            rate_limit_events=result.rate_limit_events,
        )
