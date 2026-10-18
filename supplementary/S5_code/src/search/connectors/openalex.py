"""OpenAlex connector. https://api.openalex.org/works -- no API key required;
a `mailto` param opts into OpenAlex's polite pool.
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
from review_pipeline.src.search.connectors.http_client import build_client, get_with_retry

BASE_URL = "https://api.openalex.org/works"
MAX_PER_PAGE = 200  # OpenAlex hard-rejects per-page > 200 with an HTTP 400

_PEER_REVIEWED_TYPES = {
    "article": "peer_reviewed_journal",
    "proceedings-article": "peer_reviewed_conference",
    "book-chapter": "book_chapter_peer_reviewed",
}


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]} to sidestep publisher
    copyright on the running text; rebuild the plain text from word positions."""
    if not inverted_index:
        return None
    position_to_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_to_word[pos] = word
    if not position_to_word:
        return None
    ordered = [position_to_word[i] for i in sorted(position_to_word)]
    return " ".join(ordered)


def _extract_authors(work: dict) -> list[AuthorStub]:
    authors = []
    for position, authorship in enumerate(work.get("authorships", []), start=1):
        author = authorship.get("author", {})
        full_name = author.get("display_name")
        if not full_name:
            continue
        surname = full_name.split()[-1]
        authors.append(AuthorStub(full_name=full_name, surname=surname, position=position))
    return authors


def _to_raw_record(work: dict) -> RawRecord:
    doi = work.get("doi")
    if doi:
        doi = doi.removeprefix("https://doi.org/")
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    open_access = work.get("open_access") or {}
    openalex_type = work.get("type")
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

    return RawRecord(
        doi=doi,
        title=work.get("title") or work.get("display_name") or "(untitled)",
        year=work.get("publication_year"),
        venue=source.get("display_name"),
        publication_type=_PEER_REVIEWED_TYPES.get(openalex_type, openalex_type),
        authors=_extract_authors(work),
        abstract=abstract,
        abstract_source="openalex" if abstract else None,
        keywords=[kw.get("display_name") for kw in work.get("keywords", []) if kw.get("display_name")],
        publisher=source.get("host_organization_name"),
        issn=(source.get("issn") or [None])[0],
        citation_count=work.get("cited_by_count"),
        oa_status=open_access.get("oa_status"),
        oa_url=open_access.get("oa_url"),
        # `or`-chain, not `.get(key, default)` -- see the identical fix/comment
        # in semantic_scholar._to_raw_record for why a present-but-null "id"
        # must fall through to the next fallback instead of validating as None.
        raw_source_id=work.get("id") or doi or work.get("title") or "",
        raw_metadata=work,
    )


def build_and_of_or_filter(term_groups: list[list[str]]) -> str:
    """OpenAlex's `search` param silently returns zero results once a query
    string carries too many quoted-phrase OR terms -- confirmed live: a
    9-term query returns real hits, a 26-30 term query (our CE1-14 family
    queries: ~12 genai terms + ~14 domain terms + a handful of theme terms)
    returns meta.count=0 with no error, no warning. `filter=title_and_abstract
    .search:` stays correct at any term count: `|` is OR within a group, `,`
    is AND between groups. Used with [genai_terms, domain_terms] to mirror
    matches_genai_domain()'s own semantics exactly, rather than the single
    giant OR group (confirmed live to over-match on any one common word like
    "process" and return millions of irrelevant results)."""
    groups = []
    for terms in term_groups:
        if not terms:
            continue
        escaped = [t.replace("|", " ").replace(",", " ") for t in terms]
        groups.append("title_and_abstract.search:" + "|".join(escaped))
    return ",".join(groups)


def fetch_by_doi(client: httpx.Client, doi: str, contact_email: str) -> RawRecord | None:
    """Phase 7 enrichment lookup: GET /works/doi:{doi} -- a single work, not a search."""
    result = get_with_retry(client, f"{BASE_URL}/doi:{doi}", {"mailto": contact_email})
    if result.response is None or result.response.status_code != 200:
        return None
    work = result.response.json()
    if not work or "id" not in work:
        return None
    return _to_raw_record(work)


# Citation-graph snowballing (Phase 19). Confirmed live against the real API
# (not just the docs) that both filters below behave as expected:
# `openalex_id:ID1|ID2|...` as an OR-list returns exactly those works, and
# `cites:ID` returns works whose reference list includes ID -- see the
# CE1-14 `title_and_abstract.search` bug this project shipped once already
# from an unverified assumption about OpenAlex filter syntax.
CITATION_CHUNK_SIZE = 50  # OpenAlex's documented cap on OR-list items in one filter


def fetch_referenced_works(client: httpx.Client, contact_email: str, referenced_work_ids: list[str]) -> ConnectorResponse:
    """Backward snowballing: batch-fetches full metadata for the works a seed
    work cites, given the `referenced_works` list from that seed's own OpenAlex
    record (full URLs, e.g. "https://openalex.org/W123" -- stripped to the short
    form the `openalex_id` filter expects). Chunked at CITATION_CHUNK_SIZE."""
    if not referenced_work_ids:
        return ConnectorResponse(status="zero_results", records=[])
    short_ids = [rid.rsplit("/", 1)[-1] for rid in referenced_work_ids]
    records: list[RawRecord] = []
    rate_limit_events = 0
    for i in range(0, len(short_ids), CITATION_CHUNK_SIZE):
        chunk = short_ids[i : i + CITATION_CHUNK_SIZE]
        params = {"filter": "openalex_id:" + "|".join(chunk), "per-page": len(chunk), "mailto": contact_email}
        result = get_with_retry(client, BASE_URL, params)
        rate_limit_events += result.rate_limit_events
        if result.response is None or (result.error_detail and result.response.status_code >= 400):
            return ConnectorResponse(status="error", error_detail=result.error_detail, records=records, rate_limit_events=rate_limit_events)
        payload = result.response.json()
        records.extend(_to_raw_record(work) for work in payload.get("results", []))
    status = "ok" if records else "zero_results"
    return ConnectorResponse(status=status, records=records, page_count=1, rate_limit_events=rate_limit_events)


def fetch_citing_works(client: httpx.Client, contact_email: str, openalex_work_id: str) -> ConnectorResponse:
    """Forward snowballing: works that cite `openalex_work_id`. One page of up
    to MAX_PER_PAGE -- a highly-cited seed can have more citing works than that;
    this is a recall-improving pass, not an exhaustive one, matching how CE1-14
    search itself only ever fetches one page per query."""
    short_id = openalex_work_id.rsplit("/", 1)[-1]
    params = {"filter": f"cites:{short_id}", "per-page": MAX_PER_PAGE, "mailto": contact_email}
    result = get_with_retry(client, BASE_URL, params)
    if result.response is None or (result.error_detail and result.response.status_code >= 400):
        return ConnectorResponse(status="error", error_detail=result.error_detail, rate_limit_events=result.rate_limit_events)
    payload = result.response.json()
    records = [_to_raw_record(work) for work in payload.get("results", [])]
    status = "ok" if records else "zero_results"
    return ConnectorResponse(
        status=status,
        records=records,
        page_count=1,
        api_version=payload.get("meta", {}).get("db_version"),
        rate_limit_events=result.rate_limit_events,
    )


class OpenAlexConnector(SearchConnector):
    name = "openalex"

    def __init__(self, contact_email: str, client: httpx.Client | None = None) -> None:
        self.contact_email = contact_email
        self._client = client or build_client(contact_email)

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        params: dict = {
            "per-page": min(filters.rows, MAX_PER_PAGE),
            "mailto": self.contact_email,
        }
        filter_parts = []
        if filters.from_year:
            filter_parts.append(f"from_publication_date:{filters.from_year}-01-01")
        if filters.to_year:
            filter_parts.append(f"to_publication_date:{filters.to_year}-12-31")

        # boolean_filter (see build_and_of_or_filter) replaces the free-text
        # `search` param for callers with many-term queries -- see
        # build_and_of_or_filter's docstring for why `search` alone breaks
        # there. `relevance_hint`, if given, is a short (2-4 term) stand-in
        # `search` string used ONLY for OpenAlex's relevance ranking within
        # the results the filter already restricted to -- confirmed live
        # that reusing the full many-term query as the hint over-restricts
        # (the `search` param requires matches too, not just ranks) and that
        # omitting it entirely falls back to recency sort, which surfaces
        # low-quality/junk-metadata records ahead of real papers. Without a
        # hint, sort by recency instead (no relevance score exists to sort
        # by when `search` is absent).
        boolean_filter = filters.extra.get("boolean_filter")
        if boolean_filter:
            filter_parts.append(boolean_filter)
            relevance_hint = filters.extra.get("relevance_hint")
            if relevance_hint:
                params["search"] = relevance_hint
            else:
                params["sort"] = "publication_date:desc"
        else:
            params["search"] = query

        if filter_parts:
            params["filter"] = ",".join(filter_parts)
        _internal_keys = {"boolean_filter", "relevance_hint"}
        params.update({k: v for k, v in filters.extra.items() if k not in _internal_keys})

        result = get_with_retry(self._client, BASE_URL, params)
        if result.response is None or (result.error_detail and result.response.status_code >= 400):
            return ConnectorResponse(status="error", error_detail=result.error_detail, rate_limit_events=result.rate_limit_events)

        payload = result.response.json()
        results = payload.get("results", [])
        records = [_to_raw_record(work) for work in results]
        status = "ok" if records else "zero_results"
        return ConnectorResponse(
            status=status,
            records=records,
            page_count=1,
            api_version=payload.get("meta", {}).get("db_version"),
            rate_limit_events=result.rate_limit_events,
        )
