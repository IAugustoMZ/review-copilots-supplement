"""Phase 14, first half: OA full-text resolution.

Walks sources in priority order, stopping at the first genuinely accessible
result (verified with a real HTTP request, not just "a source's metadata
claimed this was OA" -- that distinguishes "we checked and it was blocked"
from "we never checked", per docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 9).
Writes one fulltext_locations row per attempt, not just the winner.

Only ever resolves already-open, legally accessible content -- never bypasses
access controls, per CLAUDE.md "Ethics and rate limiting". Institutional
repository / author-accepted-manuscript / PMC are NOT implemented (no
connector exists for any of them) and are honestly omitted from the attempt
log rather than logged as a fake "checked and failed" row -- see
UNIMPLEMENTED_SOURCES below, surfaced in ResolveSummary so this gap is
visible, not silently hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
from sqlalchemy import Engine

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.fulltext.arxiv_matching import find_arxiv_fallback
from review_pipeline.src.search.connectors import openalex, semantic_scholar, unpaywall

UNIMPLEMENTED_SOURCES = ["institutional_repository", "author_accepted_manuscript", "pmc"]


def _check_url_accessible(client: httpx.Client, url: str, timeout: float = 10.0) -> bool:
    try:
        response = client.head(url, timeout=timeout, follow_redirects=True)
        if response.status_code == 405:  # some hosts reject HEAD; fall back to a light GET
            response = client.get(url, timeout=timeout, follow_redirects=True)
        return 200 <= response.status_code < 300
    except httpx.HTTPError:
        return False


@dataclass
class FulltextAttempt:
    source: str
    url: str | None
    accessible: bool
    license: str | None = None
    access_method: str | None = None


@dataclass
class ResolveSummary:
    record_id: str
    attempts: list[FulltextAttempt] = field(default_factory=list)
    resolved_source: str | None = None
    arxiv_fallback_used: bool = False
    unimplemented_sources: list[str] = field(default_factory=lambda: list(UNIMPLEMENTED_SOURCES))


def resolve_fulltext(engine: Engine, record_id: str, client: httpx.Client, contact_email: str) -> ResolveSummary:
    records_table = reflect_table(engine, "records")
    fulltext_locations = reflect_table(engine, "fulltext_locations")

    with engine.connect() as conn:
        record = dict(conn.execute(records_table.select().where(records_table.c.record_id == record_id)).mappings().one())

    summary = ResolveSummary(record_id=record_id)
    now = datetime.now(UTC).isoformat()

    def _log_and_check(source: str, url: str | None, license_: str | None) -> bool:
        if not url:
            return False
        accessible = _check_url_accessible(client, url)
        attempt = FulltextAttempt(source=source, url=url, accessible=accessible, license=license_, access_method="http_check")
        summary.attempts.append(attempt)
        with engine.begin() as conn:
            conn.execute(
                fulltext_locations.insert().values(
                    fulltext_location_id=new_id(),
                    record_id=record_id,
                    source=source,
                    url=url,
                    license=license_,
                    resolved_at=now,
                    accessible=1 if accessible else 0,
                    access_method="http_check",
                )
            )
        return accessible

    # 1. publisher OA -- whatever oa_url is already stored on the record (from
    # enrichment), if it looks publisher-hosted (gold/hybrid/bronze).
    publisher_like_oa = record.get("oa_url") and record.get("oa_status") in ("gold", "hybrid", "bronze")
    if publisher_like_oa and _log_and_check("publisher", record["oa_url"], None):
        summary.resolved_source = "publisher"
        return summary

    doi = record.get("doi")
    if doi:
        # 2. Unpaywall
        up = unpaywall.fetch_by_doi(client, doi, contact_email)
        if up and _log_and_check("unpaywall", up.oa_url, None):
            summary.resolved_source = "unpaywall"
            return summary

        # 3. OpenAlex best_oa_location
        oa = openalex.fetch_by_doi(client, doi, contact_email)
        if oa and _log_and_check("openalex", oa.oa_url, None):
            summary.resolved_source = "openalex"
            return summary

        # 4. Semantic Scholar openAccessPdf
        s2 = semantic_scholar.fetch_by_doi(client, doi, contact_email)
        if s2 and _log_and_check("semantic_scholar", s2.oa_url, None):
            summary.resolved_source = "semantic_scholar"
            return summary
    # else: Unpaywall/OpenAlex/S2 lookups all key on DOI -- skipped, but arXiv
    # fallback below is title-based and still runs regardless of DOI presence.

    # institutional_repository / author_accepted_manuscript / pmc: no connector,
    # honestly skipped (see UNIMPLEMENTED_SOURCES).

    # 5. arXiv fallback -- last resort, access problems only.
    summary.arxiv_fallback_used = True
    arxiv_result = find_arxiv_fallback(engine, record_id, client)
    has_arxiv_pdf = arxiv_result.auto_linked and arxiv_result.candidate
    if has_arxiv_pdf and _log_and_check("arxiv", arxiv_result.candidate.pdf_url, None):
        summary.resolved_source = "arxiv"

    return summary
