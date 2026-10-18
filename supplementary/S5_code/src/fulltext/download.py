"""Phase 14, third component: verified-OA/CC full-text PDF cache.

Downloads and commits the actual PDF bytes to data/fulltext_pdfs/<record_id>.pdf
-- but ONLY when the resolved access route is independently confirmed either
(a) an arXiv preprint, or (b) Unpaywall/OpenAlex reports an explicit Creative
Commons license (a license string starting "cc", including "cc0"). Every
other resolvable-but-unverified OA link (bronze, publisher "free-to-read",
missing/unknown license, Semantic Scholar's openAccessPdf which carries no
license field at all) is left as a link only, never downloaded -- see
CLAUDE.md "Verified-OA/CC full-text cache" for the policy this encodes,
approved 2026-08-25 as a scoped, auditable exception to the project's
metadata-only default (Ethics and rate limiting: "Store only metadata and
abstracts -- not full paper PDFs (copyright)").

Deliberately does NOT reuse resolve.py's resolve_fulltext() -- that function
stops at the first HTTP-accessible URL regardless of license, which is the
wrong short-circuit here: a bronze/publisher-free-to-read URL can be
"accessible" (200 OK) while still being someone else's copyright. This
module re-walks the same source priority order but gates on license, not
just reachability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from review_pipeline.src.common.config import DATA_DIR
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.fulltext.arxiv_matching import find_arxiv_fallback
from review_pipeline.src.search.connectors import openalex, semantic_scholar, unpaywall

PDF_DIR = DATA_DIR / "fulltext_pdfs"

DownloadStatus = str  # "downloaded" | "oa_unverified_license" | "no_oa_found" | "oa_blocked_by_publisher" | "download_failed"

DOWNLOAD_STATUSES = ("downloaded", "oa_unverified_license", "oa_blocked_by_publisher", "no_oa_found", "download_failed")


@dataclass
class DownloadOutcome:
    record_id: str
    title: str
    status: DownloadStatus
    source: str | None = None
    license: str | None = None
    url: str | None = None
    pdf_path: str | None = None
    reason: str | None = None


def _is_cc_license(license_str: str | None) -> bool:
    return bool(license_str) and license_str.strip().lower().startswith("cc")


_BOT_BLOCK_STATUS_CODES = {401, 403, 429, 999}
# citation_pdf_url is a standard Highwire-Press-convention <meta> tag most
# scholarly publishers (Frontiers, MDPI, many society journals) embed on the
# article landing page specifically so indexers/readers can find the direct
# PDF -- reading it off HTML we already legitimately fetched is not a bypass
# of anything; it's the documented, intended way to locate the PDF link.
_CITATION_PDF_URL_RE = re.compile(
    r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']', re.IGNORECASE
)


@dataclass
class FetchAttempt:
    content: bytes | None
    reason: str


def _fetch_pdf_bytes(client: httpx.Client, url: str, timeout: float = 30.0, _via_citation_tag: bool = False) -> FetchAttempt:
    """Never spoofs a browser User-Agent to get past a 403 -- CLAUDE.md's
    Ethics rule ("User-Agent identifies this scraper, not a browser UA") is
    a deliberate honesty commitment, not just a default, so a publisher's
    bot-defense blocking an otherwise CC-licensed PDF is reported as exactly
    that (oa_blocked_by_publisher), not quietly bypassed."""
    try:
        response = client.get(url, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError as exc:
        return FetchAttempt(None, f"network error: {type(exc).__name__}")
    if response.status_code in _BOT_BLOCK_STATUS_CODES:
        return FetchAttempt(None, f"blocked (HTTP {response.status_code}) -- likely publisher bot-defense against a non-browser User-Agent")
    if response.status_code != 200:
        return FetchAttempt(None, f"HTTP {response.status_code}")
    content_type = response.headers.get("content-type", "").lower()
    looks_like_pdf = "pdf" in content_type or url.lower().endswith(".pdf") or response.content[:5] == b"%PDF-"
    if looks_like_pdf:
        return FetchAttempt(response.content, "ok")

    if not _via_citation_tag and "html" in content_type:
        match = _CITATION_PDF_URL_RE.search(response.text)
        if match:
            return _fetch_pdf_bytes(client, match.group(1), timeout=timeout, _via_citation_tag=True)

    return FetchAttempt(None, f"200 OK but not a PDF (content-type={content_type!r}) -- likely a landing page, no direct PDF link available")


def _save_pdf(record_id: str, content: bytes) -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    dest = PDF_DIR / f"{record_id}.pdf"
    dest.write_bytes(content)
    return dest


def resolve_and_download(engine, record_id: str, client: httpx.Client, contact_email: str) -> DownloadOutcome:
    """Walks sources in priority order (record's own oa_url -> Unpaywall ->
    OpenAlex -> Semantic Scholar -> arXiv), downloading the PDF at the first
    CC-licensed or arXiv match. Returns the best unverified-OA candidate
    found along the way if nothing qualified, or "no_oa_found" if nothing
    was accessible at all."""
    records_table = reflect_table(engine, "records")
    with engine.connect() as conn:
        record = dict(conn.execute(records_table.select().where(records_table.c.record_id == record_id)).mappings().one())

    title = record["title"]
    doi = record.get("doi")
    unverified: DownloadOutcome | None = None

    def _try_cc(source: str, url: str | None, license_: str | None) -> DownloadOutcome | None:
        nonlocal unverified
        if not url:
            return None
        if _is_cc_license(license_):
            attempt = _fetch_pdf_bytes(client, url)
            if attempt.content is not None:
                dest = _save_pdf(record_id, attempt.content)
                return DownloadOutcome(record_id, title, "downloaded", source, license_, url, str(dest.relative_to(PDF_DIR.parent)))
            status = "oa_blocked_by_publisher" if attempt.reason.startswith("blocked") else "download_failed"
            return DownloadOutcome(record_id, title, status, source, license_, url, reason=f"CC-licensed ({license_}) but {attempt.reason}")
        if unverified is None:
            unverified = DownloadOutcome(
                record_id, title, "oa_unverified_license", source, license_, url,
                reason=f"OA link found but license={license_!r} not confirmed Creative Commons",
            )
        return None

    # 1. Record's own enrichment-time oa_url (publisher-hosted) -- no
    # license field available for this source, so it can only ever
    # contribute an "unverified" candidate, never a download.
    if record.get("oa_url") and record.get("oa_status") in ("gold", "hybrid", "bronze"):
        unverified = DownloadOutcome(
            record_id, title, "oa_unverified_license", "publisher", None, record["oa_url"],
            reason=f"oa_status={record['oa_status']!r}, no license field available for this source",
        )

    if doi:
        up = unpaywall.fetch_by_doi(client, doi, contact_email)
        if up and up.oa_url:
            license_ = (up.raw_metadata.get("best_oa_location") or {}).get("license")
            result = _try_cc("unpaywall", up.oa_url, license_)
            if result:
                return result

        oa = openalex.fetch_by_doi(client, doi, contact_email)
        if oa and oa.oa_url:
            license_ = (oa.raw_metadata.get("best_oa_location") or {}).get("license")
            result = _try_cc("openalex", oa.oa_url, license_)
            if result:
                return result

        s2 = semantic_scholar.fetch_by_doi(client, doi, contact_email)
        if s2 and s2.oa_url and unverified is None:
            # S2's openAccessPdf carries no license field at all -- always unverified.
            unverified = DownloadOutcome(
                record_id, title, "oa_unverified_license", "semantic_scholar", None, s2.oa_url,
                reason="OA link found but Semantic Scholar exposes no license field",
            )

    # arXiv fallback -- always CC/arXiv-terms eligible per policy, last resort.
    arxiv_result = find_arxiv_fallback(engine, record_id, client)
    if arxiv_result.candidate and arxiv_result.candidate.pdf_url:
        attempt = _fetch_pdf_bytes(client, arxiv_result.candidate.pdf_url)
        if attempt.content is not None:
            dest = _save_pdf(record_id, attempt.content)
            return DownloadOutcome(
                record_id, title, "downloaded", "arxiv", "arxiv-license", arxiv_result.candidate.pdf_url,
                str(dest.relative_to(PDF_DIR.parent)),
            )
        status = "oa_blocked_by_publisher" if attempt.reason.startswith("blocked") else "download_failed"
        return DownloadOutcome(
            record_id, title, status, "arxiv", "arxiv-license", arxiv_result.candidate.pdf_url,
            reason=f"arXiv candidate found but {attempt.reason}",
        )

    if unverified is not None:
        return unverified
    return DownloadOutcome(record_id, title, "no_oa_found", reason="no OA location found via publisher/unpaywall/openalex/semantic_scholar/arxiv")


def get_target_record_ids(engine) -> list[dict]:
    """Records currently AI-screened as topically included -- the
    practical target set for a full-text pass ahead of Phase 11 human
    validation completing (which would otherwise leave ~1800 D1-eligible
    records, most not yet judged on-topic, as the only alternative scope)."""
    records_table = reflect_table(engine, "records")
    classifications_ai_table = reflect_table(engine, "classifications_ai")

    with engine.connect() as conn:
        # Latest label per record, not any label: re-screening under amended
        # criteria appends a new row rather than overwriting (the audit trail is
        # the point), so a record whose first pass said "include" and whose
        # re-screen said "exclude" still has an include row sitting in the
        # table. Reading every row would fetch full text for records the corpus
        # no longer includes.
        latest_label: dict[str, tuple[str, str]] = {}
        for row in conn.execute(
            classifications_ai_table.select().where(
                classifications_ai_table.c.dimension == "topical_inclusion"
            )
        ).mappings():
            previous = latest_label.get(row["record_id"])
            if previous is None or row["timestamp"] > previous[0]:
                latest_label[row["record_id"]] = (row["timestamp"], row["label"])
        included_ids = {rid for rid, (_ts, label) in latest_label.items() if label == "include"}
        if not included_ids:
            return []
        rows = conn.execute(
            records_table.select().where(
                records_table.c.record_id.in_(included_ids),
                records_table.c.d1 == 1,
                records_table.c.canonical_record_id.is_(None),
            )
        ).mappings().all()
    return [dict(r) for r in rows]


def write_coverage_report(outcomes: list[DownloadOutcome], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_status: dict[str, list[DownloadOutcome]] = {}
    for o in outcomes:
        by_status.setdefault(o.status, []).append(o)

    total = len(outcomes)
    downloaded = len(by_status.get("downloaded", []))
    lines = [
        "# Full-Text PDF Coverage Report",
        "",
        "Scope: records AI-screened `topical_inclusion=include` (Phase 10), D1-eligible, canonical.",
        (
            "Download policy: PDF committed to `data/fulltext_pdfs/` only when the resolved access route "
            "is confirmed Creative Commons (Unpaywall/OpenAlex `license` starting `cc`) or an arXiv preprint. "
            "Every other resolvable OA link is reported but left as a link only. See CLAUDE.md "
            '"Verified-OA/CC full-text cache".'
        ),
        "",
        f"**{downloaded}/{total} PDFs downloaded and committed.**",
        "",
        "| Status | Count |",
        "|---|---|",
    ]
    for status in DOWNLOAD_STATUSES:
        lines.append(f"| {status} | {len(by_status.get(status, []))} |")
    lines.append("")

    for status in DOWNLOAD_STATUSES:
        group = by_status.get(status, [])
        if not group:
            continue
        lines.append(f"## {status} ({len(group)})")
        lines.append("")
        lines.append("| record_id | title | source | license | url | reason |")
        lines.append("|---|---|---|---|---|---|")
        for o in group:
            title = o.title.replace("|", "\\|")[:90]
            lines.append(
                f"| {o.record_id} | {title} | {o.source or '-'} | {o.license or '-'} | "
                f"{o.url or '-'} | {o.reason or '-'} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
