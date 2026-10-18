"""Maps a connector-reported publication_type (and, as a fallback, DOI prefix)
onto the controlled publication_status vocabulary the schema enforces.

Discovered as a real gap while building Phase 8 deduplication: search/executor.py
was hardcoding publication_status='unknown' for every inserted record, which
silently defeats both canonical-record selection (peer-reviewed should always
outrank preprint) and preprint<->published pair detection (which requires at
least one side to actually be tagged 'preprint'). Conservative by design: an
unrecognized type is 'unknown', never guessed.
"""

from __future__ import annotations

from review_pipeline.src.common.text_normalize import normalize_doi

_PASSTHROUGH_STATUSES = {
    "peer_reviewed_journal",
    "peer_reviewed_conference",
    "book_chapter_peer_reviewed",
    "preprint",
    "technical_report",
    "standard",
    "documentation",
}

# Raw connector `type` values (Crossref, OpenAlex, Semantic Scholar) that the
# connectors' own _PEER_REVIEWED_TYPES maps don't cover, passed straight
# through as publication_type today -- mapped here to the schema's enum.
_PREPRINT_TYPE_SIGNALS = {"posted-content", "preprint", "Preprint"}
_TECHNICAL_REPORT_SIGNALS = {"report", "report-component", "Technical Report"}
_STANDARD_SIGNALS = {"standard"}

_ARXIV_DOI_PREFIX = "10.48550/arxiv"


def classify_publication_status(publication_type: str | None, doi: str | None = None) -> str:
    if publication_type in _PASSTHROUGH_STATUSES:
        return publication_type  # type: ignore[return-value]
    if publication_type in _PREPRINT_TYPE_SIGNALS:
        return "preprint"
    if publication_type in _TECHNICAL_REPORT_SIGNALS:
        return "technical_report"
    if publication_type in _STANDARD_SIGNALS:
        return "standard"

    normalized_doi = normalize_doi(doi)
    if normalized_doi and normalized_doi.startswith(_ARXIV_DOI_PREFIX):
        return "preprint"

    return "unknown"
