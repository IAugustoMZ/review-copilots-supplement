"""Title/DOI normalization.

Used both at record-insert time (records.title_normalized, populated by every
connector-ingesting path) and by the deduplication cascade (src/deduplication),
which is the reason these live in `common` rather than inside `deduplication` --
one normalization definition, reused everywhere a normalized value is needed.
See docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 7.
"""

from __future__ import annotations

import html
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_HYPHENATION_RE = re.compile(r"(\w)-\s*\n?\s*(\w)")


def normalize_title(title: str) -> str:
    """Case/punctuation/Unicode/hyphenation/whitespace/HTML-entity/subtitle-punctuation
    insensitive normalization, for exact and fuzzy duplicate matching."""
    if not title:
        return ""
    text = html.unescape(title)
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHENATION_RE.sub(r"\1\2", text)  # de-hyphenate line-broken words
    text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)  # strips subtitle colons/periods/etc too
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_text(text: str | None) -> str | None:
    """HTML-entity-unescape and whitespace-collapse human-facing text (title,
    venue, publisher, abstract) without altering case or punctuation -- unlike
    normalize_title(), this is for display/storage, not matching. Source APIs
    (Crossref's container-title in particular) sometimes carry literal HTML
    entities (e.g. "Industrial &amp; Engineering..."), a real issue caught via
    Phase 7 live enrichment testing."""
    if not text:
        return text
    unescaped = html.unescape(text)
    return _WHITESPACE_RE.sub(" ", unescaped).strip()


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    text = doi.strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = text.strip().rstrip(".,;")
    return text or None
