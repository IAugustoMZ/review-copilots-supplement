"""Pure matching functions for Phase 8 deduplication -- no DB access, so the
matching logic is directly unit-testable. See
docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 7: "DOI exact -> title exact
(normalized) -> fuzzy title+author/year -> explicit preprint-published
linkage", each merge logged with a reason and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from review_pipeline.src.common.text_normalize import normalize_doi

# publication_status enum, best (most authoritative) first -- used to pick the
# canonical record within a duplicate group.
PUBLICATION_STATUS_RANK = {
    "peer_reviewed_journal": 0,
    "peer_reviewed_conference": 1,
    "book_chapter_peer_reviewed": 2,
    "preprint": 3,
    "technical_report": 4,
    "standard": 5,
    "documentation": 6,
    "unknown": 7,
}

FUZZY_TITLE_THRESHOLD = 0.90
YEAR_TOLERANCE = 1

PREPRINT_HIGH_THRESHOLD = 0.95
PREPRINT_MEDIUM_THRESHOLD = 0.85
PREPRINT_LOW_THRESHOLD = 0.75


@dataclass(frozen=True)
class DedupRecord:
    """A `records` row projected down to what matching needs, plus author surnames."""

    record_id: str
    doi: str | None
    title: str
    title_normalized: str
    year: int | None
    publication_status: str
    author_surnames: tuple[str, ...] = ()


def fuzzy_title_score(title_a: str, title_b: str) -> float:
    """0.0-1.0 similarity via token-sort ratio (word-order-insensitive)."""
    if not title_a or not title_b:
        return 0.0
    return fuzz.token_sort_ratio(title_a, title_b) / 100.0


def years_close(year_a: int | None, year_b: int | None, tolerance: int = YEAR_TOLERANCE) -> bool:
    if year_a is None or year_b is None:
        return False
    return abs(year_a - year_b) <= tolerance


def authors_overlap(surnames_a: tuple[str, ...], surnames_b: tuple[str, ...]) -> bool:
    if not surnames_a or not surnames_b:
        return False
    set_a = {s.lower() for s in surnames_a}
    set_b = {s.lower() for s in surnames_b}
    return bool(set_a & set_b)


def doi_exact_match(a: DedupRecord, b: DedupRecord) -> bool:
    doi_a, doi_b = normalize_doi(a.doi), normalize_doi(b.doi)
    return doi_a is not None and doi_a == doi_b


def title_exact_match(a: DedupRecord, b: DedupRecord) -> bool:
    return bool(a.title_normalized) and a.title_normalized == b.title_normalized


def title_fuzzy_match(a: DedupRecord, b: DedupRecord, threshold: float = FUZZY_TITLE_THRESHOLD) -> tuple[bool, float]:
    """Fuzzy title similarity alone is not sufficient -- also require year
    proximity or author-surname overlap, to avoid merging two distinct papers
    that happen to share a generic title pattern."""
    score = fuzzy_title_score(a.title_normalized, b.title_normalized)
    if score < threshold:
        return False, score
    corroborated = years_close(a.year, b.year) or authors_overlap(a.author_surnames, b.author_surnames)
    return corroborated, score


def preprint_published_confidence(a: DedupRecord, b: DedupRecord) -> tuple[str | None, dict]:
    """Given one record with publication_status == 'preprint' and one
    peer-reviewed record, return (confidence, evidence) where confidence is
    one of 'high'/'medium'/'low'/None (None = not a plausible pair at all).
    """
    title_sim = fuzzy_title_score(a.title_normalized, b.title_normalized)
    year_proximity = years_close(a.year, b.year)
    author_overlap = authors_overlap(a.author_surnames, b.author_surnames)
    evidence = {
        "title_similarity": round(title_sim, 4),
        "year_proximity": year_proximity,
        "author_overlap": author_overlap,
    }

    if title_sim >= PREPRINT_HIGH_THRESHOLD and year_proximity and author_overlap:
        return "high", evidence
    if title_sim >= PREPRINT_MEDIUM_THRESHOLD and (year_proximity or author_overlap):
        return "medium", evidence
    if title_sim >= PREPRINT_LOW_THRESHOLD:
        return "low", evidence
    return None, evidence


def title_first_token(title_normalized: str) -> str:
    """Blocking key for preprint<->published matching, which must NOT depend on
    year proximity (a preprint can predate its publication by several years,
    unlike ordinary cross-connector duplicates of the same search result)."""
    return title_normalized.split()[0] if title_normalized else ""


def year_bucket_keys(year: int | None) -> list[int | None]:
    """Blocking key(s) for a record: its own year plus adjacent years, so a
    fuzzy/preprint comparison never needs to be run against the full corpus --
    only against records whose year buckets overlap this one's tolerance window."""
    if year is None:
        return [None]
    return [year - 1, year, year + 1]
