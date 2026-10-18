from __future__ import annotations

from review_pipeline.src.deduplication.canonical import pick_canonical_id


def _row(**kwargs) -> dict:
    defaults = {
        "record_id": "r",
        "doi": None,
        "abstract": None,
        "venue": None,
        "citation_count": None,
        "publication_status": "unknown",
    }
    defaults.update(kwargs)
    return defaults


def test_peer_reviewed_journal_beats_preprint_regardless_of_completeness() -> None:
    preprint = _row(record_id="preprint", publication_status="preprint", doi="10.1/p", abstract="rich abstract")
    journal = _row(record_id="journal", publication_status="peer_reviewed_journal")
    assert pick_canonical_id([preprint, journal]) == "journal"


def test_more_complete_record_wins_within_same_status() -> None:
    sparse = _row(record_id="sparse", publication_status="peer_reviewed_journal")
    rich = _row(
        record_id="rich",
        publication_status="peer_reviewed_journal",
        doi="10.1/x",
        abstract="an abstract",
        venue="A Journal",
        citation_count=3,
    )
    assert pick_canonical_id([sparse, rich]) == "rich"


def test_tie_is_deterministic_by_record_id() -> None:
    a = _row(record_id="b_id", publication_status="unknown")
    b = _row(record_id="a_id", publication_status="unknown")
    assert pick_canonical_id([a, b]) == "a_id"
