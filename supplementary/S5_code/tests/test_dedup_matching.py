from __future__ import annotations

from review_pipeline.src.deduplication.disjoint_set import DisjointSet
from review_pipeline.src.deduplication.matching import (
    DedupRecord,
    authors_overlap,
    doi_exact_match,
    preprint_published_confidence,
    title_exact_match,
    title_fuzzy_match,
    years_close,
)


def _rec(**kwargs) -> DedupRecord:
    defaults = {
        "record_id": "r",
        "doi": None,
        "title": "A Title",
        "title_normalized": "a title",
        "year": 2025,
        "publication_status": "unknown",
        "author_surnames": (),
    }
    defaults.update(kwargs)
    return DedupRecord(**defaults)


def test_doi_exact_match_is_case_and_prefix_insensitive() -> None:
    a = _rec(doi="https://doi.org/10.1/X")
    b = _rec(doi="10.1/x")
    assert doi_exact_match(a, b) is True


def test_doi_exact_match_false_when_either_missing() -> None:
    a = _rec(doi=None)
    b = _rec(doi="10.1/x")
    assert doi_exact_match(a, b) is False


def test_title_exact_match_uses_normalized_field() -> None:
    a = _rec(title_normalized="llm copilots for process design")
    b = _rec(title_normalized="llm copilots for process design")
    assert title_exact_match(a, b) is True


def test_title_fuzzy_match_requires_corroboration() -> None:
    # near-identical titles but different year and no shared authors -- should NOT merge
    a = _rec(title_normalized="llm agent for chemical process simulation", year=2020, author_surnames=("Doe",))
    b = _rec(title_normalized="llm agents for chemical process simulations", year=2026, author_surnames=("Smith",))
    matched, score = title_fuzzy_match(a, b)
    assert score >= 0.90
    assert matched is False


def test_title_fuzzy_match_with_year_corroboration_merges() -> None:
    a = _rec(title_normalized="llm agent for chemical process simulation", year=2025)
    b = _rec(title_normalized="llm agents for chemical process simulations", year=2026)
    matched, _score = title_fuzzy_match(a, b)
    assert matched is True


def test_years_close_and_authors_overlap() -> None:
    assert years_close(2025, 2026) is True
    assert years_close(2025, 2027) is False
    assert years_close(None, 2025) is False
    assert authors_overlap(("Doe", "Smith"), ("smith", "Jones")) is True
    assert authors_overlap(("Doe",), ("Jones",)) is False
    assert authors_overlap((), ("Jones",)) is False


def test_preprint_published_confidence_high_requires_all_three_signals() -> None:
    preprint = _rec(
        record_id="p1",
        title_normalized="llm agent for chemical process simulation",
        year=2025,
        publication_status="preprint",
        author_surnames=("Doe", "Smith"),
    )
    journal = _rec(
        record_id="j1",
        title_normalized="llm agent for chemical process simulation",
        year=2026,
        publication_status="peer_reviewed_journal",
        author_surnames=("Doe", "Jones"),
    )
    confidence, evidence = preprint_published_confidence(preprint, journal)
    assert confidence == "high"
    assert evidence["author_overlap"] is True
    assert evidence["year_proximity"] is True


def test_preprint_published_confidence_low_title_similarity_alone() -> None:
    preprint = _rec(title_normalized="a somewhat related preprint title", year=2020, publication_status="preprint")
    journal = _rec(title_normalized="a rather different journal article", year=2026, publication_status="peer_reviewed_journal")
    confidence, _evidence = preprint_published_confidence(preprint, journal)
    assert confidence is None


def test_disjoint_set_transitive_union() -> None:
    ds = DisjointSet(["a", "b", "c", "d"])
    ds.union("a", "b")
    ds.union("b", "c")
    groups = ds.groups()
    assert sorted(groups[ds.find("a")]) == ["a", "b", "c"]
    assert groups[ds.find("d")] == ["d"]
