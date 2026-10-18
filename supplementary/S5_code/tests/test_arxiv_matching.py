from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.fulltext import arxiv_matching
from review_pipeline.src.fulltext.arxiv_matching import find_arxiv_fallback, score_match
from review_pipeline.src.search.connectors.arxiv import ArxivCandidate


def _candidate(**kwargs) -> ArxivCandidate:
    defaults = {
        "arxiv_id": "2402.10977",
        "title": "Generative AI and Process Systems Engineering: The Next Frontier",
        "author_surnames": ["Decardi-Nelson"],
        "year": 2024,
        "journal_ref": None,
        "doi": None,
        "pdf_url": "http://arxiv.org/pdf/2402.10977",
    }
    defaults.update(kwargs)
    return ArxivCandidate(**defaults)


def test_score_match_high_via_doi_match_regardless_of_title() -> None:
    candidate = _candidate(title="A completely different title", doi="10.1/x")
    confidence, evidence = score_match("Original Title", 2024, "10.1/x", [], candidate)
    assert confidence == "high"
    assert evidence.doi_or_jref_match is True


def test_score_match_high_via_title_and_author_similarity() -> None:
    candidate = _candidate(title="LLM Agent for Chemical Process Simulation", author_surnames=["Doe", "Smith"])
    confidence, _ = score_match("LLM Agent for Chemical Process Simulation", 2024, None, ["Doe", "Jones"], candidate)
    assert confidence == "high"


def test_score_match_medium() -> None:
    # title_sim ~0.92 (in [0.90, 0.97)) and author_overlap = 1/4 = 0.25 (in
    # [0.25, 0.5), so it clears MEDIUM but not HIGH's stricter 0.5 bar).
    candidate = _candidate(
        title="LLM Agent for Chemical Process Simulation Design", author_surnames=["Doe", "X", "Y", "Z"]
    )
    confidence, _ = score_match(
        "LLM Agent for Chemical Process Simulation", 2024, None, ["Doe", "A", "B", "C"], candidate
    )
    assert confidence == "medium"


def test_score_match_low_keyword_level_only() -> None:
    candidate = _candidate(title="A survey of chemical process simulation methods", author_surnames=["Nobody"])
    confidence, _ = score_match("LLM Agent for Chemical Process Simulation Automation", 2024, None, ["Doe"], candidate)
    assert confidence in ("low", None)  # depends on exact fuzzy score; must not be high/medium
    assert confidence != "high"
    assert confidence != "medium"


def test_score_match_none_for_unrelated_titles() -> None:
    candidate = _candidate(title="A totally unrelated paper about cooking recipes", author_surnames=["Nobody"])
    confidence, _ = score_match("LLM Agent for Chemical Process Simulation", 2024, None, ["Doe"], candidate)
    assert confidence is None


# --- find_arxiv_fallback (DB integration) -----------------------------------


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record_with_author(
    engine, record_id: str, title: str, doi: str | None, year: int, surname: str | list[str]
) -> None:
    records_table = reflect_table(engine, "records")
    authors_table = reflect_table(engine, "authors")
    record_authors_table = reflect_table(engine, "record_authors")
    now = datetime.now(UTC).isoformat()
    surnames = [surname] if isinstance(surname, str) else surname
    with engine.begin() as conn:
        conn.execute(
            records_table.insert().values(
                record_id=record_id, title=title, title_normalized=normalize_title(title),
                doi=doi, year=year, publication_status="peer_reviewed_journal",
                created_at=now, updated_at=now,
            )
        )
        for position, s in enumerate(surnames, start=1):
            author_id = new_id()
            conn.execute(authors_table.insert().values(author_id=author_id, full_name=s, surname=s))
            conn.execute(record_authors_table.insert().values(record_id=record_id, author_id=author_id, position=position))


def test_find_arxiv_fallback_high_confidence_auto_links(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    _insert_record_with_author(engine, "r1", "LLM Agent for Chemical Process Simulation", None, 2024, "Doe")

    candidate = _candidate(title="LLM Agent for Chemical Process Simulation", author_surnames=["Doe"])
    monkeypatch.setattr(arxiv_matching, "search_arxiv_by_title", lambda client, title, max_results=5: [candidate])

    result = find_arxiv_fallback(engine, "r1", httpx.Client())
    assert result.confidence == "high"
    assert result.auto_linked is True

    records_table = reflect_table(engine, "records")
    with engine.connect() as conn:
        row = dict(conn.execute(records_table.select().where(records_table.c.record_id == "r1")).mappings().one())
    assert row["fulltext_surrogate_url"] == candidate.pdf_url
    assert row["preprint_doi_or_id"] == candidate.arxiv_id

    candidates_table = reflect_table(engine, "fulltext_surrogate_candidates")
    with engine.connect() as conn:
        cand_row = dict(conn.execute(candidates_table.select()).mappings().one())
    assert cand_row["status"] == "auto_linked"
    assert cand_row["match_confidence"] == "high"


def test_find_arxiv_fallback_medium_confidence_pending_review_does_not_set_surrogate_url(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    _insert_record_with_author(
        engine, "r1", "LLM Agent for Chemical Process Simulation", None, 2024, ["Doe", "A", "B", "C"]
    )

    candidate = _candidate(title="LLM Agent for Chemical Process Simulation", author_surnames=["Doe", "X", "Y", "Z"])
    monkeypatch.setattr(arxiv_matching, "search_arxiv_by_title", lambda client, title, max_results=5: [candidate])

    result = find_arxiv_fallback(engine, "r1", httpx.Client())
    assert result.confidence == "medium"
    assert result.auto_linked is False

    records_table = reflect_table(engine, "records")
    with engine.connect() as conn:
        row = dict(conn.execute(records_table.select().where(records_table.c.record_id == "r1")).mappings().one())
    assert row["fulltext_surrogate_url"] is None

    candidates_table = reflect_table(engine, "fulltext_surrogate_candidates")
    with engine.connect() as conn:
        cand_row = dict(conn.execute(candidates_table.select()).mappings().one())
    assert cand_row["status"] == "pending_human_review"


def test_find_arxiv_fallback_no_candidates_found(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    _insert_record_with_author(engine, "r1", "LLM Agent for Chemical Process Simulation", None, 2024, "Doe")
    monkeypatch.setattr(arxiv_matching, "search_arxiv_by_title", lambda client, title, max_results=5: [])

    result = find_arxiv_fallback(engine, "r1", httpx.Client())
    assert result.confidence is None
    assert result.auto_linked is False

    candidates_table = reflect_table(engine, "fulltext_surrogate_candidates")
    with engine.connect() as conn:
        rows = conn.execute(candidates_table.select()).mappings().all()
    assert len(rows) == 0


def test_find_arxiv_fallback_stores_evidence_json(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    _insert_record_with_author(engine, "r1", "LLM Agent for Chemical Process Simulation", None, 2024, "Doe")
    candidate = _candidate(title="LLM Agent for Chemical Process Simulation", author_surnames=["Doe"])
    monkeypatch.setattr(arxiv_matching, "search_arxiv_by_title", lambda client, title, max_results=5: [candidate])

    find_arxiv_fallback(engine, "r1", httpx.Client())

    candidates_table = reflect_table(engine, "fulltext_surrogate_candidates")
    with engine.connect() as conn:
        cand_row = dict(conn.execute(candidates_table.select()).mappings().one())
    evidence = json.loads(cand_row["evidence_json"])
    assert "title_sim" in evidence
    assert "author_overlap" in evidence
