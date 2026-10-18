from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.search.sentinel import (
    SentinelSeed,
    check_sentinels,
    known_item_retrieval_rate,
    load_sentinels_to_db,
)


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record(engine, *, title: str, doi: str | None, query_id: str, retrieval_source: str) -> None:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    from review_pipeline.src.common.ids import new_id
    from review_pipeline.src.common.text_normalize import normalize_title

    with engine.begin() as conn:
        conn.execute(
            records_table.insert().values(
                record_id=new_id(),
                doi=doi,
                title=title,
                title_normalized=normalize_title(title),
                query_id=query_id,
                retrieval_source=retrieval_source,
                created_at=now,
                updated_at=now,
            )
        )


def test_load_sentinels_to_db_is_idempotent_replace(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(
        "review_pipeline.src.search.sentinel.load_sentinel_config",
        lambda: [
            SentinelSeed(title="Paper A", why_known_relevant="known relevant"),
            SentinelSeed(title="Paper B", why_known_relevant="known relevant"),
        ],
    )
    n1 = load_sentinels_to_db(engine)
    n2 = load_sentinels_to_db(engine)
    assert n1 == 2
    assert n2 == 2

    sentinel_table = reflect_table(engine, "sentinel_records")
    # reload replaces wholesale rather than accumulating duplicates
    with engine.connect() as conn:
        rows = conn.execute(sentinel_table.select()).mappings().all()
    assert len(rows) == 2


def test_check_sentinels_classifies_retrieval_channel(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(
        "review_pipeline.src.search.sentinel.load_sentinel_config",
        lambda: [
            SentinelSeed(title="Found via CE search", doi="10.1/ce", why_known_relevant="x"),
            SentinelSeed(title="Found via venue search", doi="10.1/venue", why_known_relevant="x"),
            SentinelSeed(title="Never found", doi="10.1/missing", why_known_relevant="x"),
        ],
    )
    load_sentinels_to_db(engine)

    _insert_record(engine, title="Found via CE search", doi="10.1/ce", query_id="CE6.crossref", retrieval_source="crossref")
    _insert_record(
        engine,
        title="Found via venue search",
        doi="10.1/venue",
        query_id="venue.dce.crossref",
        retrieval_source="crossref",
    )

    results = check_sentinels(engine)
    by_title = {r.title: r for r in results}

    assert by_title["Found via CE search"].matched is True
    assert by_title["Found via CE search"].retrieved_ce_search is True
    assert by_title["Found via CE search"].retrieved_venue_search is False

    assert by_title["Found via venue search"].matched is True
    assert by_title["Found via venue search"].retrieved_venue_search is True
    assert by_title["Found via venue search"].retrieved_ce_search is False

    assert by_title["Never found"].matched is False

    assert known_item_retrieval_rate(results) == 2 / 3


def test_check_sentinels_matches_by_doi_even_if_title_differs_slightly(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(
        "review_pipeline.src.search.sentinel.load_sentinel_config",
        lambda: [SentinelSeed(title="Original Title Wording", doi="10.1/x", why_known_relevant="x")],
    )
    load_sentinels_to_db(engine)
    _insert_record(
        engine, title="Slightly Different Title Wording", doi="10.1/x", query_id="Q1", retrieval_source="crossref"
    )

    results = check_sentinels(engine)
    assert results[0].matched is True
    assert results[0].retrieved_generic is True


def test_check_sentinels_persists_results_to_db(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(
        "review_pipeline.src.search.sentinel.load_sentinel_config",
        lambda: [SentinelSeed(title="Paper A", doi="10.1/a", why_known_relevant="x")],
    )
    load_sentinels_to_db(engine)
    _insert_record(engine, title="Paper A", doi="10.1/a", query_id="CE1.openalex", retrieval_source="openalex")
    check_sentinels(engine)

    sentinel_table = reflect_table(engine, "sentinel_records")
    with engine.connect() as conn:
        row = conn.execute(sentinel_table.select()).mappings().one()
    assert row["retrieved_ce_search"] == 1
    assert json.loads(row["retrieval_sources_json"]) == ["openalex"]
    assert row["last_checked_at"] is not None
