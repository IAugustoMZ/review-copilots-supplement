from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.deduplication.dedup import run_deduplication


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record(engine, *, authors: list[str] | None = None, **overrides) -> str:
    records_table = reflect_table(engine, "records")
    authors_table = reflect_table(engine, "authors")
    record_authors_table = reflect_table(engine, "record_authors")
    now = datetime.now(UTC).isoformat()
    record_id = overrides.pop("record_id", new_id())
    title = overrides.pop("title", "A Paper")
    values = {
        "record_id": record_id,
        "doi": None,
        "title": title,
        "title_normalized": normalize_title(title),
        "publication_status": "unknown",
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    with engine.begin() as conn:
        conn.execute(records_table.insert().values(**values))
        for position, surname in enumerate(authors or [], start=1):
            author_id = new_id()
            conn.execute(authors_table.insert().values(author_id=author_id, full_name=surname, surname=surname))
            conn.execute(
                record_authors_table.insert().values(record_id=record_id, author_id=author_id, position=position)
            )
    return record_id


def _row(engine, record_id: str) -> dict:
    records_table = reflect_table(engine, "records")
    with engine.connect() as conn:
        return dict(conn.execute(records_table.select().where(records_table.c.record_id == record_id)).mappings().one())


def _run(engine, tmp_path: Path):
    logger = RunLogger(component="deduplication", run_id="test-run", logs_dir=tmp_path / "logs")
    return run_deduplication(engine, logger)


def test_doi_exact_duplicates_are_merged(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    a = _insert_record(engine, doi="https://doi.org/10.1/X", publication_status="peer_reviewed_journal")
    b = _insert_record(engine, doi="10.1/x", publication_status="peer_reviewed_journal")

    summary = _run(engine, tmp_path)
    assert summary.duplicate_groups_created == 1
    assert summary.records_merged_away == 1

    row_a, row_b = _row(engine, a), _row(engine, b)
    canonical_ids = {row_a["canonical_record_id"], row_b["canonical_record_id"]}
    assert None in canonical_ids  # one stays canonical (NULL)
    assert (a in canonical_ids) or (b in canonical_ids)  # the other points at it


def test_title_exact_duplicates_across_different_dois_are_merged(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    a = _insert_record(engine, doi="10.1/a", title="LLM Copilot for Process Design")
    b = _insert_record(engine, doi="10.2/b", title="LLM Copilot for Process Design")

    summary = _run(engine, tmp_path)
    assert summary.duplicate_groups_created == 1

    row_a, row_b = _row(engine, a), _row(engine, b)
    assert row_a["canonical_record_id"] is not None or row_b["canonical_record_id"] is not None


def test_unrelated_records_are_not_merged(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    a = _insert_record(engine, doi="10.1/a", title="LLM Copilot for Process Design", year=2025)
    b = _insert_record(engine, doi="10.2/b", title="A Totally Unrelated Study of Reaction Kinetics", year=2019)

    summary = _run(engine, tmp_path)
    assert summary.duplicate_groups_created == 0

    assert _row(engine, a)["canonical_record_id"] is None
    assert _row(engine, b)["canonical_record_id"] is None


def test_high_confidence_preprint_journal_pair_is_merged_and_logged(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    preprint = _insert_record(
        engine,
        doi="10.48550/arxiv.2601.11650",
        title="LLM Agent for Chemical Process Simulation",
        year=2025,
        publication_status="preprint",
        authors=["Doe", "Smith"],
    )
    journal = _insert_record(
        engine,
        doi="10.1016/j.dche.2026.100123",
        title="LLM Agent for Chemical Process Simulation",
        year=2026,
        publication_status="peer_reviewed_journal",
        authors=["Doe", "Jones"],
    )

    summary = _run(engine, tmp_path)
    assert summary.duplicate_groups_created == 1
    assert summary.preprint_matches_logged == 1
    assert summary.preprint_outcomes[0].merged is True

    preprint_row = _row(engine, preprint)
    journal_row = _row(engine, journal)
    assert preprint_row["canonical_record_id"] == journal  # preprint merges INTO the journal record
    assert journal_row["canonical_record_id"] is None

    preprint_matches_table = reflect_table(engine, "preprint_matches")
    with engine.connect() as conn:
        match_row = dict(conn.execute(preprint_matches_table.select()).mappings().one())
    assert match_row["status"] == "auto_linked"
    assert match_row["match_confidence"] == "high"
    assert match_row["journal_record_id"] == journal
    assert match_row["preprint_record_id"] == preprint


def test_medium_confidence_preprint_pair_is_logged_but_not_merged(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    # Title similarity ~0.87 (medium tier) + shared author, but year 3 apart
    # (outside tolerance) -- tests that author overlap alone can corroborate
    # medium confidence without year proximity.
    preprint = _insert_record(
        engine,
        doi="10.48550/arxiv.2601.99999",
        title="LLM Agents for Chemical Process Simulation and Design",
        year=2022,
        publication_status="preprint",
        authors=["Doe"],
    )
    journal = _insert_record(
        engine,
        doi="10.1016/j.other.2026.100999",
        title="LLM Agent for Chemical Process Simulation",
        year=2026,
        publication_status="peer_reviewed_journal",
        authors=["Doe"],
    )

    summary = _run(engine, tmp_path)
    assert summary.preprint_matches_logged == 1
    outcome = summary.preprint_outcomes[0]
    assert outcome.confidence == "medium"
    assert outcome.merged is False
    assert summary.duplicate_groups_created == 0

    assert _row(engine, preprint)["canonical_record_id"] is None
    assert _row(engine, journal)["canonical_record_id"] is None

    preprint_matches_table = reflect_table(engine, "preprint_matches")
    with engine.connect() as conn:
        match_row = dict(conn.execute(preprint_matches_table.select()).mappings().one())
    assert match_row["status"] == "pending_human_review"


def test_earliest_seen_iteration_backfilled_for_merged_group(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    early = _insert_record(
        engine, doi="10.1/x", title="LLM Copilot for Process Design", discovery_iteration=1
    )
    later = _insert_record(
        engine, doi="10.2/y", title="LLM Copilot for Process Design", discovery_iteration=3
    )

    summary = _run(engine, tmp_path)
    assert summary.duplicate_groups_created == 1

    canonical_id = early if _row(engine, early)["canonical_record_id"] is None else later
    assert _row(engine, canonical_id)["earliest_seen_iteration"] == 1


def test_earliest_seen_iteration_backfilled_for_singleton(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    solo = _insert_record(engine, doi="10.1/solo", title="A Unique Paper", discovery_iteration=2)

    _run(engine, tmp_path)
    assert _row(engine, solo)["earliest_seen_iteration"] == 2


def test_canonical_selection_prefers_peer_reviewed_over_preprint(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    preprint = _insert_record(
        engine, doi="10.48550/arxiv.2601.55555", title="Shared Title Exact Match", publication_status="preprint"
    )
    journal = _insert_record(
        engine, doi="10.1016/j.x.2026.1", title="Shared Title Exact Match", publication_status="peer_reviewed_journal"
    )

    _run(engine, tmp_path)
    assert _row(engine, preprint)["canonical_record_id"] == journal
    assert _row(engine, journal)["canonical_record_id"] is None
