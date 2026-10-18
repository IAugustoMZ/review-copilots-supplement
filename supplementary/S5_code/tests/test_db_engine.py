from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table


def test_migrations_apply_and_are_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    engine = get_engine(db_path)

    first_run = run_migrations(engine)
    assert first_run == [
        "001_initial.sql",
        "002_enrichment.sql",
        "003_fulltext_surrogate.sql",
        "004_iteration_tracking.sql",
        "005_dimension_classification.sql",
        "006_ml_assist.sql",
        "007_ml_assist_dimensions.sql",
    ]

    second_run = run_migrations(engine)
    assert second_run == []


def test_migrated_schema_has_expected_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    engine = get_engine(db_path)
    run_migrations(engine)

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
    table_names = {row[0] for row in rows}

    expected = {
        "records",
        "search_runs",
        "query_hits",
        "authors",
        "record_authors",
        "fulltext_locations",
        "preprint_matches",
        "duplicate_groups",
        "classifications_ai",
        "classifications_human",
        "classifications_final",
        "screening_decisions",
        "verified_evidence",
        "sentinel_records",
        "gap_searches",
        "audit_events",
        "reviewer_sessions",
        "corpus_snapshots",
        "documented_exceptions",
        "schema_migrations",
        "search_iterations",
        "ml_assist_suggestions",
    }
    assert expected.issubset(table_names)


def test_foreign_keys_enforced(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    engine = get_engine(db_path)
    run_migrations(engine)

    records = reflect_table(engine, "records")
    with engine.connect() as conn:
        # canonical_record_id references records.record_id; pointing at a
        # nonexistent record must fail with FK enforcement on.
        try:
            conn.execute(
                records.insert().values(
                    record_id="r1",
                    title="t",
                    title_normalized="t",
                    canonical_record_id="does-not-exist",
                    created_at="2026-01-01T00:00:00Z",
                    updated_at="2026-01-01T00:00:00Z",
                )
            )
            conn.commit()
            raised = False
        except Exception:  # noqa: BLE001 -- asserting a DB constraint fired, any error qualifies
            raised = True
    assert raised, "expected a foreign key violation"


def test_publication_status_check_constraint(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    engine = get_engine(db_path)
    run_migrations(engine)

    records = reflect_table(engine, "records")
    with engine.connect() as conn:
        try:
            conn.execute(
                records.insert().values(
                    record_id="r1",
                    title="t",
                    title_normalized="t",
                    publication_status="not_a_real_status",
                    created_at="2026-01-01T00:00:00Z",
                    updated_at="2026-01-01T00:00:00Z",
                )
            )
            conn.commit()
            raised = False
        except Exception:  # noqa: BLE001 -- asserting a DB constraint fired, any error qualifies
            raised = True
    assert raised, "expected a CHECK constraint violation"
