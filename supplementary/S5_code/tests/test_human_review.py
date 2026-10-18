from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.screening.ai_screening import ScreeningDecision
from review_pipeline.src.validation.blinded_queue import set_current_record
from review_pipeline.src.validation.human_review import (
    finalize_record_review,
    save_human_decision,
    skip_current_record,
)


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record(engine, record_id: str = "r1") -> str:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    title = "A Paper"
    with engine.begin() as conn:
        conn.execute(
            records_table.insert().values(
                record_id=record_id,
                title=title,
                title_normalized=normalize_title(title),
                publication_status="peer_reviewed_journal",
                year=2025,
                d1=1,
                created_at=now,
                updated_at=now,
            )
        )
    return record_id


def test_save_human_decision_writes_classification_and_screening_decision(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    record_id = _insert_record(engine)
    decision = ScreeningDecision(record_id=record_id, decision="include", confidence="high", rationale="on-topic")

    save_human_decision(engine, decision, reviewer_id="alice")

    classifications_human = reflect_table(engine, "classifications_human")
    screening_decisions = reflect_table(engine, "screening_decisions")
    with engine.connect() as conn:
        cls_row = dict(conn.execute(classifications_human.select()).mappings().one())
        dec_row = dict(conn.execute(screening_decisions.select()).mappings().one())

    assert cls_row["coder"] == "alice"
    assert cls_row["label"] == "include"
    assert dec_row["decision_type"] == "human"
    assert dec_row["coder"] == "alice"


def test_save_human_decision_does_not_touch_reviewer_sessions(tmp_path: Path) -> None:
    """An 'include' decision still has the dimension-classification step
    ahead of it -- the record isn't done yet, so save_human_decision alone
    must not increment records_completed or clear current_record_id. Only
    finalize_record_review() does that (see below)."""
    engine = _db(tmp_path)
    record_id = _insert_record(engine)
    set_current_record(engine, "alice", record_id)
    decision = ScreeningDecision(record_id=record_id, decision="include", confidence="high", rationale="on-topic")

    save_human_decision(engine, decision, reviewer_id="alice")

    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    with engine.connect() as conn:
        row = dict(conn.execute(reviewer_sessions.select()).mappings().one())
    assert row["records_completed"] == 0
    assert row["current_record_id"] == record_id


def test_finalize_record_review_increments_records_completed_and_clears_current(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    record_id = _insert_record(engine)
    set_current_record(engine, "alice", record_id)

    finalize_record_review(engine, "alice")

    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    with engine.connect() as conn:
        row = dict(conn.execute(reviewer_sessions.select()).mappings().one())
    assert row["records_completed"] == 1
    assert row["current_record_id"] is None


def test_finalize_record_review_twice_increments_across_records(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    r1 = _insert_record(engine, record_id="r1")
    r2 = _insert_record(engine, record_id="r2")
    save_human_decision(
        engine, ScreeningDecision(record_id=r1, decision="include", confidence="high", rationale="x"), "alice"
    )
    finalize_record_review(engine, "alice")
    save_human_decision(
        engine, ScreeningDecision(record_id=r2, decision="exclude", confidence="high", rationale="x", exclusion_reason="off_topic"), "alice"
    )
    finalize_record_review(engine, "alice")

    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    with engine.connect() as conn:
        row = dict(conn.execute(reviewer_sessions.select()).mappings().one())
    assert row["records_completed"] == 2


def test_skip_current_record_clears_without_saving_decision(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    record_id = _insert_record(engine)
    set_current_record(engine, "alice", record_id)

    skip_current_record(engine, "alice")

    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        row = dict(conn.execute(reviewer_sessions.select()).mappings().one())
        n_classifications = len(conn.execute(select(classifications_human)).all())
    assert row["current_record_id"] is None
    assert n_classifications == 0
