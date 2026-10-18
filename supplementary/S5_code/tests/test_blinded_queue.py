from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.validation import blinded_queue
from review_pipeline.src.validation.blinded_queue import next_for_human_review, set_current_record


def test_blinded_queue_module_never_references_classifications_ai() -> None:
    """Structural enforcement of the blinding guarantee: this module must have
    no access to the AI-labels table at all, not just avoid displaying it."""
    source = inspect.getsource(blinded_queue)
    assert "classifications_ai" not in source


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record(engine, **overrides) -> str:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    record_id = overrides.pop("record_id", new_id())
    title = overrides.pop("title", "A Paper")
    values = {
        "record_id": record_id,
        "title": title,
        "title_normalized": normalize_title(title),
        "publication_status": "peer_reviewed_journal",
        "year": 2025,
        "d1": 1,
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    with engine.begin() as conn:
        conn.execute(records_table.insert().values(**values))
    return record_id


def _insert_human_classification(engine, record_id: str, reviewer: str) -> None:
    classifications_human = reflect_table(engine, "classifications_human")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            classifications_human.insert().values(
                classification_id=new_id(),
                record_id=record_id,
                dimension="topical_inclusion",
                label="include",
                confidence="high",
                rationale="x",
                coder=reviewer,
                timestamp=now,
            )
        )


def test_next_for_human_review_skips_already_reviewed_records(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    reviewed = _insert_record(engine, record_id="r1")
    pending = _insert_record(engine, record_id="r2")
    _insert_human_classification(engine, reviewed, "alice")

    result = next_for_human_review(engine, "alice")
    assert result["record_id"] == pending


def test_next_for_human_review_ignores_non_d1_and_merged_records(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    core_record = _insert_record(engine, record_id="r1")
    _insert_record(engine, record_id="r2", d1=0)
    _insert_record(engine, record_id="r3", canonical_record_id=core_record)

    result = next_for_human_review(engine, "alice")
    assert result["record_id"] == core_record


def test_next_for_human_review_empty_queue_returns_none(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    assert next_for_human_review(engine, "alice") is None


def test_next_for_human_review_resumes_in_progress_record(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, record_id="r1")
    in_progress = _insert_record(engine, record_id="r2")
    set_current_record(engine, "alice", in_progress)

    result = next_for_human_review(engine, "alice")
    assert result["record_id"] == in_progress


def test_review_view_does_not_leak_internal_columns(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, record_id="r1", query_id="CE6.crossref", d1_exception_rule_id=None)
    result = next_for_human_review(engine, "alice")
    assert "query_id" not in result
    assert "canonical_record_id" not in result
    assert "core_or_contextual" not in result


def test_set_current_record_creates_and_updates_session(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, record_id="r1")
    set_current_record(engine, "alice", "r1")

    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    with engine.connect() as conn:
        row = dict(conn.execute(reviewer_sessions.select()).mappings().one())
    assert row["current_record_id"] == "r1"

    set_current_record(engine, "alice", None)
    with engine.connect() as conn:
        row = dict(conn.execute(reviewer_sessions.select()).mappings().one())
    assert row["current_record_id"] is None
