"""Phase 11: the blinded human-review queue.

docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 15: the human-classify code
path must not just "not display" AI labels -- it must have no access to call
them at all. This module enforces that structurally: it never reflects or
queries the DB table that stores AI-generated screening labels, full stop.
tests/test_blinded_queue.py greps this file's own source for that table's
name and fails if a reference is ever added, so this isn't just a comment --
it's a checked invariant. (That table name is deliberately not spelled out
in this docstring, so the grep only matches real code references.)
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Engine, select

from review_pipeline.src.db.tables import reflect_table


def next_for_human_review(
    engine: Engine, reviewer_id: str, allowed_record_ids: set[str] | None = None
) -> dict | None:
    """The next record this reviewer should see: their in-progress record if
    they have one (session resume -- returned regardless of
    allowed_record_ids, since a record already started must be finishable
    even if the sample changed since), else the next D1-eligible, canonical
    record with no human decision yet from anyone. When allowed_record_ids
    is given, candidates are restricted to it (e.g. the validation sample
    from src/validation/sampling.py) -- this is a plain ID set, not a query
    against AI labels, so it doesn't weaken the blinding guarantee below.
    Returns None if the queue is empty. Only fields a reviewer should see are
    returned -- no AI label, no other reviewer's decision."""
    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    records_table = reflect_table(engine, "records")
    classifications_human = reflect_table(engine, "classifications_human")

    with engine.connect() as conn:
        session_row = conn.execute(
            select(reviewer_sessions.c.current_record_id).where(reviewer_sessions.c.reviewer_id == reviewer_id)
        ).first()
        if session_row and session_row[0]:
            record = conn.execute(
                records_table.select().where(records_table.c.record_id == session_row[0])
            ).mappings().first()
            if record:
                return _to_review_view(dict(record))

        already_reviewed = {
            row[0] for row in conn.execute(select(classifications_human.c.record_id))
        }
        candidates = conn.execute(
            records_table.select()
            .where(records_table.c.d1 == 1, records_table.c.canonical_record_id.is_(None))
            .order_by(records_table.c.record_id)
        ).mappings().all()

    for record in candidates:
        if record["record_id"] in already_reviewed:
            continue
        if allowed_record_ids is not None and record["record_id"] not in allowed_record_ids:
            continue
        return _to_review_view(dict(record))
    return None


def _to_review_view(record: dict) -> dict:
    return {
        "record_id": record["record_id"],
        "title": record["title"],
        "abstract": record["abstract"],
        "venue": record["venue"],
        "year": record["year"],
        "publication_status": record["publication_status"],
        "publisher": record.get("publisher"),
        "oa_url": record.get("oa_url"),
        "oa_status": record.get("oa_status"),
        "keywords": record.get("keywords"),
    }


def set_current_record(engine: Engine, reviewer_id: str, record_id: str | None) -> None:
    """Marks the record a reviewer is actively working on, for resume-on-crash/
    quit. Pass None to clear it (after saving a decision or on session end)."""
    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        existing = conn.execute(
            select(reviewer_sessions.c.reviewer_id).where(reviewer_sessions.c.reviewer_id == reviewer_id)
        ).first()
        if existing:
            conn.execute(
                reviewer_sessions.update()
                .where(reviewer_sessions.c.reviewer_id == reviewer_id)
                .values(current_record_id=record_id, last_session_at=now)
            )
        else:
            conn.execute(
                reviewer_sessions.insert().values(
                    reviewer_id=reviewer_id,
                    records_completed=0,
                    current_record_id=record_id,
                    last_session_at=now,
                )
            )
