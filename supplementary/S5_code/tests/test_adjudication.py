from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.validation.adjudication import (
    apply_adjudication,
    auto_finalize_agreements,
    queue_for_adjudication,
)


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record(engine, record_id: str) -> None:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    title = f"Paper {record_id}"
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


def _insert_classification(
    engine, table_name: str, record_id: str, dimension: str, label: str, confidence: str, coder: str, rationale: str = "x"
) -> None:
    table = reflect_table(engine, table_name)
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            table.insert().values(
                classification_id=new_id(),
                record_id=record_id,
                dimension=dimension,
                label=label,
                confidence=confidence,
                rationale=rationale,
                coder=coder,
                timestamp=now,
            )
        )


def _setup_pair(engine, record_id, ai_label, ai_conf, human_label, human_conf):
    _insert_record(engine, record_id)
    _insert_classification(engine, "classifications_ai", record_id, "topical_inclusion", ai_label, ai_conf, "model")
    _insert_classification(engine, "classifications_human", record_id, "topical_inclusion", human_label, human_conf, "alice")


# --- queue --------------------------------------------------------------


def test_queue_includes_disagreements(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "exclude", "high")
    queue = queue_for_adjudication(engine)
    assert len(queue) == 1
    assert queue[0].record_id == "r1"


def test_queue_includes_low_confidence_even_if_agreeing(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "low", "include", "high")
    queue = queue_for_adjudication(engine)
    assert len(queue) == 1


def test_queue_excludes_confident_agreement(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "include", "high")
    queue = queue_for_adjudication(engine)
    assert len(queue) == 0


def test_queue_excludes_already_resolved_records(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "exclude", "high")
    assert len(queue_for_adjudication(engine)) == 1

    apply_adjudication(engine, "r1", "topical_inclusion", "accept_human", "bob", "resolved")

    # raw AI/human rows are untouched (by design, for the audit trail), so a
    # naive "AI != human" check alone would still flag this -- must not.
    assert len(queue_for_adjudication(engine)) == 0


# --- apply_adjudication ---------------------------------------------------


def test_apply_adjudication_raises_if_no_paired_decisions(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, "r1")
    with pytest.raises(ValueError, match="no paired"):
        apply_adjudication(engine, "r1", "topical_inclusion", "accept_human", "bob", "test")


def test_apply_adjudication_accept_human(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "exclude", "high")
    apply_adjudication(engine, "r1", "topical_inclusion", "accept_human", "bob", "human was right")

    classifications_final = reflect_table(engine, "classifications_final")
    with engine.connect() as conn:
        row = dict(conn.execute(classifications_final.select()).mappings().one())
    assert row["label"] == "exclude"
    assert row["source_of_truth"] == "adjudicated"
    assert row["adjudicator"] == "bob"


def test_apply_adjudication_accept_ai(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "exclude", "high")
    apply_adjudication(engine, "r1", "topical_inclusion", "accept_ai", "bob", "AI was right")

    classifications_final = reflect_table(engine, "classifications_final")
    with engine.connect() as conn:
        row = dict(conn.execute(classifications_final.select()).mappings().one())
    assert row["label"] == "include"


def test_apply_adjudication_new_value_requires_new_decision(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "exclude", "high")
    with pytest.raises(ValueError, match="new_decision"):
        apply_adjudication(engine, "r1", "topical_inclusion", "new_value", "bob", "neither was right")


def test_apply_adjudication_new_value(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "exclude", "high")
    apply_adjudication(
        engine, "r1", "topical_inclusion", "new_value", "bob", "both wrong",
        new_decision="uncertain", new_confidence="medium",
    )
    classifications_final = reflect_table(engine, "classifications_final")
    with engine.connect() as conn:
        row = dict(conn.execute(classifications_final.select()).mappings().one())
    assert row["label"] == "uncertain"
    assert row["confidence"] == "medium"


def test_apply_adjudication_unresolved_sets_uncertain(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "exclude", "high")
    apply_adjudication(engine, "r1", "topical_inclusion", "unresolved", "bob", "genuinely ambiguous")
    classifications_final = reflect_table(engine, "classifications_final")
    with engine.connect() as conn:
        row = dict(conn.execute(classifications_final.select()).mappings().one())
    assert row["label"] == "uncertain"


def test_apply_adjudication_writes_screening_decision_final(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "exclude", "high")
    apply_adjudication(engine, "r1", "topical_inclusion", "accept_human", "bob", "x")

    screening_decisions = reflect_table(engine, "screening_decisions")
    with engine.connect() as conn:
        row = dict(conn.execute(screening_decisions.select()).mappings().one())
    assert row["decision_type"] == "final"
    assert row["decision"] == "exclude"


def test_apply_adjudication_writes_audit_event_with_before_and_after(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "exclude", "high")
    apply_adjudication(engine, "r1", "topical_inclusion", "accept_human", "bob", "x")

    audit_events = reflect_table(engine, "audit_events")
    with engine.connect() as conn:
        row = dict(conn.execute(audit_events.select()).mappings().one())
    before = json.loads(row["before_json"])
    after = json.loads(row["after_json"])
    assert before["ai"]["label"] == "include"
    assert before["human"]["label"] == "exclude"
    assert after["label"] == "exclude"
    assert row["actor"] == "bob"


def test_apply_adjudication_supersedes_previous_final(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "exclude", "high")
    first_id = apply_adjudication(engine, "r1", "topical_inclusion", "accept_human", "bob", "first pass")
    second_id = apply_adjudication(engine, "r1", "topical_inclusion", "accept_ai", "carol", "reversed on review")

    classifications_final = reflect_table(engine, "classifications_final")
    with engine.connect() as conn:
        rows = {r["classification_id"]: dict(r) for r in conn.execute(classifications_final.select()).mappings().all()}
    assert rows[second_id]["supersedes_classification_id"] == first_id
    assert rows[first_id]["supersedes_classification_id"] is None


# --- auto_finalize_agreements ---------------------------------------------


def test_auto_finalize_finalizes_confident_agreements(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "include", "high")
    summary = auto_finalize_agreements(engine)
    assert summary.finalized == 1

    classifications_final = reflect_table(engine, "classifications_final")
    with engine.connect() as conn:
        row = dict(conn.execute(classifications_final.select()).mappings().one())
    assert row["source_of_truth"] == "human"
    assert row["label"] == "include"


def test_auto_finalize_skips_disagreements(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "exclude", "high")
    summary = auto_finalize_agreements(engine)
    assert summary.finalized == 0


def test_auto_finalize_skips_low_confidence_agreements(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "low", "include", "high")
    summary = auto_finalize_agreements(engine)
    assert summary.finalized == 0


def test_auto_finalize_is_idempotent(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _setup_pair(engine, "r1", "include", "high", "include", "high")
    first = auto_finalize_agreements(engine)
    second = auto_finalize_agreements(engine)
    assert first.finalized == 1
    assert second.finalized == 0
    assert second.skipped_already_final == 1

    classifications_final = reflect_table(engine, "classifications_final")
    with engine.connect() as conn:
        rows = conn.execute(classifications_final.select()).mappings().all()
    assert len(rows) == 1  # not double-finalized
