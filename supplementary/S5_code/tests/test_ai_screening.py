from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.screening.ai_screening import (
    ScreeningDecision,
    export_batch,
    ingest_decisions,
    load_decisions,
    write_export,
)

# --- ScreeningDecision schema -----------------------------------------------


def test_exclude_requires_exclusion_reason() -> None:
    with pytest.raises(Exception, match="exclusion_reason"):
        ScreeningDecision(record_id="r1", decision="exclude", confidence="high", rationale="off topic")


def test_uncertain_requires_uncertainty_reason() -> None:
    with pytest.raises(Exception, match="uncertainty_reason"):
        ScreeningDecision(record_id="r1", decision="uncertain", confidence="low", rationale="ambiguous")


def test_include_needs_no_reason_fields() -> None:
    d = ScreeningDecision(record_id="r1", decision="include", confidence="high", rationale="clearly on-topic")
    assert d.exclusion_reason is None


def test_invalid_decision_value_rejected() -> None:
    with pytest.raises(Exception, match="decision"):
        ScreeningDecision(record_id="r1", decision="maybe", confidence="high", rationale="x")


# --- export/ingest against a real DB ----------------------------------------


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


def test_export_batch_only_includes_d1_canonical_records(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    eligible = _insert_record(engine, d1=1)
    not_eligible = _insert_record(engine, d1=0)
    merged_away = _insert_record(engine, d1=1, canonical_record_id=eligible)

    batch = export_batch(engine)
    ids = {r["record_id"] for r in batch}
    assert eligible in ids
    assert not_eligible not in ids
    assert merged_away not in ids


def test_export_batch_respects_limit(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    for _ in range(5):
        _insert_record(engine)
    assert len(export_batch(engine, limit=2)) == 2
    assert len(export_batch(engine)) == 5


def test_export_batch_excludes_already_ai_screened_records(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    screened = _insert_record(engine)
    unscreened = _insert_record(engine)

    decisions = [ScreeningDecision(record_id=screened, decision="include", confidence="high", rationale="on-topic")]
    ingest_decisions(engine, decisions, coder="test-model", prompt_version="v1")

    batch = export_batch(engine)
    ids = {r["record_id"] for r in batch}
    assert screened not in ids
    assert unscreened in ids


def test_write_export_roundtrips_through_json(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, title="A Paper About LLM Copilots")
    batch = export_batch(engine)
    out_path = tmp_path / "export.json"
    write_export(batch, out_path)

    loaded = json.loads(out_path.read_text())
    assert loaded[0]["title"] == "A Paper About LLM Copilots"


def test_ingest_writes_classifications_ai_and_screening_decisions(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    record_id = _insert_record(engine)
    decisions = [
        ScreeningDecision(
            record_id=record_id,
            decision="exclude",
            confidence="high",
            rationale="Purely classical ML, no generative component.",
            exclusion_reason="wrong_technology",
        )
    ]
    summary = ingest_decisions(engine, decisions, coder="claude-sonnet-5", prompt_version="v1")
    assert summary.ingested == 1
    assert summary.decision_counts == {"exclude": 1}

    classifications_ai = reflect_table(engine, "classifications_ai")
    screening_decisions = reflect_table(engine, "screening_decisions")
    with engine.connect() as conn:
        cls_row = dict(conn.execute(classifications_ai.select()).mappings().one())
        decision_row = dict(conn.execute(screening_decisions.select()).mappings().one())

    assert cls_row["label"] == "exclude"
    assert cls_row["dimension"] == "topical_inclusion"
    assert decision_row["decision_type"] == "ai"
    assert decision_row["decision"] == "exclude"
    assert decision_row["exclusion_reason"] == "wrong_technology"


def test_ingest_never_writes_final_screening_decision(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    record_id = _insert_record(engine)
    decisions = [ScreeningDecision(record_id=record_id, decision="include", confidence="high", rationale="on-topic")]
    ingest_decisions(engine, decisions, coder="claude-sonnet-5", prompt_version="v1")

    screening_decisions = reflect_table(engine, "screening_decisions")
    with engine.connect() as conn:
        decision_types = {r[0] for r in conn.execute(select(screening_decisions.c.decision_type))}
    assert decision_types == {"ai"}


def test_ingest_skips_unknown_record_id_without_crashing(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    decisions = [
        ScreeningDecision(record_id="does-not-exist", decision="include", confidence="high", rationale="x")
    ]
    summary = ingest_decisions(engine, decisions, coder="claude-sonnet-5", prompt_version="v1")
    assert summary.ingested == 0
    assert summary.skipped_unknown_record_id == ["does-not-exist"]


def test_load_decisions_reports_malformed_entries_without_dropping_valid_ones(tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps(
            [
                {"record_id": "r1", "decision": "include", "confidence": "high", "rationale": "ok"},
                {"record_id": "r2", "decision": "exclude", "confidence": "high", "rationale": "missing reason"},
            ]
        )
    )
    result = load_decisions(path)
    assert len(result.decisions) == 1
    assert result.decisions[0].record_id == "r1"
    assert len(result.errors) == 1
    assert "r2" in result.errors[0]
