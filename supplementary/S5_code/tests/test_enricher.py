from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.enrichment.enricher import enrich_one_record, enrich_records
from review_pipeline.src.search.connectors.base import RawRecord


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
        "doi": "10.1/x",
        "title": title,
        "title_normalized": normalize_title(title),
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    with engine.begin() as conn:
        conn.execute(records_table.insert().values(**values))
    return record_id


def _row(engine, record_id: str) -> dict:
    records_table = reflect_table(engine, "records")
    with engine.connect() as conn:
        return dict(conn.execute(records_table.select().where(records_table.c.record_id == record_id)).mappings().one())


def test_enrich_one_record_skips_no_doi() -> None:
    row = {"record_id": "r1", "doi": None}
    outcome, updates = enrich_one_record(row, fetchers={})
    assert outcome.skipped_reason == "no_doi"
    assert updates == {}


def test_enrich_one_record_fills_empty_fields_from_a_single_source() -> None:
    row = {
        "record_id": "r1",
        "doi": "10.1/x",
        "abstract": None,
        "abstract_source": None,
        "citation_count": None,
        "field_provenance_json": None,
    }
    candidate = RawRecord(
        doi="10.1/x",
        title="A Paper",
        abstract="Fresh abstract.",
        abstract_source="openalex",
        citation_count=9,
        raw_source_id="10.1/x",
    )
    outcome, updates = enrich_one_record(row, fetchers={"openalex": lambda _doi: candidate})
    assert "abstract" in outcome.fields_updated
    assert updates["abstract"] == "Fresh abstract."
    assert updates["citation_count"] == 9
    provenance = json.loads(updates["field_provenance_json"])
    assert provenance["abstract"] == "openalex"


def test_enrich_one_record_does_not_downgrade_a_higher_priority_field() -> None:
    row = {
        "record_id": "r1",
        "doi": "10.1/x",
        "abstract": "Original crossref abstract.",
        "abstract_source": "crossref",
        "citation_count": None,
        "field_provenance_json": json.dumps({"abstract": "crossref"}),
    }
    weaker_candidate = RawRecord(
        doi="10.1/x",
        title="A Paper",
        abstract="Weaker semantic_scholar abstract.",
        abstract_source="semantic_scholar",
        raw_source_id="10.1/x",
    )
    outcome, updates = enrich_one_record(row, fetchers={"semantic_scholar": lambda _doi: weaker_candidate})
    assert "abstract" not in outcome.fields_updated
    assert "abstract" not in updates


def test_enrich_one_record_a_failing_source_does_not_abort_others() -> None:
    row = {"record_id": "r1", "doi": "10.1/x", "citation_count": None, "field_provenance_json": None}

    def _boom(_doi: str):
        raise RuntimeError("network down")

    candidate = RawRecord(doi="10.1/x", title="A Paper", citation_count=4, raw_source_id="10.1/x")
    outcome, updates = enrich_one_record(
        row, fetchers={"broken_source": _boom, "openalex": lambda _doi: candidate}
    )
    assert "broken_source" in outcome.sources_failed
    assert updates["citation_count"] == 4


def test_enrich_records_updates_db_and_summary(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    with_doi = _insert_record(engine, doi="10.1/has-doi", citation_count=None)
    without_doi = _insert_record(engine, doi=None)

    candidate = RawRecord(doi="10.1/has-doi", title="A Paper", citation_count=42, raw_source_id="10.1/has-doi")
    run_logger = RunLogger(component="enrichment", run_id="test-run", logs_dir=tmp_path / "logs")
    summary = enrich_records(engine, fetchers={"openalex": lambda _doi: candidate}, run_logger=run_logger)

    assert summary.records_considered == 2
    assert summary.records_skipped_no_doi == 1
    assert summary.records_updated == 1

    updated_row = _row(engine, with_doi)
    assert updated_row["citation_count"] == 42
    assert json.loads(updated_row["field_provenance_json"])["citation_count"] == "openalex"

    untouched_row = _row(engine, without_doi)
    assert untouched_row["citation_count"] is None

    assert run_logger.path.exists()
