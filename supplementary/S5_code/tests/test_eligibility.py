from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.screening.eligibility import (
    EligibilityRules,
    add_documented_exception,
    compute_core_or_contextual,
    compute_d1,
    recompute_eligibility,
)

RULES = EligibilityRules(allowed_publication_statuses={"peer_reviewed_journal", "peer_reviewed_conference"}, primary_from_year=2023)


def test_compute_d1_accepts_peer_reviewed_in_scope() -> None:
    row = {"canonical_record_id": None, "publication_status": "peer_reviewed_journal", "year": 2025}
    d1, reason = compute_d1(row, RULES, has_documented_exception=False)
    assert d1 is True
    assert reason is None


def test_compute_d1_rejects_merged_away_record() -> None:
    row = {"canonical_record_id": "other-id", "publication_status": "peer_reviewed_journal", "year": 2025}
    d1, reason = compute_d1(row, RULES, has_documented_exception=False)
    assert d1 is False
    assert reason == "duplicate_or_merged"


def test_compute_d1_rejects_preprint_by_default() -> None:
    row = {"canonical_record_id": None, "publication_status": "preprint", "year": 2025}
    d1, reason = compute_d1(row, RULES, has_documented_exception=False)
    assert d1 is False
    assert reason == "not_peer_reviewed"


def test_compute_d1_rejects_before_temporal_scope() -> None:
    row = {"canonical_record_id": None, "publication_status": "peer_reviewed_journal", "year": 2021}
    d1, reason = compute_d1(row, RULES, has_documented_exception=False)
    assert d1 is False
    assert reason == "before_temporal_scope"


def test_compute_d1_missing_year_is_excluded_not_assumed_in_scope() -> None:
    row = {"canonical_record_id": None, "publication_status": "peer_reviewed_journal", "year": None}
    d1, reason = compute_d1(row, RULES, has_documented_exception=False)
    assert d1 is False
    assert reason == "before_temporal_scope"


def test_documented_exception_overrides_publication_status_and_year() -> None:
    row = {"canonical_record_id": None, "publication_status": "preprint", "year": 2019}
    d1, reason = compute_d1(row, RULES, has_documented_exception=True)
    assert d1 is True
    assert reason is None


def test_documented_exception_does_not_override_duplicate_status() -> None:
    row = {"canonical_record_id": "other-id", "publication_status": "peer_reviewed_journal", "year": 2025}
    d1, reason = compute_d1(row, RULES, has_documented_exception=True)
    assert d1 is False
    assert reason == "duplicate_or_merged"


def test_core_or_contextual_tags_ce_family_query_id_as_core() -> None:
    assert compute_core_or_contextual(["CE6.crossref"], {"CE1", "CE6"}) == "core"


def test_core_or_contextual_tags_venue_only_as_contextual() -> None:
    assert compute_core_or_contextual(["venue.dce.crossref"], {"CE1", "CE6"}) == "contextual"


def test_core_or_contextual_with_no_query_ids_is_contextual() -> None:
    assert compute_core_or_contextual([], {"CE1"}) == "contextual"


# --- integration: real DB ---------------------------------------------------


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


def test_recompute_eligibility_sets_d1_and_core_tag(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    core_id = _insert_record(engine, query_id="CE6.crossref")
    contextual_id = _insert_record(engine, query_id="venue.dce.crossref")
    excluded_id = _insert_record(engine, publication_status="preprint")

    summary = recompute_eligibility(engine, ce_family_keys={"CE1", "CE6"})
    assert summary.records_considered == 3
    assert summary.d1_eligible == 2
    assert summary.core_count == 1
    assert summary.contextual_count == 1
    assert summary.exclusion_reason_counts["not_peer_reviewed"] == 1

    assert _row(engine, core_id)["d1"] == 1
    assert _row(engine, core_id)["core_or_contextual"] == "core"
    assert _row(engine, contextual_id)["core_or_contextual"] == "contextual"
    assert _row(engine, excluded_id)["d1"] == 0


def test_merged_away_record_inherits_no_core_tag_but_canonical_gets_its_provenance(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    canonical_id = _insert_record(engine, query_id="venue.dce.crossref")
    merged_id = _insert_record(engine, query_id="CE2.openalex", canonical_record_id=canonical_id)

    summary = recompute_eligibility(engine, ce_family_keys={"CE2"})
    assert summary.d1_eligible == 1  # only the canonical counts

    canonical_row = _row(engine, canonical_id)
    merged_row = _row(engine, merged_id)
    assert canonical_row["core_or_contextual"] == "core"  # inherited from the merged CE2 hit
    assert merged_row["core_or_contextual"] is None  # non-canonical rows aren't independently tagged
    assert merged_row["d1"] == 0


def test_add_documented_exception_makes_a_preprint_d1_eligible(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    record_id = _insert_record(engine, publication_status="preprint", year=2019)

    add_documented_exception(
        engine,
        record_id,
        rule_text="foundational pre-2023 work",
        justification="Introduces the LLM-agent-for-process-simulation pattern this review traces.",
        approved_by="test-reviewer",
    )
    summary = recompute_eligibility(engine, ce_family_keys=set())
    assert summary.d1_eligible == 1

    row = _row(engine, record_id)
    assert row["d1"] == 1
    assert row["d1_exception_rule_id"] is not None
