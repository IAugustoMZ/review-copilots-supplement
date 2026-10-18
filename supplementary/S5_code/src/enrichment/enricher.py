"""Phase 7: metadata enrichment with field-level provenance.

For every record with a DOI, fetch fresh metadata from each configured
enrichment source and merge it field-by-field using source_priority rules --
never silently overwriting higher-quality metadata with lower-quality
metadata (docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 8). Provenance per
field is persisted in records.field_provenance_json so later enrichment runs
can keep applying the same rule.

Records without a DOI are skipped (not silently dropped -- reported in the
summary) since every enrichment source here is a DOI lookup.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import Engine

from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.enrichment.source_priority import MERGEABLE_FIELDS, should_overwrite
from review_pipeline.src.search.connectors.base import RawRecord

# source_name -> fetch_by_doi(doi) -> RawRecord | None
EnrichmentFetcher = Callable[[str], RawRecord | None]


@dataclass
class EnrichmentOutcome:
    record_id: str
    doi: str | None
    fields_updated: list[str] = field(default_factory=list)
    sources_attempted: list[str] = field(default_factory=list)
    sources_failed: list[str] = field(default_factory=list)
    skipped_reason: str | None = None


@dataclass
class EnrichmentSummary:
    records_considered: int = 0
    records_skipped_no_doi: int = 0
    records_updated: int = 0
    records_unchanged: int = 0
    outcomes: list[EnrichmentOutcome] = field(default_factory=list)


def _is_empty(value: object) -> bool:
    return value is None or value in ("", [], "[]")


def _candidate_field_value(candidate: RawRecord, field_name: str) -> object:
    value = getattr(candidate, field_name, None)
    if field_name == "keywords" and isinstance(value, list):
        return json.dumps(value) if value else None
    return value


def enrich_one_record(
    record_row: dict,
    fetchers: dict[str, EnrichmentFetcher],
    run_logger: RunLogger | None = None,
) -> tuple[EnrichmentOutcome, dict]:
    """Returns (outcome, column_updates). column_updates is empty if nothing changed."""
    record_id = record_row["record_id"]
    doi = record_row.get("doi")
    if not doi:
        outcome = EnrichmentOutcome(record_id=record_id, doi=None, skipped_reason="no_doi")
        if run_logger:
            run_logger.log_event(event="enrichment_skipped", record_id=record_id, reason="no_doi")
        return outcome, {}

    provenance: dict[str, str] = json.loads(record_row.get("field_provenance_json") or "{}")
    # abstract_source is a real column already; seed its provenance from itself
    # on first run so a pre-Phase-7 abstract isn't treated as unranked forever.
    if record_row.get("abstract_source") and "abstract" not in provenance:
        provenance["abstract"] = record_row["abstract_source"]

    working_values = {f: record_row.get(f) for f in MERGEABLE_FIELDS}
    updates: dict[str, object] = {}
    outcome = EnrichmentOutcome(record_id=record_id, doi=doi)

    for source_name, fetch in fetchers.items():
        outcome.sources_attempted.append(source_name)
        try:
            candidate = fetch(doi)
        except Exception as exc:  # noqa: BLE001 -- one source's transient failure must not abort the record
            outcome.sources_failed.append(source_name)
            if run_logger:
                run_logger.log_event(
                    event="enrichment_source_failed", record_id=record_id, source=source_name, error=str(exc)
                )
            continue
        if candidate is None:
            continue

        for field_name in MERGEABLE_FIELDS:
            candidate_value = _candidate_field_value(candidate, field_name)
            if _is_empty(candidate_value):
                continue
            current_value = working_values[field_name]
            current_source = provenance.get(field_name)
            if should_overwrite(field_name, current_value, current_source, source_name):
                working_values[field_name] = candidate_value
                updates[field_name] = candidate_value
                provenance[field_name] = source_name

    if updates:
        outcome.fields_updated = sorted(updates)
        updates["field_provenance_json"] = json.dumps(provenance)
        updates["updated_at"] = datetime.now(UTC).isoformat()
        if run_logger:
            run_logger.log_event(
                event="enrichment_applied",
                record_id=record_id,
                doi=doi,
                fields_updated=outcome.fields_updated,
                sources_attempted=outcome.sources_attempted,
                sources_failed=outcome.sources_failed,
            )
    elif run_logger:
        run_logger.log_event(
            event="enrichment_no_change",
            record_id=record_id,
            doi=doi,
            sources_attempted=outcome.sources_attempted,
            sources_failed=outcome.sources_failed,
        )

    return outcome, updates


def enrich_records(
    engine: Engine,
    fetchers: dict[str, EnrichmentFetcher],
    run_logger: RunLogger,
    limit: int | None = None,
) -> EnrichmentSummary:
    """Enrich every canonical (non-merged-away), non-excluded-from-consideration
    record with a DOI. Iterates and commits per-record so a crash partway
    through leaves already-enriched records enriched, not rolled back."""
    records_table = reflect_table(engine, "records")
    summary = EnrichmentSummary()

    with engine.connect() as conn:
        query = records_table.select().where(records_table.c.canonical_record_id.is_(None))
        if limit is not None:
            query = query.limit(limit)
        rows = [dict(r) for r in conn.execute(query).mappings().all()]

    for row in rows:
        summary.records_considered += 1
        outcome, updates = enrich_one_record(row, fetchers, run_logger)
        summary.outcomes.append(outcome)
        if outcome.skipped_reason == "no_doi":
            summary.records_skipped_no_doi += 1
            continue
        if not updates:
            summary.records_unchanged += 1
            continue
        with engine.begin() as conn:
            conn.execute(
                records_table.update().where(records_table.c.record_id == row["record_id"]).values(**updates)
            )
        summary.records_updated += 1

    return summary
