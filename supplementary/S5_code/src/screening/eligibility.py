"""Phase 9: mechanical D0 -> D1 eligibility filter + core/contextual tagging.

D0 is already 1 for every discovered record (set at insert in
search/executor.py). D1 marks which of those are even candidates for
AI/human screening: a canonical (non-merged-away) record whose
publication_status is peer-reviewed, inside the temporal scope, OR covered
by an explicit, human-approved documented_exceptions row.

D2 (screening-confirmed subset) and D3 (verified evidence set) are NOT
computed here -- they depend on screening_decisions (Phase 10/11) and
verified_evidence (Phase 14) data that doesn't exist yet. Setting them now
would mean guessing, which this pipeline's own rules forbid.

core_or_contextual is a mechanical, retrieval-provenance heuristic (was this
record ever surfaced by a CE1-14 family query, on itself or on any record
later merged into it?), not a topical judgment -- it is provisional and
meant to be overridden by human/AI screening, not treated as a final label.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import Engine

from review_pipeline.src.common.config import load_yaml
from review_pipeline.src.common.ids import new_id
from review_pipeline.src.db.tables import reflect_table


def add_documented_exception(engine: Engine, record_id: str, rule_text: str, justification: str, approved_by: str) -> str:
    """Registers a human-approved D1 exception for one record (e.g. a
    foundational pre-2023 work, or a non-peer-reviewed technical report the
    reviewer has decided to include for a stated reason). Never called by any
    automated step -- exceptions are only ever entered by a human via the CLI.
    Returns the new exception_id; recompute_eligibility() must be re-run
    afterward to apply it."""
    documented_exceptions_table = reflect_table(engine, "documented_exceptions")
    exception_id = new_id()
    with engine.begin() as conn:
        conn.execute(
            documented_exceptions_table.insert().values(
                exception_id=exception_id,
                record_id=record_id,
                rule_text=rule_text,
                justification=justification,
                approved_by=approved_by,
                created_at=datetime.now(UTC).isoformat(),
            )
        )
    return exception_id


@dataclass
class EligibilityRules:
    allowed_publication_statuses: set[str]
    primary_from_year: int


def load_eligibility_rules() -> EligibilityRules:
    raw = load_yaml("inclusion_rules.yaml")
    d1 = raw.get("d1_eligibility", {})
    return EligibilityRules(
        allowed_publication_statuses=set(d1.get("allowed_publication_statuses", [])),
        primary_from_year=d1.get("temporal_scope", {}).get("primary_from_year", 2023),
    )


def compute_d1(row: dict, rules: EligibilityRules, has_documented_exception: bool) -> tuple[bool, str | None]:
    """Returns (d1, exclusion_reason). exclusion_reason is None iff d1 is True."""
    if row.get("canonical_record_id") is not None:
        return False, "duplicate_or_merged"
    if has_documented_exception:
        return True, None
    if row.get("publication_status") not in rules.allowed_publication_statuses:
        return False, "not_peer_reviewed"
    year = row.get("year")
    if year is None or year < rules.primary_from_year:
        return False, "before_temporal_scope"
    return True, None


def compute_core_or_contextual(query_ids: list[str], ce_family_keys: set[str]) -> str:
    for query_id in query_ids:
        family_key = query_id.split(".", 1)[0] if query_id else ""
        if family_key in ce_family_keys:
            return "core"
    return "contextual"


@dataclass
class EligibilitySummary:
    records_considered: int = 0
    d1_eligible: int = 0
    exclusion_reason_counts: dict[str, int] = field(default_factory=dict)
    core_count: int = 0
    contextual_count: int = 0


def recompute_eligibility(engine: Engine, ce_family_keys: set[str] | None = None) -> EligibilitySummary:
    if ce_family_keys is None:
        from review_pipeline.src.search.query_builder import load_search_queries_config

        ce_family_keys = set(load_search_queries_config().families.keys())

    rules = load_eligibility_rules()
    records_table = reflect_table(engine, "records")
    documented_exceptions_table = reflect_table(engine, "documented_exceptions")

    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(records_table.select()).mappings().all()]
        exception_rows = conn.execute(documented_exceptions_table.select()).mappings().all()

    exceptions_by_record: dict[str, str] = {e["record_id"]: e["exception_id"] for e in exception_rows}

    # Every record's own query_id, grouped by canonical record (a merged-away
    # record contributes its discovery provenance to its canonical row's tag).
    query_ids_by_canonical: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        target = row["canonical_record_id"] or row["record_id"]
        if row.get("query_id"):
            query_ids_by_canonical[target].append(row["query_id"])

    summary = EligibilitySummary(records_considered=len(rows))
    now = datetime.now(UTC).isoformat()

    with engine.begin() as conn:
        for row in rows:
            record_id = row["record_id"]
            exception_id = exceptions_by_record.get(record_id)
            d1, reason = compute_d1(row, rules, has_documented_exception=exception_id is not None)

            is_canonical = row["canonical_record_id"] is None
            core_or_contextual = (
                compute_core_or_contextual(query_ids_by_canonical.get(record_id, []), ce_family_keys)
                if is_canonical
                else None
            )

            conn.execute(
                records_table.update()
                .where(records_table.c.record_id == record_id)
                .values(
                    d1=1 if d1 else 0,
                    d1_exception_rule_id=exception_id,
                    core_or_contextual=core_or_contextual,
                    updated_at=now,
                )
            )

            if d1:
                summary.d1_eligible += 1
                if core_or_contextual == "core":
                    summary.core_count += 1
                elif core_or_contextual == "contextual":
                    summary.contextual_count += 1
            elif reason:
                summary.exclusion_reason_counts[reason] = summary.exclusion_reason_counts.get(reason, 0) + 1

    return summary
