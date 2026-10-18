"""Phase 15: gap-search verification.

For every ZERO/ONLY/NONE/FIRST/ABSENT/NO-STUDY claim the manuscript makes
(config/gap_claims.yaml), run the claim's hand-authored queries across every
connector and collect candidates that MIGHT be counterexamples -- papers a
fresh, targeted search finds that the original corpus search missed.

Deliberately separate from the main corpus pipeline: gap-search candidates
are NOT inserted into `records`/`search_runs`/`query_hits` (that would
silently inflate D0 with records nobody screened through the normal funnel).
Everything here lives in its own `gap_searches` row -- one per verification
run (not per claim; re-running the same claim later creates a new row, so
the manuscript's absence claims can be re-checked before every submission
without losing history).

Final status is never SUPPORTED_WITHIN_UPDATED_SEARCH by default -- a human
(or a subagent, mirroring Phase 10's screening pattern) must review the
candidates and call resolve_gap_search() to set the real status. Until then
a run sits at its initial 'UNCERTAIN' state, which is also the DB column's
own default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel
from sqlalchemy import Engine

from review_pipeline.src.common.config import load_yaml
from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.common.text_normalize import normalize_doi, normalize_title
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.search.connectors.base import QueryFilters, RawRecord, SearchConnector


class GapClaim(BaseModel):
    gap_id: str
    claim_text: str
    source_location: str
    queries: list[str]


def load_gap_claims() -> list[GapClaim]:
    raw = load_yaml("gap_claims.yaml")
    return [GapClaim(**c) for c in raw.get("gap_claims", [])]


@dataclass
class GapCandidate:
    doi: str | None
    title: str
    year: int | None
    venue: str | None
    abstract: str | None
    matched_query: str
    connector: str


@dataclass
class GapSearchRun:
    gap_search_id: str
    gap_id: str
    claim_text: str
    candidates: list[GapCandidate] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)


def _dedupe_candidates(raw_hits: list[tuple[RawRecord, str, str]]) -> list[GapCandidate]:
    """raw_hits: (record, query, connector_name). Dedupes by normalized DOI,
    falling back to normalized title -- same cascade logic as corpus dedup,
    applied to a small, ephemeral candidate list rather than persisted rows."""
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    result = []
    for record, query, connector_name in raw_hits:
        doi_key = normalize_doi(record.doi)
        title_key = normalize_title(record.title)
        if doi_key and doi_key in seen_dois:
            continue
        if not doi_key and title_key in seen_titles:
            continue
        if doi_key:
            seen_dois.add(doi_key)
        seen_titles.add(title_key)
        result.append(
            GapCandidate(
                doi=record.doi,
                title=record.title,
                year=record.year,
                venue=record.venue,
                abstract=record.abstract,
                matched_query=query,
                connector=connector_name,
            )
        )
    return result


def run_gap_search(
    engine: Engine, gap_claim: GapClaim, connectors: list[SearchConnector], run_logger: RunLogger
) -> GapSearchRun:
    gap_search_id = new_id()
    run = GapSearchRun(gap_search_id=gap_search_id, gap_id=gap_claim.gap_id, claim_text=gap_claim.claim_text)
    raw_hits: list[tuple[RawRecord, str, str]] = []

    for query in gap_claim.queries:
        run.queries_used.append(query)
        for connector in connectors:
            if connector.name not in run.sources_used:
                run.sources_used.append(connector.name)
            response = connector.search(query, QueryFilters(rows=20))
            run_logger.log_event(
                event="gap_search_query",
                gap_id=gap_claim.gap_id,
                query=query,
                connector=connector.name,
                status=response.status,
                record_count=len(response.records),
            )
            for record in response.records:
                raw_hits.append((record, query, connector.name))

    run.candidates = _dedupe_candidates(raw_hits)

    gap_searches_table = reflect_table(engine, "gap_searches")
    with engine.begin() as conn:
        conn.execute(
            gap_searches_table.insert().values(
                gap_id=gap_search_id,
                claim=f"[{gap_claim.gap_id}] {gap_claim.claim_text}",
                queries_json=json.dumps(run.queries_used),
                sources_json=json.dumps(run.sources_used),
                date=datetime.now(UTC).isoformat(),
                candidate_count=len(run.candidates),
                records_reviewed=0,
                counterexamples_json=None,
                status="UNCERTAIN",
            )
        )

    return run


def resolve_gap_search(
    engine: Engine,
    gap_search_id: str,
    status: str,
    records_reviewed: int,
    counterexamples: list[dict] | None = None,
) -> None:
    """Sets the final status after a human/subagent has actually reviewed the
    candidates -- never called automatically by run_gap_search()."""
    if status not in ("SUPPORTED_WITHIN_UPDATED_SEARCH", "COUNTEREXAMPLE_FOUND", "UNCERTAIN"):
        raise ValueError(f"invalid status: {status}")

    gap_searches_table = reflect_table(engine, "gap_searches")
    with engine.begin() as conn:
        conn.execute(
            gap_searches_table.update()
            .where(gap_searches_table.c.gap_id == gap_search_id)
            .values(
                status=status,
                records_reviewed=records_reviewed,
                counterexamples_json=json.dumps(counterexamples or []),
            )
        )
