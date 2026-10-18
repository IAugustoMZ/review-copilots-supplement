"""Phase 19 orchestration: forward/backward citation-graph snowballing from
the D1-eligible corpus (docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 6b).

Seeds are every canonical (non-merged-away), D1-eligible record with a DOI.
For each seed, OpenAlex and Semantic Scholar are queried for the works it
cites (backward) and/or the works that cite it (forward). Every response --
including zero_results and error -- is persisted through the exact same
run_search() ingestion path CE1-14 search uses, tagged corpus_type='snowball'
so results are never confused with the primary corpus (CLAUDE.md: "Snowballed
papers are tagged corpus_type: snowball and go through the same three-stage
screening as primary papers"). Downstream dedup and D0->D1 eligibility
recompute treat snowballed records exactly like any other D0 record; no
special-casing lives outside this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from sqlalchemy import Engine, text

from review_pipeline.src.common.config import load_yaml
from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.search.connectors import openalex as openalex_connector
from review_pipeline.src.search.connectors import semantic_scholar as s2_connector
from review_pipeline.src.search.connectors.base import (
    ConnectorResponse,
    QueryFilters,
    SearchConnector,
)
from review_pipeline.src.search.connectors.http_client import build_client
from review_pipeline.src.search.executor import run_search

DIRECTIONS = ("backward", "forward")


class _StaticConnector(SearchConnector):
    """Adapts an already-fetched ConnectorResponse to the SearchConnector
    protocol so run_search()'s existing persistence logic (records,
    query_hits, authors, search_runs) can be reused unmodified -- citation-
    graph results come from a paper-ID lookup, not a query-string search."""

    def __init__(self, name: str, response: ConnectorResponse) -> None:
        self.name = name
        self._response = response

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        return self._response


@dataclass
class SeedOutcome:
    record_id: str
    title: str
    connector: str
    direction: str
    status: str
    candidate_count: int
    error_detail: str | None = None


@dataclass
class CitationTraceSummary:
    seeds_considered: int = 0
    seeds_skipped_no_doi: int = 0
    outcomes: list[SeedOutcome] = field(default_factory=list)

    @property
    def total_candidates(self) -> int:
        return sum(o.candidate_count for o in self.outcomes)


# A record's co-discovery degree is the number of distinct search runs that
# returned it, counted through the canonical record so that hits landing on
# duplicates merged away in Phase 8 still credit the surviving record. A degree
# of k means k independent retrieval paths (CE query strings, and the reference
# or citation lists of k distinct seeds) converged on the same work, which is
# the citation-graph sense of "core": a hub the field keeps pointing at, rather
# than a leaf reached once.
_CODISCOVERY_DEGREE_SQL = """
WITH canonical AS (
    SELECT record_id, COALESCE(canonical_record_id, record_id) AS surviving_id FROM records
)
SELECT c.surviving_id AS record_id, COUNT(DISTINCT h.search_run_id) AS degree
FROM query_hits h JOIN canonical c ON c.record_id = h.record_id
GROUP BY c.surviving_id
"""

def _codiscovery_degrees(engine: Engine) -> dict[str, int]:
    with engine.connect() as conn:
        return {row.record_id: row.degree for row in conn.execute(text(_CODISCOVERY_DEGREE_SQL))}


def already_traced_seed_ids(engine: Engine) -> set[str]:
    """Seeds a previous round already traced. Re-tracing them is a no-op: the
    citation graph is deterministic, so a repeat round spends hours of API
    budget to re-ingest records already in the DB."""
    with engine.connect() as conn:
        query_ids = [
            row[0] for row in conn.execute(text("SELECT DISTINCT query_id FROM search_runs WHERE query_id LIKE 'SNOWBALL.%'"))
        ]
    return {qid.rsplit(".", 1)[-1] for qid in query_ids if "." in qid}


def get_d1_eligible_seeds(
    engine: Engine,
    min_codiscovery_degree: int = 1,
    skip_already_traced: bool = False,
) -> list[dict]:
    """The canonical, D1-eligible records snowballing traces from.

    `min_codiscovery_degree` restricts the round to citation-graph hubs (see
    _CODISCOVERY_DEGREE_SQL); `skip_already_traced` drops seeds a previous
    round covered. Both default off, so an unfiltered call still traces the
    whole D1 corpus.
    """
    records_table = reflect_table(engine, "records")
    with engine.connect() as conn:
        rows = (
            conn.execute(
                records_table.select().where(
                    records_table.c.d1 == 1,
                    records_table.c.canonical_record_id.is_(None),
                )
            )
            .mappings()
            .all()
        )
    seeds = [dict(r) for r in rows]

    if min_codiscovery_degree > 1:
        degrees = _codiscovery_degrees(engine)
        seeds = [s for s in seeds if degrees.get(s["record_id"], 0) >= min_codiscovery_degree]
    if skip_already_traced:
        traced = already_traced_seed_ids(engine)
        seeds = [s for s in seeds if s["record_id"] not in traced]
    return seeds


def _resolve_contact() -> tuple[str, str | None]:
    connectors_cfg = load_yaml("search_queries.yaml").get("connectors", {})
    email_env = connectors_cfg.get("contact_email_env", "REVIEW_PIPELINE_CONTACT_EMAIL")
    contact_email = os.environ.get(email_env)
    if not contact_email:
        raise RuntimeError(f"No contact email configured. Set the {email_env} environment variable.")
    s2_key_env = connectors_cfg.get("semantic_scholar_api_key_env", "SEMANTIC_SCHOLAR_API_KEY")
    return contact_email, os.environ.get(s2_key_env)


def _ingest(
    engine: Engine,
    run_logger: RunLogger,
    connector_name: str,
    direction: str,
    seed: dict,
    response: ConnectorResponse,
    iteration: int,
) -> SeedOutcome:
    query_id = f"SNOWBALL.{direction}.{connector_name}.{seed['record_id']}"
    query_text = f"{direction} citations of {seed['record_id']} ({seed['title'][:80]!r}) via {connector_name}"
    run_search(
        engine,
        _StaticConnector(connector_name, response),
        query_id=query_id,
        query_text=query_text,
        filters=QueryFilters(),
        corpus_type="snowball",
        run_logger=run_logger,
        iteration=iteration,
    )
    return SeedOutcome(
        record_id=seed["record_id"],
        title=seed["title"],
        connector=connector_name,
        direction=direction,
        status=response.status,
        candidate_count=len(response.records),
        error_detail=response.error_detail,
    )


def run_citation_trace(
    engine: Engine,
    direction: str = "both",
    iteration: int = 1,
    run_logger: RunLogger | None = None,
    min_codiscovery_degree: int = 1,
    skip_already_traced: bool = False,
) -> CitationTraceSummary:
    if direction not in ("backward", "forward", "both"):
        raise ValueError(f"direction must be backward|forward|both, got {direction!r}")
    directions = DIRECTIONS if direction == "both" else (direction,)

    contact_email, s2_api_key = _resolve_contact()
    run_logger = run_logger or RunLogger(component="citation_trace", run_id=new_id())

    openalex_client = build_client(contact_email)
    s2_headers = {"x-api-key": s2_api_key} if s2_api_key else None
    s2_client = build_client(contact_email, extra_headers=s2_headers)

    seeds = get_d1_eligible_seeds(
        engine,
        min_codiscovery_degree=min_codiscovery_degree,
        skip_already_traced=skip_already_traced,
    )
    summary = CitationTraceSummary(seeds_considered=len(seeds))

    for seed in seeds:
        doi = seed.get("doi")
        if not doi:
            summary.seeds_skipped_no_doi += 1
            continue

        # The seed's own OpenAlex work object supplies both its
        # referenced_works list (backward) and its native id (needed by the
        # cites: filter, forward) -- resolved once per seed, reused for both
        # directions rather than re-fetched per direction.
        openalex_work = openalex_connector.fetch_by_doi(openalex_client, doi, contact_email)

        for d in directions:
            if openalex_work is not None:
                work_meta = openalex_work.raw_metadata
                if d == "backward":
                    oa_response = openalex_connector.fetch_referenced_works(
                        openalex_client, contact_email, work_meta.get("referenced_works", [])
                    )
                else:
                    oa_response = openalex_connector.fetch_citing_works(openalex_client, contact_email, work_meta.get("id", ""))
            else:
                oa_response = ConnectorResponse(status="zero_results", error_detail="seed DOI not found in OpenAlex")
            summary.outcomes.append(_ingest(engine, run_logger, "openalex", d, seed, oa_response, iteration))

            s2_fetch = s2_connector.fetch_references if d == "backward" else s2_connector.fetch_citations
            s2_response = s2_fetch(s2_client, doi)
            summary.outcomes.append(_ingest(engine, run_logger, "semantic_scholar", d, seed, s2_response, iteration))

    return summary
