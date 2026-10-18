from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from sqlalchemy import text

from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.search.citation_trace import get_d1_eligible_seeds, run_citation_trace
from review_pipeline.src.search.connectors.base import ConnectorResponse, QueryFilters, RawRecord
from review_pipeline.src.search.connectors.openalex import BASE_URL as OPENALEX_URL
from review_pipeline.src.search.connectors.openalex import (
    fetch_citing_works,
    fetch_referenced_works,
)
from review_pipeline.src.search.connectors.semantic_scholar import fetch_citations, fetch_references
from review_pipeline.src.search.executor import run_search

SEED_WORK = {
    "id": "https://openalex.org/W111",
    "doi": "https://doi.org/10.1016/j.dche.2026.100001",
    "title": "Seed paper",
    "publication_year": 2026,
    "type": "article",
    "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
}
REFERENCED_WORK = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.1016/j.dche.2020.100002",
    "title": "A foundational reference",
    "publication_year": 2020,
    "type": "article",
}
CITING_WORK = {
    "id": "https://openalex.org/W3",
    "doi": "https://doi.org/10.1016/j.dche.2026.100003",
    "title": "A later citing work",
    "publication_year": 2026,
    "type": "article",
}


class _StubConnector:
    name = "stub"

    def __init__(self, records: list[RawRecord]) -> None:
        self._records = records

    def search(self, query, filters) -> ConnectorResponse:
        return ConnectorResponse(status="ok", records=self._records)


@respx.mock
def test_fetch_referenced_works_batches_by_id() -> None:
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": [REFERENCED_WORK]}))
    result = fetch_referenced_works(httpx.Client(), "test@example.com", SEED_WORK["referenced_works"])
    assert result.status == "ok"
    assert result.records[0].title == "A foundational reference"


def test_fetch_referenced_works_empty_list_is_zero_results() -> None:
    result = fetch_referenced_works(httpx.Client(), "test@example.com", [])
    assert result.status == "zero_results"
    assert result.records == []


@respx.mock
def test_fetch_citing_works_uses_cites_filter() -> None:
    route = respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"meta": {}, "results": [CITING_WORK]}))
    result = fetch_citing_works(httpx.Client(), "test@example.com", "https://openalex.org/W111")
    assert result.status == "ok"
    assert result.records[0].title == "A later citing work"
    assert route.calls.last.request.url.params["filter"] == "cites:W111"


@respx.mock
def test_fetch_references_unwraps_cited_paper() -> None:
    url = "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.dche.2026.100001/references"
    respx.get(url).mock(
        return_value=httpx.Response(
            200, json={"data": [{"citedPaper": {"paperId": "p1", "title": "A cited paper", "externalIds": {"DOI": "10.1/x"}}}]}
        )
    )
    result = fetch_references(httpx.Client(), "10.1016/j.dche.2026.100001")
    assert result.status == "ok"
    assert result.records[0].title == "A cited paper"


@respx.mock
def test_fetch_citations_unwraps_citing_paper() -> None:
    url = "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.dche.2026.100001/citations"
    respx.get(url).mock(
        return_value=httpx.Response(
            200, json={"data": [{"citingPaper": {"paperId": "p2", "title": "A citing paper", "externalIds": {"DOI": "10.2/y"}}}]}
        )
    )
    result = fetch_citations(httpx.Client(), "10.1016/j.dche.2026.100001")
    assert result.status == "ok"
    assert result.records[0].title == "A citing paper"


@respx.mock
def test_fetch_references_treats_null_data_as_zero_results() -> None:
    # Confirmed live: a paper with no references gets {"data": null} back,
    # not {"data": []} -- `for item in data` would crash on the former.
    url = "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.dche.2026.100001/references"
    respx.get(url).mock(return_value=httpx.Response(200, json={"data": None}))
    result = fetch_references(httpx.Client(), "10.1016/j.dche.2026.100001")
    assert result.status == "zero_results"
    assert result.records == []


@respx.mock
def test_fetch_citations_treats_null_data_as_zero_results() -> None:
    url = "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.dche.2026.100001/citations"
    respx.get(url).mock(return_value=httpx.Response(200, json={"data": None}))
    result = fetch_citations(httpx.Client(), "10.1016/j.dche.2026.100001")
    assert result.status == "zero_results"
    assert result.records == []


def test_get_d1_eligible_seeds_excludes_non_d1(tmp_path: Path) -> None:
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)

    eligible = RawRecord(doi="10.1/a", title="Eligible seed", raw_source_id="a")
    not_eligible = RawRecord(doi="10.1/b", title="Not yet eligible", raw_source_id="b")
    run_search(
        engine,
        _StubConnector([eligible, not_eligible]),
        query_id="CE1.stub",
        query_text="q",
        filters=QueryFilters(),
    )

    records_table = reflect_table(engine, "records")
    with engine.begin() as conn:
        conn.execute(records_table.update().where(records_table.c.title == "Eligible seed").values(d1=1))

    seeds = get_d1_eligible_seeds(engine)
    assert len(seeds) == 1
    assert seeds[0]["title"] == "Eligible seed"


def test_seed_filter_keeps_only_codiscovery_hubs(tmp_path: Path) -> None:
    """A work two search runs both returned is a hub; one returned by a single
    run is a leaf, and a degree-2 round must leave it out.

    Ingestion writes one row per search run and Phase 8 merges them afterwards,
    so a hub only *looks* like a hub once its duplicate points at it.
    """
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)

    hub = RawRecord(doi="10.1/hub", title="Hub paper", raw_source_id="hub")
    leaf = RawRecord(doi="10.1/leaf", title="Leaf paper", raw_source_id="leaf")
    run_search(engine, _StubConnector([hub, leaf]), query_id="CE1.stub", query_text="q", filters=QueryFilters())
    run_search(engine, _StubConnector([hub]), query_id="CE2.stub", query_text="q", filters=QueryFilters())

    records_table = reflect_table(engine, "records")
    with engine.begin() as conn:
        conn.execute(records_table.update().values(d1=1))
        hub_ids = [
            r["record_id"]
            for r in conn.execute(records_table.select().where(records_table.c.title == "Hub paper")).mappings()
        ]
        conn.execute(
            records_table.update()
            .where(records_table.c.record_id == hub_ids[1])
            .values(canonical_record_id=hub_ids[0])
        )

    assert {s["title"] for s in get_d1_eligible_seeds(engine)} == {"Hub paper", "Leaf paper"}
    assert [s["title"] for s in get_d1_eligible_seeds(engine, min_codiscovery_degree=2)] == ["Hub paper"]


def test_seed_filter_counts_hits_landing_on_merged_duplicates(tmp_path: Path) -> None:
    """Phase 8 merges duplicates away; their query_hits must still credit the
    surviving record, or dedup would silently demote hubs to leaves."""
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)

    run_search(
        engine,
        _StubConnector([RawRecord(doi="10.1/a", title="Survivor", raw_source_id="a")]),
        query_id="CE1.stub",
        query_text="q",
        filters=QueryFilters(),
    )
    run_search(
        engine,
        _StubConnector([RawRecord(doi="10.1/b", title="Duplicate of survivor", raw_source_id="b")]),
        query_id="CE2.stub",
        query_text="q",
        filters=QueryFilters(),
    )

    records_table = reflect_table(engine, "records")
    with engine.begin() as conn:
        survivor_id = conn.execute(
            records_table.select().where(records_table.c.title == "Survivor")
        ).mappings().one()["record_id"]
        conn.execute(records_table.update().values(d1=1))
        conn.execute(
            records_table.update()
            .where(records_table.c.title == "Duplicate of survivor")
            .values(canonical_record_id=survivor_id)
        )

    assert [s["title"] for s in get_d1_eligible_seeds(engine, min_codiscovery_degree=2)] == ["Survivor"]


def test_skip_already_traced_drops_seeds_from_an_earlier_round(tmp_path: Path) -> None:
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)

    done = RawRecord(doi="10.1/done", title="Traced last round", raw_source_id="done")
    fresh = RawRecord(doi="10.1/fresh", title="Not traced yet", raw_source_id="fresh")
    run_search(engine, _StubConnector([done, fresh]), query_id="CE1.stub", query_text="q", filters=QueryFilters())

    records_table = reflect_table(engine, "records")
    with engine.begin() as conn:
        conn.execute(records_table.update().values(d1=1))
        done_id = conn.execute(
            records_table.select().where(records_table.c.title == "Traced last round")
        ).mappings().one()["record_id"]

    run_search(
        engine,
        _StubConnector([]),
        query_id=f"SNOWBALL.backward.openalex.{done_id}",
        query_text="prior round",
        filters=QueryFilters(),
    )

    assert [s["title"] for s in get_d1_eligible_seeds(engine, skip_already_traced=True)] == ["Not traced yet"]


def test_run_citation_trace_persists_snowball_tagged_records(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_PIPELINE_CONTACT_EMAIL", "test@example.com")
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)

    seed = RawRecord(doi="10.1016/j.dche.2026.100001", title="Seed paper", raw_source_id="seed")
    run_search(engine, _StubConnector([seed]), query_id="CE1.stub", query_text="q", filters=QueryFilters())
    records_table = reflect_table(engine, "records")
    with engine.begin() as conn:
        conn.execute(records_table.update().values(d1=1))

    with respx.mock:
        respx.get(f"{OPENALEX_URL}/doi:10.1016/j.dche.2026.100001").mock(return_value=httpx.Response(200, json=SEED_WORK))
        respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"meta": {}, "results": [REFERENCED_WORK]}))
        respx.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.dche.2026.100001/references"
        ).mock(return_value=httpx.Response(200, json={"data": []}))

        summary = run_citation_trace(engine, direction="backward", iteration=7)

    assert summary.seeds_considered == 1
    assert summary.seeds_skipped_no_doi == 0

    with engine.connect() as conn:
        snowball_rows = conn.execute(
            text("SELECT title, corpus_type, discovery_iteration FROM records WHERE corpus_type = 'snowball'")
        ).all()
    assert ("A foundational reference", "snowball", 7) in snowball_rows


def test_run_citation_trace_skips_seeds_without_doi(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_PIPELINE_CONTACT_EMAIL", "test@example.com")
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)

    seed = RawRecord(doi=None, title="No-DOI seed", raw_source_id="no-doi")
    run_search(engine, _StubConnector([seed]), query_id="CE1.stub", query_text="q", filters=QueryFilters())
    records_table = reflect_table(engine, "records")
    with engine.begin() as conn:
        conn.execute(records_table.update().values(d1=1))

    summary = run_citation_trace(engine, direction="both", iteration=1)
    assert summary.seeds_considered == 1
    assert summary.seeds_skipped_no_doi == 1
    assert summary.outcomes == []


def test_run_citation_trace_rejects_unknown_direction(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_PIPELINE_CONTACT_EMAIL", "test@example.com")
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)

    with pytest.raises(ValueError, match="direction"):
        run_citation_trace(engine, direction="sideways")
