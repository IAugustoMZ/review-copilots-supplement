from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.search.connectors.base import (
    ConnectorResponse,
    QueryFilters,
    RawRecord,
    SearchConnector,
)
from review_pipeline.src.search.gap_search import (
    GapClaim,
    load_gap_claims,
    resolve_gap_search,
    run_gap_search,
)


def test_load_gap_claims_returns_real_manuscript_claims() -> None:
    claims = load_gap_claims()
    assert len(claims) > 0
    for c in claims:
        assert c.gap_id
        assert c.claim_text.strip()
        assert c.source_location
        assert len(c.queries) > 0


class _FakeConnector(SearchConnector):
    def __init__(self, name: str, records: list[RawRecord]) -> None:
        self.name = name
        self._records = records

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        return ConnectorResponse(status="ok" if self._records else "zero_results", records=self._records)


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _claim(**kwargs) -> GapClaim:
    defaults = {
        "gap_id": "gap_test",
        "claim_text": "No study addresses X.",
        "source_location": "test.tex:1",
        "queries": ["query one", "query two"],
    }
    defaults.update(kwargs)
    return GapClaim(**defaults)


def test_run_gap_search_no_candidates_found(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    logger = RunLogger(component="gap_search", run_id="t1", logs_dir=tmp_path / "logs")
    connectors = [_FakeConnector("crossref", []), _FakeConnector("openalex", [])]

    run = run_gap_search(engine, _claim(), connectors, logger)
    assert run.candidates == []

    gap_searches = reflect_table(engine, "gap_searches")
    with engine.connect() as conn:
        row = dict(conn.execute(gap_searches.select()).mappings().one())
    assert row["status"] == "UNCERTAIN"
    assert row["candidate_count"] == 0
    assert json.loads(row["queries_json"]) == ["query one", "query two"]
    assert "crossref" in json.loads(row["sources_json"])


def test_run_gap_search_deduplicates_by_doi_across_connectors_and_queries(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    logger = RunLogger(component="gap_search", run_id="t2", logs_dir=tmp_path / "logs")
    same_paper = RawRecord(doi="10.1/x", title="A Counterexample Paper", raw_source_id="x")
    connectors = [_FakeConnector("crossref", [same_paper]), _FakeConnector("openalex", [same_paper])]

    run = run_gap_search(engine, _claim(), connectors, logger)
    # 2 queries x 2 connectors = 4 raw hits, all the same DOI -> deduped to 1
    assert len(run.candidates) == 1


def test_run_gap_search_dedupes_by_title_when_doi_missing(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    logger = RunLogger(component="gap_search", run_id="t3", logs_dir=tmp_path / "logs")
    no_doi_paper = RawRecord(doi=None, title="A Counterexample Paper", raw_source_id="x")
    connectors = [_FakeConnector("crossref", [no_doi_paper])]

    run = run_gap_search(engine, _claim(queries=["only one query"]), connectors, logger)
    assert len(run.candidates) == 1


def test_run_gap_search_distinct_papers_are_not_merged(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    logger = RunLogger(component="gap_search", run_id="t4", logs_dir=tmp_path / "logs")
    paper_a = RawRecord(doi="10.1/a", title="Paper A", raw_source_id="a")
    paper_b = RawRecord(doi="10.1/b", title="Paper B", raw_source_id="b")
    connectors = [_FakeConnector("crossref", [paper_a, paper_b])]

    run = run_gap_search(engine, _claim(queries=["one query"]), connectors, logger)
    assert len(run.candidates) == 2


def test_run_gap_search_does_not_write_to_records_table(tmp_path: Path) -> None:
    """Gap-search candidates must never silently enter the corpus (D0)."""
    engine = _db(tmp_path)
    logger = RunLogger(component="gap_search", run_id="t5", logs_dir=tmp_path / "logs")
    connectors = [_FakeConnector("crossref", [RawRecord(doi="10.1/x", title="X", raw_source_id="x")])]
    run_gap_search(engine, _claim(), connectors, logger)

    records_table = reflect_table(engine, "records")
    with engine.connect() as conn:
        count = len(conn.execute(records_table.select()).mappings().all())
    assert count == 0


def test_resolve_gap_search_sets_final_status(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    logger = RunLogger(component="gap_search", run_id="t6", logs_dir=tmp_path / "logs")
    connectors = [_FakeConnector("crossref", [])]
    run = run_gap_search(engine, _claim(), connectors, logger)

    resolve_gap_search(
        engine, run.gap_search_id, "SUPPORTED_WITHIN_UPDATED_SEARCH", records_reviewed=0, counterexamples=[]
    )

    gap_searches = reflect_table(engine, "gap_searches")
    with engine.connect() as conn:
        row = dict(conn.execute(gap_searches.select()).mappings().one())
    assert row["status"] == "SUPPORTED_WITHIN_UPDATED_SEARCH"


def test_resolve_gap_search_counterexample_found_stores_evidence(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    logger = RunLogger(component="gap_search", run_id="t7", logs_dir=tmp_path / "logs")
    connectors = [_FakeConnector("crossref", [RawRecord(doi="10.1/x", title="A real counterexample", raw_source_id="x")])]
    run = run_gap_search(engine, _claim(queries=["one query"]), connectors, logger)

    resolve_gap_search(
        engine,
        run.gap_search_id,
        "COUNTEREXAMPLE_FOUND",
        records_reviewed=1,
        counterexamples=[{"doi": "10.1/x", "title": "A real counterexample"}],
    )

    gap_searches = reflect_table(engine, "gap_searches")
    with engine.connect() as conn:
        row = dict(conn.execute(gap_searches.select()).mappings().one())
    assert row["status"] == "COUNTEREXAMPLE_FOUND"
    assert row["records_reviewed"] == 1
    assert json.loads(row["counterexamples_json"])[0]["doi"] == "10.1/x"


def test_resolve_gap_search_rejects_invalid_status(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    logger = RunLogger(component="gap_search", run_id="t8", logs_dir=tmp_path / "logs")
    run = run_gap_search(engine, _claim(), [_FakeConnector("crossref", [])], logger)
    with pytest.raises(ValueError, match="invalid status"):
        resolve_gap_search(engine, run.gap_search_id, "NO_RESEARCH_EXISTS", records_reviewed=0)
