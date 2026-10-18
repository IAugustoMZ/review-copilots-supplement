from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.search.connectors.base import (
    ConnectorResponse,
    QueryFilters,
    RawRecord,
    SearchConnector,
)
from review_pipeline.src.search.venue_search import (
    VenueFilterConnector,
    VenueTarget,
    load_venue_search_config,
    run_venue_search,
)

GENAI = ["large language model", "LLM"]


class _StubConnector(SearchConnector):
    def __init__(self, name: str, records: list[RawRecord]) -> None:
        self.name = name
        self._records = records

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        return ConnectorResponse(status="ok", records=self._records, page_count=1)


def test_load_venue_search_config_has_expected_venues() -> None:
    config = load_venue_search_config()
    venue_names = {v.venue for v in config.venues}
    assert "Digital Chemical Engineering" in venue_names
    assert "AIChE Journal" in venue_names
    assert config.genai_only_query_terms


def test_venue_filter_connector_requires_venue_and_genai_term_match() -> None:
    on_venue = RawRecord(
        title="An LLM Copilot for Plant Operators",
        abstract="A large language model system.",
        venue="Digital Chemical Engineering",
        raw_source_id="1",
    )
    off_venue = RawRecord(
        title="An LLM Copilot for Retail",
        abstract="A large language model system.",
        venue="Journal of Retail Studies",
        raw_source_id="2",
    )
    no_genai_term = RawRecord(
        title="A Statistical Method for Plant Operators",
        abstract="No AI here.",
        venue="Digital Chemical Engineering",
        raw_source_id="3",
    )
    connector = VenueFilterConnector(
        inner=_StubConnector("stub", [on_venue, off_venue, no_genai_term]),
        genai_terms=GENAI,
        container_title="Digital Chemical Engineering",
    )
    result = connector.search("q", QueryFilters())
    assert result.status == "ok"
    assert [r.raw_source_id for r in result.records] == ["1"]


def test_venue_filter_connector_zero_results_when_nothing_matches() -> None:
    off_venue = RawRecord(title="x", venue="Somewhere Else", raw_source_id="1")
    connector = VenueFilterConnector(
        inner=_StubConnector("stub", [off_venue]), genai_terms=GENAI, container_title="Digital Chemical Engineering"
    )
    result = connector.search("q", QueryFilters())
    assert result.status == "zero_results"


def test_run_venue_search_persists_matching_records(tmp_path: Path) -> None:
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)

    on_venue = RawRecord(
        title="An LLM Copilot for Plant Operators",
        abstract="A large language model system.",
        venue="AIChE Journal",
        raw_source_id="1",
    )
    connectors = [_StubConnector("stub", [on_venue])]
    venue = VenueTarget(venue="AIChE Journal", container_title="AIChE Journal")

    results = run_venue_search(engine, venue, connectors=connectors, genai_terms=GENAI)
    assert len(results) == 1
    _run_id, name, response = results[0]
    assert name == "stub"
    assert response.status == "ok"
    assert len(response.records) == 1

    with engine.connect() as conn:
        query_id = conn.execute(text("SELECT query_id FROM search_runs")).scalar_one()
    assert query_id == "venue.aiche_journal.stub"
