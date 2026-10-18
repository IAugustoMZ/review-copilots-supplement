from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.search.ce_search import build_default_connectors, run_ce_family
from review_pipeline.src.search.connectors.base import (
    ConnectorResponse,
    QueryFilters,
    RawRecord,
    SearchConnector,
)
from review_pipeline.src.search.query_builder import load_search_queries_config


class _StubConnector(SearchConnector):
    """Returns a mix of on-topic and off-topic records so the family filter
    (client-side Boolean-AND-of-ORs) has something real to prove out."""

    def __init__(self, name: str) -> None:
        self.name = name

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        return ConnectorResponse(
            status="ok",
            records=[
                RawRecord(
                    title="An LLM Agent for HAZOP Review in Chemical Plants",
                    abstract="We use a large language model to assist HAZOP studies in a chemical process plant.",
                    raw_source_id="on-topic-1",
                ),
                RawRecord(
                    title="A Large Language Model for Customer Support Chatbots",
                    abstract="An LLM-based chatbot for retail customer service.",
                    raw_source_id="off-topic-1",
                ),
            ],
            page_count=1,
        )


def test_run_ce_family_persists_only_family_matching_records(tmp_path: Path) -> None:
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    config = load_search_queries_config()

    connectors = [_StubConnector("stub_a"), _StubConnector("stub_b")]
    results = run_ce_family(engine, "CE6", connectors=connectors, config=config)

    assert len(results) == 2
    for _search_run_id, _name, response in results:
        assert response.status == "ok"
        assert len(response.records) == 1
        assert "HAZOP" in response.records[0].title

    with engine.connect() as conn:
        titles = [row[0] for row in conn.execute(text("SELECT title FROM records"))]
    assert titles == ["An LLM Agent for HAZOP Review in Chemical Plants"] * 2

    with engine.connect() as conn:
        query_ids = {row[0] for row in conn.execute(text("SELECT DISTINCT query_id FROM search_runs"))}
    assert query_ids == {"CE6.stub_a", "CE6.stub_b"}


class _RecordingConnector(SearchConnector):
    """Captures the filters it was called with, so a test can inspect what
    each connector actually received (e.g. whether OpenAlex got routed
    through the boolean_filter path)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.received_filters: QueryFilters | None = None

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        self.received_filters = filters
        return ConnectorResponse(status="zero_results", records=[])


def test_run_ce_family_routes_only_openalex_through_boolean_filter(tmp_path: Path) -> None:
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    config = load_search_queries_config()

    crossref_stub = _RecordingConnector("crossref")
    openalex_stub = _RecordingConnector("openalex")
    run_ce_family(engine, "CE6", connectors=[crossref_stub, openalex_stub], config=config)

    assert crossref_stub.received_filters.extra == {}
    assert "boolean_filter" in openalex_stub.received_filters.extra
    boolean_filter = openalex_stub.received_filters.extra["boolean_filter"]
    assert boolean_filter.count("title_and_abstract.search:") == 2  # genai group AND domain group
    assert openalex_stub.received_filters.extra["relevance_hint"]


def test_build_default_connectors_requires_contact_email(monkeypatch) -> None:
    monkeypatch.delenv("REVIEW_PIPELINE_CONTACT_EMAIL", raising=False)
    with pytest.raises(RuntimeError, match="contact email"):
        build_default_connectors()


def test_build_default_connectors_succeeds_with_email(monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_PIPELINE_CONTACT_EMAIL", "test@example.com")
    connectors = build_default_connectors()
    assert {c.name for c in connectors} == {"crossref", "openalex", "semantic_scholar"}
