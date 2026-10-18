from __future__ import annotations

from review_pipeline.src.search.connectors.base import (
    ConnectorResponse,
    QueryFilters,
    RawRecord,
    SearchConnector,
)
from review_pipeline.src.search.family_filter import FamilyFilterConnector

GENAI = ["large language model"]
DOMAIN = ["chemical engineering"]
THEME = ["HAZOP"]


class _StubConnector(SearchConnector):
    def __init__(self, response: ConnectorResponse) -> None:
        self.name = "stub"
        self._response = response

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        return self._response


def _wrap(response: ConnectorResponse) -> FamilyFilterConnector:
    return FamilyFilterConnector(
        inner=_StubConnector(response), genai_terms=GENAI, domain_terms=DOMAIN, theme_terms=THEME
    )


def test_error_response_passes_through_unchanged() -> None:
    response = ConnectorResponse(status="error", error_detail="boom")
    result = _wrap(response).search("q", QueryFilters())
    assert result.status == "error"
    assert result.error_detail == "boom"


def test_zero_results_passes_through_unchanged() -> None:
    response = ConnectorResponse(status="zero_results", records=[])
    result = _wrap(response).search("q", QueryFilters())
    assert result.status == "zero_results"


def test_ok_response_is_filtered_to_family_matches() -> None:
    on_topic = RawRecord(
        title="LLM for HAZOP in chemical engineering",
        abstract="A large language model applied to HAZOP.",
        raw_source_id="1",
    )
    off_topic = RawRecord(title="LLM for cooking recipes", abstract="Unrelated.", raw_source_id="2")
    response = ConnectorResponse(status="ok", records=[on_topic, off_topic])
    result = _wrap(response).search("q", QueryFilters())
    assert result.status == "ok"
    assert [r.raw_source_id for r in result.records] == ["1"]


def test_ok_response_becomes_zero_results_if_nothing_matches_family() -> None:
    off_topic = RawRecord(title="LLM for cooking recipes", abstract="Unrelated.", raw_source_id="2")
    response = ConnectorResponse(status="ok", records=[off_topic])
    result = _wrap(response).search("q", QueryFilters())
    assert result.status == "zero_results"
    assert result.records == []


def test_default_gate_admits_genai_domain_match_without_theme() -> None:
    # matches genai + domain but not the HAZOP theme block -- the strict
    # matches_family test would reject this; the default (broad) D0 gate admits it.
    no_theme = RawRecord(
        title="A large language model for chemical engineering education",
        abstract="Helps students.",
        raw_source_id="1",
    )
    response = ConnectorResponse(status="ok", records=[no_theme])
    result = _wrap(response).search("q", QueryFilters())
    assert result.status == "ok"
    assert [r.raw_source_id for r in result.records] == ["1"]


def test_strict_theme_true_rejects_genai_domain_match_without_theme() -> None:
    no_theme = RawRecord(
        title="A large language model for chemical engineering education",
        abstract="Helps students.",
        raw_source_id="1",
    )
    response = ConnectorResponse(status="ok", records=[no_theme])
    strict = FamilyFilterConnector(
        inner=_StubConnector(response), genai_terms=GENAI, domain_terms=DOMAIN, theme_terms=THEME, strict_theme=True
    )
    result = strict.search("q", QueryFilters())
    assert result.status == "zero_results"


def test_connector_name_is_taken_from_inner() -> None:
    wrapped = _wrap(ConnectorResponse(status="ok", records=[]))
    assert wrapped.name == "stub"
