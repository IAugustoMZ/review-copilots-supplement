from __future__ import annotations

import httpx
import respx

from review_pipeline.src.search.connectors.base import QueryFilters
from review_pipeline.src.search.connectors.crossref import BASE_URL, MAX_ROWS, CrossrefConnector

SAMPLE_ITEM = {
    "DOI": "10.1016/j.dche.2026.100123",
    "title": ["LLM Agent for Chemical Process Simulation"],
    "container-title": ["Digital Chemical Engineering"],
    "type": "journal-article",
    "published": {"date-parts": [[2026, 3]]},
    "author": [{"given": "Jane", "family": "Doe"}, {"given": "John", "family": "Smith"}],
    "abstract": "<jats:p>We present an LLM agent for process simulation.</jats:p>",
    "publisher": "Elsevier",
    "ISSN": ["2772-5081"],
    "is-referenced-by-count": 3,
    "subject": ["Chemical Engineering"],
}


@respx.mock
def test_crossref_search_ok() -> None:
    respx.get(BASE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "message-version": "1.0.0",
                "message": {"items": [SAMPLE_ITEM]},
            },
        )
    )
    connector = CrossrefConnector(contact_email="test@example.com")
    result = connector.search("LLM chemical process", QueryFilters(rows=10))

    assert result.status == "ok"
    assert len(result.records) == 1
    record = result.records[0]
    assert record.doi == "10.1016/j.dche.2026.100123"
    assert record.title == "LLM Agent for Chemical Process Simulation"
    assert record.year == 2026
    assert record.venue == "Digital Chemical Engineering"
    assert record.publication_type == "peer_reviewed_journal"
    assert record.abstract == "We present an LLM agent for process simulation."
    assert record.abstract_source == "crossref"
    assert [a.surname for a in record.authors] == ["Doe", "Smith"]
    assert record.citation_count == 3


@respx.mock
def test_crossref_venue_and_title_html_entities_are_unescaped() -> None:
    item = dict(SAMPLE_ITEM)
    item["container-title"] = ["Industrial &amp; Engineering Chemistry Research"]
    item["title"] = ["Copilots &amp; Process Design"]
    respx.get(BASE_URL).mock(
        return_value=httpx.Response(200, json={"message-version": "1.0.0", "message": {"items": [item]}})
    )
    connector = CrossrefConnector(contact_email="test@example.com")
    result = connector.search("query", QueryFilters())

    record = result.records[0]
    assert record.venue == "Industrial & Engineering Chemistry Research"
    assert record.title == "Copilots & Process Design"


@respx.mock
def test_crossref_excludes_peer_review_and_grant_artifacts() -> None:
    peer_review_item = dict(SAMPLE_ITEM)
    peer_review_item["type"] = "peer-review"
    peer_review_item["title"] = ['Review for "Some Real Paper"']
    peer_review_item["DOI"] = "10.1002/x/v1/review1"
    grant_item = dict(SAMPLE_ITEM)
    grant_item["type"] = "grant"
    grant_item["DOI"] = "10.1002/y/grant1"

    respx.get(BASE_URL).mock(
        return_value=httpx.Response(
            200,
            json={"message-version": "1.0.0", "message": {"items": [SAMPLE_ITEM, peer_review_item, grant_item]}},
        )
    )
    connector = CrossrefConnector(contact_email="test@example.com")
    result = connector.search("query", QueryFilters())

    assert result.status == "ok"
    assert len(result.records) == 1
    assert result.records[0].doi == SAMPLE_ITEM["DOI"]


@respx.mock
def test_crossref_zero_results() -> None:
    respx.get(BASE_URL).mock(
        return_value=httpx.Response(200, json={"message-version": "1.0.0", "message": {"items": []}})
    )
    connector = CrossrefConnector(contact_email="test@example.com")
    result = connector.search("a query that matches nothing", QueryFilters())

    assert result.status == "zero_results"
    assert result.records == []


@respx.mock
def test_crossref_error_is_distinguishable_from_zero_results(monkeypatch) -> None:
    monkeypatch.setattr("review_pipeline.src.search.connectors.http_client.time.sleep", lambda _: None)
    respx.get(BASE_URL).mock(return_value=httpx.Response(503, text="service unavailable"))
    connector = CrossrefConnector(contact_email="test@example.com")
    result = connector.search("query", QueryFilters())

    assert result.status == "error"
    assert result.error_detail is not None
    assert result.records == []


@respx.mock
def test_crossref_clamps_rows_to_api_max() -> None:
    route = respx.get(BASE_URL).mock(
        return_value=httpx.Response(200, json={"message-version": "1.0.0", "message": {"items": []}})
    )
    connector = CrossrefConnector(contact_email="test@example.com")
    connector.search("query", QueryFilters(rows=5000))
    assert route.calls.last.request.url.params["rows"] == str(MAX_ROWS)
