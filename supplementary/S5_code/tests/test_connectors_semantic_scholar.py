from __future__ import annotations

import httpx
import respx

from review_pipeline.src.search.connectors.base import QueryFilters
from review_pipeline.src.search.connectors.semantic_scholar import (
    BASE_URL,
    MAX_LIMIT,
    SemanticScholarConnector,
)

SAMPLE_PAPER = {
    "paperId": "abc123",
    "title": "Multi-Agent LLM Orchestration for Plant Fault Diagnosis",
    "abstract": "We propose a multi-agent LLM system for fault diagnosis.",
    "year": 2026,
    "venue": "AIChE Journal",
    "publicationTypes": ["JournalArticle"],
    "externalIds": {"DOI": "10.1002/aic.18000"},
    "authors": [{"name": "Carlos Mendez"}],
    "openAccessPdf": {"url": "https://example.com/oa.pdf"},
    "citationCount": 5,
}


@respx.mock
def test_semantic_scholar_search_ok() -> None:
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"data": [SAMPLE_PAPER]}))
    connector = SemanticScholarConnector(contact_email="test@example.com")
    result = connector.search("multi-agent fault diagnosis", QueryFilters(rows=10))

    assert result.status == "ok"
    record = result.records[0]
    assert record.doi == "10.1002/aic.18000"
    assert record.publication_type == "peer_reviewed_journal"
    assert record.oa_url == "https://example.com/oa.pdf"
    assert record.authors[0].surname == "Mendez"


@respx.mock
def test_semantic_scholar_zero_results() -> None:
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    connector = SemanticScholarConnector(contact_email="test@example.com")
    result = connector.search("nothing matches this", QueryFilters())
    assert result.status == "zero_results"


@respx.mock
def test_semantic_scholar_clamps_limit_to_api_max() -> None:
    route = respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    connector = SemanticScholarConnector(contact_email="test@example.com")
    connector.search("query", QueryFilters(rows=500))
    assert route.calls.last.request.url.params["limit"] == str(MAX_LIMIT)


@respx.mock
def test_semantic_scholar_falls_back_when_paper_id_is_explicitly_null() -> None:
    # A common real shape for an unmatched reference in /references or
    # /citations: paperId present but null, no DOI, no title. `.get(key,
    # default)` would pass the null straight through and crash RawRecord's
    # required str field -- must fall through to "" instead.
    unmatched = {"paperId": None, "externalIds": {}}
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"data": [unmatched]}))
    connector = SemanticScholarConnector(contact_email="test@example.com")
    result = connector.search("query", QueryFilters())
    assert result.status == "ok"
    assert result.records[0].raw_source_id == ""
    assert result.records[0].title == "(untitled)"


@respx.mock
def test_semantic_scholar_search_treats_null_data_as_zero_results() -> None:
    # Confirmed live: S2 can return {"data": null} rather than {"data": []}.
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"data": None}))
    connector = SemanticScholarConnector(contact_email="test@example.com")
    result = connector.search("query", QueryFilters())
    assert result.status == "zero_results"
    assert result.records == []
