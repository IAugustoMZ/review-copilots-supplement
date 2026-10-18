from __future__ import annotations

import httpx
import respx

from review_pipeline.src.search.connectors.base import QueryFilters
from review_pipeline.src.search.connectors.openalex import (
    BASE_URL,
    MAX_PER_PAGE,
    OpenAlexConnector,
    build_and_of_or_filter,
    reconstruct_abstract,
)

SAMPLE_WORK = {
    "id": "https://openalex.org/W1234567890",
    "doi": "https://doi.org/10.1016/j.dche.2026.100999",
    "title": "Retrieval-Augmented Generation for P&ID Interpretation",
    "publication_year": 2025,
    "type": "article",
    "primary_location": {"source": {"display_name": "Computers & Chemical Engineering", "host_organization_name": "Elsevier", "issn": ["0098-1354"]}},
    "authorships": [{"author": {"display_name": "Alice Wang"}}],
    "abstract_inverted_index": {"We": [0], "propose": [1], "a": [2], "RAG": [3], "system.": [4]},
    "open_access": {"oa_status": "hybrid", "oa_url": "https://example.com/oa.pdf"},
    "cited_by_count": 7,
    "keywords": [{"display_name": "retrieval augmented generation"}],
}


def test_reconstruct_abstract() -> None:
    assert reconstruct_abstract({"We": [0], "propose": [1], "X": [2]}) == "We propose X"
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


@respx.mock
def test_openalex_search_ok() -> None:
    respx.get(BASE_URL).mock(
        return_value=httpx.Response(200, json={"meta": {"db_version": "2026-08-01"}, "results": [SAMPLE_WORK]})
    )
    connector = OpenAlexConnector(contact_email="test@example.com")
    result = connector.search("RAG P&ID", QueryFilters(rows=10))

    assert result.status == "ok"
    record = result.records[0]
    assert record.doi == "10.1016/j.dche.2026.100999"
    assert record.venue == "Computers & Chemical Engineering"
    assert record.publication_type == "peer_reviewed_journal"
    assert record.abstract == "We propose a RAG system."
    assert record.abstract_source == "openalex"
    assert record.oa_status == "hybrid"
    assert record.citation_count == 7
    assert record.authors[0].surname == "Wang"


@respx.mock
def test_openalex_zero_results() -> None:
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"meta": {}, "results": []}))
    connector = OpenAlexConnector(contact_email="test@example.com")
    result = connector.search("nothing matches this", QueryFilters())
    assert result.status == "zero_results"


@respx.mock
def test_openalex_clamps_rows_to_api_max() -> None:
    route = respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"meta": {}, "results": []}))
    connector = OpenAlexConnector(contact_email="test@example.com")
    connector.search("query", QueryFilters(rows=500))
    assert route.calls.last.request.url.params["per-page"] == str(MAX_PER_PAGE)


@respx.mock
def test_openalex_falls_back_when_id_is_explicitly_null() -> None:
    # `.get(key, default)` only falls through when the key is absent -- a
    # present-but-null "id" would crash RawRecord's required str field
    # unless the fallback chain is `or`-based instead.
    malformed = {"id": None, "title": None, "doi": None}
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"meta": {}, "results": [malformed]}))
    connector = OpenAlexConnector(contact_email="test@example.com")
    result = connector.search("query", QueryFilters())
    assert result.status == "ok"
    assert result.records[0].raw_source_id == ""
    assert result.records[0].title == "(untitled)"


def test_build_and_of_or_filter_joins_or_within_and_and_between_groups() -> None:
    result = build_and_of_or_filter([["LLM", "copilot"], ["chemical engineering", "PSE"]])
    assert result == (
        "title_and_abstract.search:LLM|copilot,title_and_abstract.search:chemical engineering|PSE"
    )


def test_build_and_of_or_filter_skips_empty_groups() -> None:
    result = build_and_of_or_filter([[], ["LLM"]])
    assert result == "title_and_abstract.search:LLM"


def test_build_and_of_or_filter_escapes_reserved_characters() -> None:
    # a literal "|" or "," inside a term would smuggle in a stray OR-separator
    # or AND-group-boundary that the term never meant -- must be stripped.
    result = build_and_of_or_filter([["a|b, c"]])
    assert result == "title_and_abstract.search:a b  c"


@respx.mock
def test_openalex_boolean_filter_without_hint_falls_back_to_recency_sort() -> None:
    route = respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"meta": {}, "results": []}))
    connector = OpenAlexConnector(contact_email="test@example.com")
    boolean_filter = build_and_of_or_filter([["LLM", "copilot"], ["chemical engineering"]])
    connector.search(
        "this huge free-text query would otherwise be sent and silently break",
        QueryFilters(extra={"boolean_filter": boolean_filter}),
    )

    sent_params = route.calls.last.request.url.params
    assert "search" not in sent_params
    assert sent_params["filter"] == boolean_filter
    assert sent_params["sort"] == "publication_date:desc"


@respx.mock
def test_openalex_boolean_filter_with_hint_uses_hint_as_search_and_no_sort_override() -> None:
    route = respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"meta": {}, "results": []}))
    connector = OpenAlexConnector(contact_email="test@example.com")
    boolean_filter = build_and_of_or_filter([["LLM", "copilot"], ["chemical engineering"]])
    connector.search(
        "this huge free-text query would otherwise be sent and silently break",
        QueryFilters(extra={"boolean_filter": boolean_filter, "relevance_hint": "LLM chemical engineering"}),
    )

    sent_params = route.calls.last.request.url.params
    assert sent_params["search"] == "LLM chemical engineering"
    assert sent_params["filter"] == boolean_filter
    assert "sort" not in sent_params


@respx.mock
def test_openalex_falls_back_to_search_param_when_no_boolean_filter() -> None:
    route = respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"meta": {}, "results": []}))
    connector = OpenAlexConnector(contact_email="test@example.com")
    connector.search("RAG P&ID", QueryFilters())

    sent_params = route.calls.last.request.url.params
    assert sent_params["search"] == "RAG P&ID"
    assert "sort" not in sent_params


@respx.mock
def test_openalex_combines_boolean_filter_with_date_filters() -> None:
    route = respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"meta": {}, "results": []}))
    connector = OpenAlexConnector(contact_email="test@example.com")
    boolean_filter = build_and_of_or_filter([["LLM"]])
    connector.search("query", QueryFilters(from_year=2023, extra={"boolean_filter": boolean_filter}))

    sent_filter = route.calls.last.request.url.params["filter"]
    assert "from_publication_date:2023-01-01" in sent_filter
    assert boolean_filter in sent_filter
