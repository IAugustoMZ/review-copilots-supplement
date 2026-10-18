from __future__ import annotations

import httpx
import respx

from review_pipeline.src.search.connectors import crossref, openalex, semantic_scholar, unpaywall


@respx.mock
def test_crossref_fetch_by_doi_ok() -> None:
    respx.get(f"{crossref.BASE_URL}/10.1016/j.dche.2026.100123").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "DOI": "10.1016/j.dche.2026.100123",
                    "title": ["LLM Agent for Process Simulation"],
                    "type": "journal-article",
                    "container-title": ["Digital Chemical Engineering"],
                    "published": {"date-parts": [[2026]]},
                }
            },
        )
    )
    client = httpx.Client()
    record = crossref.fetch_by_doi(client, "10.1016/j.dche.2026.100123", "test@example.com")
    assert record is not None
    assert record.title == "LLM Agent for Process Simulation"
    assert record.venue == "Digital Chemical Engineering"


@respx.mock
def test_crossref_fetch_by_doi_not_found() -> None:
    respx.get(f"{crossref.BASE_URL}/10.1/missing").mock(return_value=httpx.Response(404))
    client = httpx.Client()
    assert crossref.fetch_by_doi(client, "10.1/missing", "test@example.com") is None


@respx.mock
def test_openalex_fetch_by_doi_ok() -> None:
    respx.get(f"{openalex.BASE_URL}/doi:10.1/x").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "https://openalex.org/W123",
                "doi": "https://doi.org/10.1/x",
                "title": "A Paper",
                "publication_year": 2025,
                "primary_location": {"source": {"display_name": "Some Journal"}},
                "open_access": {"oa_status": "gold", "oa_url": "https://example.com/paper.pdf"},
                "cited_by_count": 7,
            },
        )
    )
    client = httpx.Client()
    record = openalex.fetch_by_doi(client, "10.1/x", "test@example.com")
    assert record is not None
    assert record.oa_status == "gold"
    assert record.oa_url == "https://example.com/paper.pdf"
    assert record.citation_count == 7


@respx.mock
def test_openalex_fetch_by_doi_not_found() -> None:
    respx.get(f"{openalex.BASE_URL}/doi:10.1/missing").mock(return_value=httpx.Response(404))
    client = httpx.Client()
    assert openalex.fetch_by_doi(client, "10.1/missing", "test@example.com") is None


@respx.mock
def test_semantic_scholar_fetch_by_doi_ok() -> None:
    respx.get(semantic_scholar.DOI_LOOKUP_URL.format(doi="10.1/x")).mock(
        return_value=httpx.Response(
            200,
            json={
                "paperId": "abc123",
                "title": "A Paper",
                "abstract": "An abstract.",
                "citationCount": 5,
                "externalIds": {"DOI": "10.1/x"},
            },
        )
    )
    client = httpx.Client()
    record = semantic_scholar.fetch_by_doi(client, "10.1/x", "test@example.com")
    assert record is not None
    assert record.abstract == "An abstract."
    assert record.abstract_source == "semantic_scholar"
    assert record.citation_count == 5


@respx.mock
def test_unpaywall_fetch_by_doi_ok() -> None:
    respx.get(f"{unpaywall.BASE_URL}/10.1/x").mock(
        return_value=httpx.Response(
            200,
            json={
                "doi": "10.1/x",
                "title": "A Paper",
                "oa_status": "green",
                "best_oa_location": {"url": "https://repo.example.com/paper.pdf"},
            },
        )
    )
    client = httpx.Client()
    record = unpaywall.fetch_by_doi(client, "10.1/x", "test@example.com")
    assert record is not None
    assert record.oa_status == "green"
    assert record.oa_url == "https://repo.example.com/paper.pdf"


@respx.mock
def test_unpaywall_fetch_by_doi_not_found() -> None:
    respx.get(f"{unpaywall.BASE_URL}/10.1/missing").mock(return_value=httpx.Response(404))
    client = httpx.Client()
    assert unpaywall.fetch_by_doi(client, "10.1/missing", "test@example.com") is None
