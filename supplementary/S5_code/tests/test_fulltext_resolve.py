from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import respx

from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.fulltext import resolve as resolve_module
from review_pipeline.src.fulltext.resolve import resolve_fulltext
from review_pipeline.src.search.connectors import openalex, semantic_scholar, unpaywall


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record(engine, **overrides) -> str:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    record_id = overrides.pop("record_id", "r1")
    title = overrides.pop("title", "A Paper")
    values = {
        "record_id": record_id,
        "title": title,
        "title_normalized": normalize_title(title),
        "publication_status": "peer_reviewed_journal",
        "year": 2025,
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    with engine.begin() as conn:
        conn.execute(records_table.insert().values(**values))
    return record_id


@respx.mock
def test_resolve_fulltext_stops_at_first_accessible_publisher_url(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, oa_url="https://good.example.com/paper.pdf", oa_status="gold")
    respx.head("https://good.example.com/paper.pdf").mock(return_value=httpx.Response(200))

    summary = resolve_fulltext(engine, "r1", httpx.Client(), "test@example.com")
    assert summary.resolved_source == "publisher"
    assert len(summary.attempts) == 1  # stopped immediately, didn't try Unpaywall etc.


@respx.mock
def test_resolve_fulltext_falls_through_to_unpaywall_when_publisher_url_blocked(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, doi="10.1/x", oa_url="https://blocked.example.com/paper.pdf", oa_status="gold")
    respx.head("https://blocked.example.com/paper.pdf").mock(return_value=httpx.Response(403))
    respx.get(f"{unpaywall.BASE_URL}/10.1/x").mock(
        return_value=httpx.Response(
            200, json={"doi": "10.1/x", "title": "A Paper", "oa_status": "green", "best_oa_location": {"url": "https://repo.example.com/paper.pdf"}}
        )
    )
    respx.head("https://repo.example.com/paper.pdf").mock(return_value=httpx.Response(200))

    summary = resolve_fulltext(engine, "r1", httpx.Client(), "test@example.com")
    assert summary.resolved_source == "unpaywall"
    sources_tried = [a.source for a in summary.attempts]
    assert sources_tried == ["publisher", "unpaywall"]


@respx.mock
def test_resolve_fulltext_no_doi_skips_doi_based_sources(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, doi=None)
    from review_pipeline.src.search.connectors.arxiv import BASE_URL as ARXIV_BASE_URL

    respx.get(ARXIV_BASE_URL).mock(
        return_value=httpx.Response(200, text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>')
    )
    summary = resolve_fulltext(engine, "r1", httpx.Client(), "test@example.com")
    # no oa_url, no doi -- nothing to check except arXiv fallback (title-only)
    assert summary.resolved_source is None
    assert summary.arxiv_fallback_used is True
    sources_tried = [a.source for a in summary.attempts]
    assert "unpaywall" not in sources_tried
    assert "openalex" not in sources_tried


@respx.mock
def test_resolve_fulltext_all_sources_fail_falls_back_to_arxiv(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, doi="10.1/x")
    respx.get(f"{unpaywall.BASE_URL}/10.1/x").mock(return_value=httpx.Response(404))
    respx.get(f"{openalex.BASE_URL}/doi:10.1/x").mock(return_value=httpx.Response(404))
    respx.get(semantic_scholar.DOI_LOOKUP_URL.format(doi="10.1/x")).mock(return_value=httpx.Response(404))
    from review_pipeline.src.search.connectors.arxiv import BASE_URL as ARXIV_BASE_URL

    respx.get(ARXIV_BASE_URL).mock(
        return_value=httpx.Response(200, text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>')
    )

    summary = resolve_fulltext(engine, "r1", httpx.Client(), "test@example.com")
    assert summary.arxiv_fallback_used is True
    assert summary.resolved_source is None
    assert "institutional_repository" in summary.unimplemented_sources


def test_check_url_accessible_handles_network_error_gracefully() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("boom"))))
    assert resolve_module._check_url_accessible(client, "https://example.com") is False
