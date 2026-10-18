from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import respx

from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.fulltext import download as download_module
from review_pipeline.src.fulltext.download import (
    DownloadOutcome,
    get_target_record_ids,
    resolve_and_download,
    write_coverage_report,
)
from review_pipeline.src.search.connectors import openalex, semantic_scholar, unpaywall
from review_pipeline.src.search.connectors.arxiv import BASE_URL as ARXIV_BASE_URL

_PDF_BYTES = b"%PDF-1.4 minimal fake pdf content"


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


def _empty_arxiv_feed():
    return httpx.Response(200, text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>')


@respx.mock
def test_downloads_pdf_when_unpaywall_reports_cc_license(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(download_module, "PDF_DIR", tmp_path / "fulltext_pdfs")
    _insert_record(engine, doi="10.1/x")

    respx.get(f"{unpaywall.BASE_URL}/10.1/x").mock(
        return_value=httpx.Response(
            200,
            json={
                "doi": "10.1/x",
                "title": "A Paper",
                "oa_status": "gold",
                "best_oa_location": {"url": "https://good.example.com/paper.pdf", "license": "cc-by"},
            },
        )
    )
    respx.get("https://good.example.com/paper.pdf").mock(
        return_value=httpx.Response(200, content=_PDF_BYTES, headers={"content-type": "application/pdf"})
    )

    outcome = resolve_and_download(engine, "r1", httpx.Client(), "test@example.com")
    assert outcome.status == "downloaded"
    assert outcome.source == "unpaywall"
    assert outcome.license == "cc-by"
    saved = tmp_path / "fulltext_pdfs" / "r1.pdf"
    assert saved.read_bytes() == _PDF_BYTES


@respx.mock
def test_does_not_download_when_license_not_cc(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(download_module, "PDF_DIR", tmp_path / "fulltext_pdfs")
    _insert_record(engine, doi="10.1/x")

    respx.get(f"{unpaywall.BASE_URL}/10.1/x").mock(
        return_value=httpx.Response(
            200,
            json={
                "doi": "10.1/x",
                "title": "A Paper",
                "oa_status": "bronze",
                "best_oa_location": {"url": "https://pub.example.com/paper.pdf", "license": None},
            },
        )
    )
    respx.get(f"{openalex.BASE_URL}/doi:10.1/x").mock(return_value=httpx.Response(404))
    respx.get(semantic_scholar.DOI_LOOKUP_URL.format(doi="10.1/x")).mock(return_value=httpx.Response(404))
    respx.get(ARXIV_BASE_URL).mock(return_value=_empty_arxiv_feed())

    outcome = resolve_and_download(engine, "r1", httpx.Client(), "test@example.com")
    assert outcome.status == "oa_unverified_license"
    assert outcome.source == "unpaywall"
    assert not (tmp_path / "fulltext_pdfs" / "r1.pdf").exists()


@respx.mock
def test_arxiv_fallback_always_downloads(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(download_module, "PDF_DIR", tmp_path / "fulltext_pdfs")
    _insert_record(engine, doi=None, title="Some Preprint")

    arxiv_feed = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2501.00001</id>
        <title>Some Preprint</title>
        <published>2025-01-01T00:00:00Z</published>
        <author><name>A. Author</name></author>
        <link title="pdf" href="https://arxiv.org/pdf/2501.00001"/>
      </entry>
    </feed>"""
    respx.get(ARXIV_BASE_URL).mock(return_value=httpx.Response(200, text=arxiv_feed))
    respx.get("https://arxiv.org/pdf/2501.00001").mock(
        return_value=httpx.Response(200, content=_PDF_BYTES, headers={"content-type": "application/pdf"})
    )

    outcome = resolve_and_download(engine, "r1", httpx.Client(), "test@example.com")
    assert outcome.status == "downloaded"
    assert outcome.source == "arxiv"
    assert (tmp_path / "fulltext_pdfs" / "r1.pdf").exists()


@respx.mock
def test_cc_licensed_but_403_blocked_is_reported_not_bypassed(tmp_path: Path, monkeypatch) -> None:
    # A publisher can bot-block an otherwise CC-licensed PDF (see
    # download.py's _BOT_BLOCK_STATUS_CODES docstring) -- this must be
    # reported honestly, never retried with a spoofed browser User-Agent.
    engine = _db(tmp_path)
    monkeypatch.setattr(download_module, "PDF_DIR", tmp_path / "fulltext_pdfs")
    _insert_record(engine, doi="10.1/x")

    respx.get(f"{unpaywall.BASE_URL}/10.1/x").mock(
        return_value=httpx.Response(
            200,
            json={
                "doi": "10.1/x",
                "title": "A Paper",
                "oa_status": "gold",
                "best_oa_location": {"url": "https://pub.example.com/paper.pdf", "license": "cc-by"},
            },
        )
    )
    respx.get("https://pub.example.com/paper.pdf").mock(return_value=httpx.Response(403, text="blocked"))

    outcome = resolve_and_download(engine, "r1", httpx.Client(), "test@example.com")
    assert outcome.status == "oa_blocked_by_publisher"
    assert not (tmp_path / "fulltext_pdfs" / "r1.pdf").exists()


@respx.mock
def test_cc_licensed_landing_page_only_is_reported_as_download_failed(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(download_module, "PDF_DIR", tmp_path / "fulltext_pdfs")
    _insert_record(engine, doi="10.1/x")

    respx.get(f"{unpaywall.BASE_URL}/10.1/x").mock(
        return_value=httpx.Response(
            200,
            json={
                "doi": "10.1/x",
                "title": "A Paper",
                "oa_status": "gold",
                "best_oa_location": {"url": "https://pub.example.com/article", "license": "cc-by"},
            },
        )
    )
    respx.get("https://pub.example.com/article").mock(
        return_value=httpx.Response(200, text="<html>landing page</html>", headers={"content-type": "text/html"})
    )

    outcome = resolve_and_download(engine, "r1", httpx.Client(), "test@example.com")
    assert outcome.status == "download_failed"
    assert "landing page" in outcome.reason


@respx.mock
def test_follows_citation_pdf_url_meta_tag_from_landing_page(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(download_module, "PDF_DIR", tmp_path / "fulltext_pdfs")
    _insert_record(engine, doi="10.1/x")

    respx.get(f"{unpaywall.BASE_URL}/10.1/x").mock(
        return_value=httpx.Response(
            200,
            json={
                "doi": "10.1/x",
                "title": "A Paper",
                "oa_status": "gold",
                "best_oa_location": {"url": "https://pub.example.com/article", "license": "cc-by"},
            },
        )
    )
    landing_html = (
        '<html><head><meta name="citation_pdf_url" content="https://pub.example.com/article.pdf"></head></html>'
    )
    respx.get("https://pub.example.com/article").mock(
        return_value=httpx.Response(200, text=landing_html, headers={"content-type": "text/html"})
    )
    respx.get("https://pub.example.com/article.pdf").mock(
        return_value=httpx.Response(200, content=_PDF_BYTES, headers={"content-type": "application/pdf"})
    )

    outcome = resolve_and_download(engine, "r1", httpx.Client(), "test@example.com")
    assert outcome.status == "downloaded"
    assert (tmp_path / "fulltext_pdfs" / "r1.pdf").read_bytes() == _PDF_BYTES


@respx.mock
def test_no_oa_found_when_nothing_resolves(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(download_module, "PDF_DIR", tmp_path / "fulltext_pdfs")
    _insert_record(engine, doi="10.1/x")

    respx.get(f"{unpaywall.BASE_URL}/10.1/x").mock(return_value=httpx.Response(404))
    respx.get(f"{openalex.BASE_URL}/doi:10.1/x").mock(return_value=httpx.Response(404))
    respx.get(semantic_scholar.DOI_LOOKUP_URL.format(doi="10.1/x")).mock(return_value=httpx.Response(404))
    respx.get(ARXIV_BASE_URL).mock(return_value=_empty_arxiv_feed())

    outcome = resolve_and_download(engine, "r1", httpx.Client(), "test@example.com")
    assert outcome.status == "no_oa_found"


def test_get_target_record_ids_only_returns_ai_included_d1_records(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    included = _insert_record(engine, record_id="included", d1=1)
    _insert_record(engine, record_id="excluded", d1=1)
    _insert_record(engine, record_id="not_d1", d1=0)

    classifications_ai_table = reflect_table(engine, "classifications_ai")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            classifications_ai_table.insert().values(
                classification_id="c1", record_id="included", dimension="topical_inclusion",
                label="include", confidence="high", rationale="on topic", coder="test", prompt_version="v1", timestamp=now,
            )
        )
        conn.execute(
            classifications_ai_table.insert().values(
                classification_id="c2", record_id="excluded", dimension="topical_inclusion",
                label="exclude", confidence="high", rationale="off topic", coder="test", prompt_version="v1", timestamp=now,
            )
        )

    targets = get_target_record_ids(engine)
    assert [r["record_id"] for r in targets] == [included]


def test_write_coverage_report_includes_all_status_groups(tmp_path: Path) -> None:
    outcomes = [
        DownloadOutcome("r1", "Paper One", "downloaded", "arxiv", "arxiv-license", "https://x/1.pdf", "fulltext_pdfs/r1.pdf"),
        DownloadOutcome("r2", "Paper Two", "oa_unverified_license", "unpaywall", None, "https://x/2.pdf", reason="not CC"),
        DownloadOutcome("r3", "Paper Three", "no_oa_found", reason="nothing found"),
    ]
    out_path = tmp_path / "coverage.md"
    write_coverage_report(outcomes, out_path)
    text = out_path.read_text()
    assert "1/3 PDFs downloaded" in text
    assert "r1" in text and "r2" in text and "r3" in text


def test_include_list_uses_the_latest_ai_label_not_any_label(tmp_path: Path) -> None:
    """Re-screening under amended criteria appends a superseding row rather than
    overwriting, so a record downgraded include -> exclude must drop out of the
    fetch list even though its original include row is still on the table."""
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    NOW = datetime.now(UTC).isoformat()

    records = reflect_table(engine, "records")
    classifications_ai = reflect_table(engine, "classifications_ai")
    with engine.begin() as conn:
        conn.execute(
            records.insert().values(
                record_id="r1",
                title="Downgraded on re-screen",
                title_normalized=normalize_title("Downgraded on re-screen"),
                d0=1,
                d1=1,
                corpus_type="primary",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        conn.execute(
            records.insert().values(
                record_id="r2",
                title="Still included",
                title_normalized=normalize_title("Still included"),
                d0=1,
                d1=1,
                corpus_type="primary",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        for cid, rid, label, ts in [
            ("c1", "r1", "include", "2026-08-29T00:00:00+00:00"),
            ("c2", "r1", "exclude", "2026-08-30T00:00:00+00:00"),
            ("c3", "r2", "include", "2026-08-29T00:00:00+00:00"),
        ]:
            conn.execute(
                classifications_ai.insert().values(
                    classification_id=cid,
                    record_id=rid,
                    dimension="topical_inclusion",
                    label=label,
                    coder="test",
                    timestamp=ts,
                )
            )

    assert [r["record_id"] for r in get_target_record_ids(engine)] == ["r2"]
