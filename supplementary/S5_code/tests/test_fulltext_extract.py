from __future__ import annotations

import io
from pathlib import Path

import httpx
import respx
from pypdf import PdfWriter

from review_pipeline.src.fulltext.extract import (
    _extract_html_text,
    _extract_pdf_text,
    extract_text_from_url,
    save_extracted_text,
)


def _make_minimal_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_extract_pdf_text_from_valid_but_blank_pdf_returns_none() -> None:
    # a blank page has no text -- extraction should return None, not crash
    assert _extract_pdf_text(_make_minimal_pdf_bytes()) is None


def test_extract_pdf_text_from_garbage_bytes_returns_none() -> None:
    assert _extract_pdf_text(b"not a pdf at all") is None


def test_extract_html_text_strips_tags_and_collapses_whitespace() -> None:
    html = "<html><body><p>Hello   world.</p>\n\n<div>Second   part.</div></body></html>"
    assert _extract_html_text(html) == "Hello world. Second part."


def test_extract_html_text_empty_returns_none() -> None:
    assert _extract_html_text("<html></html>") is None


@respx.mock
def test_extract_text_from_url_html_response() -> None:
    respx.get("https://example.com/paper").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/html"}, text="<p>Some real content here.</p>")
    )
    client = httpx.Client()
    text = extract_text_from_url(client, "https://example.com/paper")
    assert text == "Some real content here."


@respx.mock
def test_extract_text_from_url_404_returns_none() -> None:
    respx.get("https://example.com/missing").mock(return_value=httpx.Response(404))
    client = httpx.Client()
    assert extract_text_from_url(client, "https://example.com/missing") is None


def test_extract_text_from_url_network_error_returns_none() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("boom"))))
    assert extract_text_from_url(client, "https://example.com/x") is None


def test_save_extracted_text_writes_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("review_pipeline.src.fulltext.extract.VERIFIED_DIR", tmp_path / "verified")
    save_extracted_text("r1", "some extracted text")
    assert (tmp_path / "verified" / "r1.txt").read_text() == "some extracted text"
