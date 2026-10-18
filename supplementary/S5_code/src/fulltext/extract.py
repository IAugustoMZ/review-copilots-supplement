"""Phase 14: extract text from a resolved OA location into
data/verified/<record_id>.txt -- never the original PDF, matching this
project's metadata-only storage policy (CLAUDE.md: "Store only metadata and
abstracts -- not full paper PDFs"); only the plain extracted text is kept,
and even that only for records this pipeline has itself verified as legally
open access via resolve.py.
"""

from __future__ import annotations

import io
import re

import httpx
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from review_pipeline.src.common.config import DATA_DIR

VERIFIED_DIR = DATA_DIR / "verified"

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _extract_pdf_text(content: bytes) -> str | None:
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError):
        return None
    text = "\n".join(pages).strip()
    return text or None


def _extract_html_text(content: str) -> str | None:
    text = _TAG_RE.sub(" ", content)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def extract_text_from_url(client: httpx.Client, url: str, timeout: float = 20.0) -> str | None:
    """Fetches url and extracts plain text -- PDF via pypdf, HTML via crude
    tag-stripping (consistent with this project's existing JATS-stripping
    convention in the Crossref connector; no HTML-parsing dependency added
    for what's meant to be a best-effort text pull, not a structured parse).
    Returns None on any fetch/parse failure -- never raises, since a failed
    extraction for one record must not abort a batch."""
    try:
        response = client.get(url, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None

    content_type = response.headers.get("content-type", "")
    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        return _extract_pdf_text(response.content)
    return _extract_html_text(response.text)


def save_extracted_text(record_id: str, text: str) -> None:
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIED_DIR / f"{record_id}.txt").write_text(text, encoding="utf-8")
