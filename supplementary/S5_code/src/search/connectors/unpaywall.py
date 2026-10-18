"""Unpaywall connector. https://api.unpaywall.org/v2/{doi}?email=... -- no API key,
just a contact email (required, not optional, by Unpaywall's own terms of use).

Enrichment-only: Unpaywall is a secondary/OA-resolution source per
docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 8, not one of the three primary
search backbones. It has no search() method, only a DOI lookup.
"""

from __future__ import annotations

import httpx

from review_pipeline.src.search.connectors.base import RawRecord
from review_pipeline.src.search.connectors.http_client import get_with_retry

BASE_URL = "https://api.unpaywall.org/v2"


def fetch_by_doi(client: httpx.Client, doi: str, contact_email: str) -> RawRecord | None:
    """A single work's OA status, or None if not found / request failed."""
    result = get_with_retry(client, f"{BASE_URL}/{doi}", {"email": contact_email})
    if result.response is None or result.response.status_code != 200:
        return None
    payload = result.response.json()
    if not payload or "doi" not in payload:
        return None

    best_oa = payload.get("best_oa_location") or {}
    return RawRecord(
        doi=payload.get("doi"),
        title=payload.get("title") or "(untitled)",
        year=payload.get("year"),
        oa_status=payload.get("oa_status"),
        oa_url=best_oa.get("url") or best_oa.get("url_for_pdf"),
        raw_source_id=payload.get("doi"),
        raw_metadata=payload,
    )
