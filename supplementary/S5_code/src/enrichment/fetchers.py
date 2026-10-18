"""Builds the Phase 7 enrichment fetcher set from env-configured contact info,
reusing the same connector modules Phase 3 search uses (fetch_by_doi
functions live alongside each connector's search() implementation)."""

from __future__ import annotations

import os

from review_pipeline.src.common.config import load_yaml
from review_pipeline.src.enrichment.enricher import EnrichmentFetcher
from review_pipeline.src.search.connectors import crossref, openalex, semantic_scholar, unpaywall
from review_pipeline.src.search.connectors.http_client import build_client


def build_default_fetchers() -> dict[str, EnrichmentFetcher]:
    """Raises RuntimeError if no contact email is configured (same policy as
    Phase 3/4's build_default_connectors -- see CLAUDE.md "Ethics and rate
    limiting" and Unpaywall's own terms of use, which require a real email)."""
    connectors_cfg = load_yaml("search_queries.yaml").get("connectors", {})
    email_env = connectors_cfg.get("contact_email_env", "REVIEW_PIPELINE_CONTACT_EMAIL")
    contact_email = os.environ.get(email_env)
    if not contact_email:
        raise RuntimeError(
            f"No contact email configured. Set the {email_env} environment variable "
            f"(required for Crossref/OpenAlex/Unpaywall polite-pool access)."
        )

    crossref_client = build_client(contact_email)
    openalex_client = build_client(contact_email)
    s2_key_env = connectors_cfg.get("semantic_scholar_api_key_env", "SEMANTIC_SCHOLAR_API_KEY")
    s2_key = os.environ.get(s2_key_env)
    s2_client = build_client(contact_email, extra_headers={"x-api-key": s2_key} if s2_key else None)
    unpaywall_client = build_client(contact_email)

    return {
        "crossref": lambda doi: crossref.fetch_by_doi(crossref_client, doi, contact_email),
        "openalex": lambda doi: openalex.fetch_by_doi(openalex_client, doi, contact_email),
        "semantic_scholar": lambda doi: semantic_scholar.fetch_by_doi(s2_client, doi, contact_email),
        "unpaywall": lambda doi: unpaywall.fetch_by_doi(unpaywall_client, doi, contact_email),
    }
