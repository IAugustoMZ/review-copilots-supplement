"""Phase 4 orchestration: run one CE1-14 family across a set of connectors,
query_id-namespaced as `<family>.<connector>` so downstream diagnostics
(chemical_pse_coverage_matrix.csv, Phase 21) can attribute records back to the
family that found them. See docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 4.
"""

from __future__ import annotations

import os
from dataclasses import replace

from sqlalchemy import Engine

from review_pipeline.src.common.config import load_yaml
from review_pipeline.src.search.connectors.base import (
    ConnectorResponse,
    QueryFilters,
    SearchConnector,
)
from review_pipeline.src.search.connectors.crossref import CrossrefConnector
from review_pipeline.src.search.connectors.openalex import OpenAlexConnector, build_and_of_or_filter
from review_pipeline.src.search.connectors.semantic_scholar import SemanticScholarConnector
from review_pipeline.src.search.executor import run_search
from review_pipeline.src.search.family_filter import FamilyFilterConnector
from review_pipeline.src.search.query_builder import (
    QueryFamily,
    SearchQueriesConfig,
    load_search_queries_config,
    render_query_string,
)


def build_default_connectors(config: SearchQueriesConfig | None = None) -> list[SearchConnector]:
    """Instantiate the primary connectors from connectors.* settings + env vars.

    Raises RuntimeError if no contact email is configured -- Crossref/OpenAlex's
    polite pool and this project's own scraping-ethics policy both require one
    (see CLAUDE.md "Ethics and rate limiting"); we don't silently fall back to
    an unset/anonymous identity.
    """
    connectors_cfg = load_yaml("search_queries.yaml").get("connectors", {})
    email_env = connectors_cfg.get("contact_email_env", "REVIEW_PIPELINE_CONTACT_EMAIL")
    contact_email = os.environ.get(email_env)
    if not contact_email:
        raise RuntimeError(
            f"No contact email configured. Set the {email_env} environment variable "
            f"(used for Crossref/OpenAlex polite-pool access and connector User-Agent)."
        )
    s2_key_env = connectors_cfg.get("semantic_scholar_api_key_env", "SEMANTIC_SCHOLAR_API_KEY")
    s2_key = os.environ.get(s2_key_env)

    return [
        CrossrefConnector(contact_email=contact_email),
        OpenAlexConnector(contact_email=contact_email),
        SemanticScholarConnector(contact_email=contact_email, api_key=s2_key),
    ]


def run_ce_family(
    engine: Engine,
    family_key: str,
    connectors: list[SearchConnector] | None = None,
    config: SearchQueriesConfig | None = None,
    filters: QueryFilters | None = None,
    iteration: int = 1,
) -> list[tuple[str, str, ConnectorResponse]]:
    """Run one CE family across all given connectors. Returns
    [(search_run_id, connector_name, response), ...] for reporting."""
    config = config or load_search_queries_config()
    connectors = connectors if connectors is not None else build_default_connectors(config)
    family: QueryFamily = config.families[family_key]

    query_text = render_query_string(config.genai_block, config.domain_block, family.theme_block)
    if filters is None:
        connectors_cfg = load_yaml("search_queries.yaml").get("connectors", {})
        filters = QueryFilters(rows=connectors_cfg.get("default_rows", 50), from_year=connectors_cfg.get("from_year"))

    results = []
    for connector in connectors:
        wrapped = FamilyFilterConnector(
            inner=connector,
            genai_terms=config.genai_block,
            domain_terms=config.domain_block,
            theme_terms=family.theme_block,
        )
        query_id = f"{family_key}.{connector.name}"
        connector_filters = filters
        if connector.name == "openalex":
            # OpenAlex's free-text `search` param silently returns zero
            # results for a query this long (~26-30 terms) -- see
            # openalex.build_and_of_or_filter's docstring. Route it through
            # the filter-based boolean-AND-of-OR-groups form instead, using
            # only genai+domain (mirrors matches_genai_domain, the D0 gate
            # itself -- theme is deliberately not required here either).
            boolean_filter = build_and_of_or_filter([config.genai_block, config.domain_block])
            # A short, generic stand-in for OpenAlex's `search` relevance
            # ranking only -- NOT the actual eligibility gate (that's the
            # filter above); the full query_text remains what's logged to
            # search_runs for methodology reporting.
            relevance_hint = f"{config.genai_block[0]} {config.domain_block[0]}"
            connector_filters = replace(
                filters,
                extra={**filters.extra, "boolean_filter": boolean_filter, "relevance_hint": relevance_hint},
            )
        search_run_id, response = run_search(
            engine, wrapped, query_id=query_id, query_text=query_text, filters=connector_filters, iteration=iteration
        )
        results.append((search_run_id, connector.name, response))
    return results
