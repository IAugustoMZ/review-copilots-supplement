"""Wraps a connector so its search() applies the D0 discovery-pool gate
(query_builder.matches_genai_domain: genai AND domain terms, no theme
requirement) before results reach the executor. Deliberately broad -- the
point is to give AI/human screening (Phase 10/11) a real corpus to narrow
down from, not to pre-narrow to near-nothing here. Set strict_theme=True to
additionally require the family's theme block (matches_family), for
diagnostic/precision use only. Errors and genuine zero-results pass through
unchanged; only an 'ok' response is filtered.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from review_pipeline.src.search.connectors.base import (
    ConnectorResponse,
    QueryFilters,
    SearchConnector,
)
from review_pipeline.src.search.query_builder import matches_family, matches_genai_domain


@dataclass
class FamilyFilterConnector(SearchConnector):
    inner: SearchConnector
    genai_terms: list[str]
    domain_terms: list[str]
    theme_terms: list[str]
    strict_theme: bool = False

    def __post_init__(self) -> None:
        self.name = self.inner.name

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        response = self.inner.search(query, filters)
        if response.status != "ok":
            return response

        if self.strict_theme:
            matched = [
                record
                for record in response.records
                if matches_family(record, self.genai_terms, self.domain_terms, self.theme_terms)
            ]
        else:
            matched = [
                record for record in response.records if matches_genai_domain(record, self.genai_terms, self.domain_terms)
            ]
        status = "ok" if matched else "zero_results"
        return replace(response, status=status, records=matched)
