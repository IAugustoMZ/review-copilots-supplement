"""Phase 5: venue-level search. A recall diagnostic, not an inclusion
mechanism -- hits are tagged retrieval_source='venue_search' and go through
the same D0->D1 eligibility screen as everything else (docs/
TECHNICAL_IMPLEMENTATION_PLAN.md section 5).

Rather than rely on each connector's own (inconsistent, sometimes unreliable)
container-title filter syntax, venue membership is enforced the same way CE-
family membership is (query_builder.matches_family / family_filter.py):
broad retrieval, then a real client-side test against the record's own
`venue` field, so what "counted" for a venue is auditable independent of
whatever the API's relevance ranking happened to surface.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

from pydantic import BaseModel
from sqlalchemy import Engine

from review_pipeline.src.common.config import load_yaml
from review_pipeline.src.search.ce_search import build_default_connectors
from review_pipeline.src.search.connectors.base import (
    ConnectorResponse,
    QueryFilters,
    SearchConnector,
)
from review_pipeline.src.search.executor import run_search
from review_pipeline.src.search.query_builder import render_query_string


class VenueTarget(BaseModel):
    venue: str
    container_title: str


class VenueSearchConfig(BaseModel):
    venues: list[VenueTarget]
    genai_only_query_terms: list[str]


def load_venue_search_config() -> VenueSearchConfig:
    raw = load_yaml("journals.yaml")
    return VenueSearchConfig(
        venues=[VenueTarget(**v) for v in raw.get("venues", [])],
        genai_only_query_terms=raw.get("genai_only_query_terms", []),
    )


def _venue_matches(record_venue: str | None, container_title: str) -> bool:
    if not record_venue:
        return False
    a, b = record_venue.lower(), container_title.lower()
    return a in b or b in a


def venue_filter_extra(connector_name: str, container_title: str) -> dict:
    """Per-connector server-side venue restriction, where one exists.

    Without this, a venue's genuinely relevant papers can rank far below
    keyword-dense noise in a whole-corpus relevance search (a book chapter
    titled "RAG Artificial Intelligence" out-ranks an on-topic paper that
    merely mentions "LLM" once) -- confirmed by a live smoke test against
    Digital Chemical Engineering before this restriction was added. The
    client-side check in VenueFilterConnector remains as a safety net in
    case a given filter syntax is fuzzy/imprecise.
    """
    if connector_name == "crossref":
        return {"filter": f"container-title:{container_title}"}
    if connector_name == "semantic_scholar":
        return {"venue": container_title}
    # OpenAlex has no filter for a source's display_name text (confirmed live
    # against its own valid-fields error message); it only supports filtering
    # by source.id or source.issn, which journals.yaml doesn't carry yet. Falls
    # back to the client-side VenueFilterConnector check alone for this
    # connector -- a documented, known recall gap, not a silent one.
    return {}


@dataclass
class VenueFilterConnector(SearchConnector):
    inner: SearchConnector
    genai_terms: list[str]
    container_title: str

    def __post_init__(self) -> None:
        self.name = self.inner.name

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        response = self.inner.search(query, filters)
        if response.status != "ok":
            return response

        haystack_terms = [t.lower() for t in self.genai_terms]
        matched = [
            record
            for record in response.records
            if _venue_matches(record.venue, self.container_title)
            and any(term in f"{record.title} {record.abstract or ''}".lower() for term in haystack_terms)
        ]
        status = "ok" if matched else "zero_results"
        return replace(response, status=status, records=matched)


def run_venue_search(
    engine: Engine,
    venue: VenueTarget,
    connectors: list[SearchConnector] | None = None,
    genai_terms: list[str] | None = None,
    filters: QueryFilters | None = None,
    iteration: int = 1,
) -> list[tuple[str, str, ConnectorResponse]]:
    config = load_venue_search_config()
    genai_terms = genai_terms or config.genai_only_query_terms
    connectors = connectors if connectors is not None else build_default_connectors()

    # GenAI/LLM-only query text (no venue name in it -- venue restriction is a
    # separate, server-side filter per connector; folding the venue name into
    # free-text relevance ranking is what buried real hits under keyword-dense
    # noise, see venue_filter_extra's docstring).
    query_text = render_query_string(genai_terms, [], [])
    slug = venue.venue.lower().replace(" ", "_").replace("&", "and")

    results = []
    for connector in connectors:
        wrapped = VenueFilterConnector(inner=connector, genai_terms=genai_terms, container_title=venue.container_title)
        query_id = f"venue.{slug}.{connector.name}"
        # rows=200 (OpenAlex's hard per-page ceiling; Crossref/S2 clamp lower
        # independently) partly compensates connectors without a working
        # server-side venue filter (see venue_filter_extra), where recall
        # otherwise depends entirely on the client-side check below finding a
        # true hit within the relevance-ranked page.
        connector_filters = filters or QueryFilters(
            rows=200, extra=venue_filter_extra(connector.name, venue.container_title)
        )
        search_run_id, response = run_search(
            engine, wrapped, query_id=query_id, query_text=query_text, filters=connector_filters, iteration=iteration
        )
        results.append((search_run_id, connector.name, response))
    return results


def run_all_venue_searches(
    engine: Engine, connectors: list[SearchConnector] | None = None, iteration: int = 1
) -> dict[str, list[tuple[str, str, ConnectorResponse]]]:
    config = load_venue_search_config()
    connectors = connectors if connectors is not None else build_default_connectors()
    return {
        venue.venue: run_venue_search(
            engine, venue, connectors=connectors, genai_terms=config.genai_only_query_terms, iteration=iteration
        )
        for venue in config.venues
    }


def generate_venue_search_log(results_by_venue: dict[str, list[tuple[str, str, ConnectorResponse]]]) -> str:
    """Markdown per docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 5. new_unique_records
    and new_eligible_records require Phase 8 (dedup) and Phase 9 (eligibility) to have
    run against this batch -- until then they're reported as "pending" rather than
    guessed, per the project's no-fabrication rule."""
    lines = [
        "# Venue Search Log",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "| Venue | Connector | Search date | Candidate count | New unique | New eligible |",
        "|---|---|---|---|---|---|",
    ]
    for venue_name, connector_results in results_by_venue.items():
        for _search_run_id, connector_name, response in connector_results:
            candidate_count = len(response.records) if response.status == "ok" else 0
            lines.append(
                f"| {venue_name} | {connector_name} | {datetime.now(UTC).date().isoformat()} | "
                f"{candidate_count} | pending (Phase 8) | pending (Phase 9) |"
            )
    return "\n".join(lines) + "\n"
