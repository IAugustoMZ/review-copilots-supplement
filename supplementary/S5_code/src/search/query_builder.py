"""Compiles a CE1-14 family definition (genai_block AND domain_block AND
theme_block) into (a) a relevance-query string for a connector's free-text
search param, and (b) a genuine client-side Boolean-AND-of-ORs filter applied
to candidates after retrieval. See the scope note at the top of
config/search_queries.yaml for why both halves exist -- none of
Crossref/OpenAlex/Semantic Scholar's search params support real Boolean
grammar, so (b) is what actually decides family membership.
"""

from __future__ import annotations

from pydantic import BaseModel

from review_pipeline.src.common.config import load_yaml
from review_pipeline.src.search.connectors.base import RawRecord


class QueryFamily(BaseModel):
    key: str
    theme_label: str
    theme_block: list[str]


class SearchQueriesConfig(BaseModel):
    genai_block: list[str]
    domain_block: list[str]
    families: dict[str, QueryFamily]


def load_search_queries_config() -> SearchQueriesConfig:
    raw = load_yaml("search_queries.yaml")
    families = {
        key: QueryFamily(key=key, **value) for key, value in raw.get("families", {}).items()
    }
    return SearchQueriesConfig(
        genai_block=raw.get("genai_block", []),
        domain_block=raw.get("domain_block", []),
        families=families,
    )


def render_query_string(genai_terms: list[str], domain_terms: list[str], theme_terms: list[str]) -> str:
    """A single free-text query string covering all three blocks, for connectors
    whose search param is relevance-ranked bag-of-words rather than real Boolean."""
    all_terms = [*genai_terms, *domain_terms, *theme_terms]
    return " ".join(f'"{t}"' if " " in t else t for t in all_terms)


def _block_matches(text: str, terms: list[str]) -> bool:
    return any(term.lower() in text for term in terms)


def matches_family(record: RawRecord, genai_terms: list[str], domain_terms: list[str], theme_terms: list[str]) -> bool:
    """The strict Boolean-AND-of-ORs test: at least one term from EACH of the
    three blocks (genai/domain/theme) must appear in the title+abstract. High
    precision, low recall -- not used as the default D0 gate (see
    matches_genai_domain), but kept for diagnostics (e.g. a CE-coverage-matrix
    precision column) where the narrower match is exactly what's wanted."""
    haystack = f"{record.title} {record.abstract or ''}".lower()
    return (
        _block_matches(haystack, genai_terms)
        and _block_matches(haystack, domain_terms)
        and _block_matches(haystack, theme_terms)
    )


def matches_genai_domain(record: RawRecord, genai_terms: list[str], domain_terms: list[str]) -> bool:
    """The D0 discovery gate: genai AND domain terms only, no theme
    requirement. Deliberately broader than matches_family() -- the theme
    block is what a CE1-14 family query steers relevance toward, but
    requiring it client-side collapsed D0 to ~3-10 records per family in
    testing, defeating the point of casting a wide net for screening (Phase
    10/11) to narrow down from. theme_block membership is still recorded via
    query_id for the core/contextual tag and CE-coverage matrix, just not used
    to exclude a record from D0."""
    haystack = f"{record.title} {record.abstract or ''}".lower()
    return _block_matches(haystack, genai_terms) and _block_matches(haystack, domain_terms)
