"""Per-field source-priority rules for Phase 7 enrichment merging.

docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 8: "Never silently overwrite
higher-quality metadata with lower-quality metadata." A field is only
overwritten when the candidate source outranks the source that produced the
value currently stored (tracked in records.field_provenance_json), or when
the field is currently empty.
"""

from __future__ import annotations

# Best source first. A source absent from a field's list is still allowed to
# fill an EMPTY field, but is never allowed to overwrite a value that came
# from a ranked source for that field.
FIELD_SOURCE_PRIORITY: dict[str, list[str]] = {
    "title": ["crossref", "openalex", "semantic_scholar", "unpaywall"],
    "year": ["crossref", "openalex", "semantic_scholar", "unpaywall"],
    "venue": ["crossref", "openalex", "semantic_scholar"],
    "publication_type": ["crossref", "openalex", "semantic_scholar"],
    "abstract": ["crossref", "openalex", "semantic_scholar"],
    "abstract_source": ["crossref", "openalex", "semantic_scholar"],
    "keywords": ["openalex", "crossref"],
    "publisher": ["crossref", "openalex"],
    "issn": ["crossref", "openalex"],
    "citation_count": ["openalex", "semantic_scholar", "crossref"],
    "oa_status": ["unpaywall", "openalex", "semantic_scholar"],
    "oa_url": ["unpaywall", "openalex", "semantic_scholar"],
}

MERGEABLE_FIELDS = tuple(FIELD_SOURCE_PRIORITY.keys())


def should_overwrite(field_name: str, current_value: object, current_source: str | None, candidate_source: str) -> bool:
    """Whether a candidate value from `candidate_source` should replace the
    current value (from `current_source`, or None if the field is empty)."""
    if current_value in (None, "", [], "[]"):
        return True
    priority = FIELD_SOURCE_PRIORITY.get(field_name, [])
    if current_source not in priority:
        # Existing value has no tracked/ranked provenance (e.g. it came from the
        # original search connector, before enrichment ran) -- treat it as the
        # weakest possible source and let any enrichment fetch improve it.
        return True
    if candidate_source not in priority:
        # Candidate isn't a ranked source for this field: never risk downgrading
        # a value that a ranked source produced.
        return False
    return priority.index(candidate_source) < priority.index(current_source)
