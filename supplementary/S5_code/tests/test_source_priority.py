from __future__ import annotations

from review_pipeline.src.enrichment.source_priority import should_overwrite


def test_empty_current_value_is_always_overwritten() -> None:
    assert should_overwrite("abstract", None, None, "semantic_scholar") is True
    assert should_overwrite("keywords", [], None, "crossref") is True
    assert should_overwrite("abstract", "", None, "openalex") is True


def test_higher_priority_source_overwrites_lower() -> None:
    # crossref outranks semantic_scholar for abstract
    assert should_overwrite("abstract", "old text", "semantic_scholar", "crossref") is True


def test_lower_priority_source_does_not_overwrite_higher() -> None:
    assert should_overwrite("abstract", "old text", "crossref", "semantic_scholar") is False


def test_same_source_does_not_reapply() -> None:
    assert should_overwrite("abstract", "old text", "crossref", "crossref") is False


def test_untracked_current_source_is_always_overwritten() -> None:
    # a value present before Phase 7 enrichment ran (no provenance recorded yet)
    assert should_overwrite("citation_count", 3, None, "semantic_scholar") is True


def test_unranked_candidate_source_never_downgrades_a_ranked_value() -> None:
    assert should_overwrite("oa_status", "gold", "unpaywall", "some_new_source") is False
