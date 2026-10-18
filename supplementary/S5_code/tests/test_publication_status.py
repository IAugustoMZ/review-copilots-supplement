from __future__ import annotations

from review_pipeline.src.common.publication_status import classify_publication_status


def test_passthrough_values_are_returned_unchanged() -> None:
    assert classify_publication_status("peer_reviewed_journal") == "peer_reviewed_journal"
    assert classify_publication_status("preprint") == "preprint"


def test_crossref_posted_content_maps_to_preprint() -> None:
    assert classify_publication_status("posted-content") == "preprint"


def test_report_types_map_to_technical_report() -> None:
    assert classify_publication_status("report") == "technical_report"


def test_arxiv_doi_prefix_maps_to_preprint_even_with_unknown_type() -> None:
    assert classify_publication_status(None, doi="10.48550/arXiv.2402.10977") == "preprint"
    assert classify_publication_status(None, doi="https://doi.org/10.48550/arXiv.2402.10977") == "preprint"


def test_unrecognized_type_is_unknown_not_guessed() -> None:
    assert classify_publication_status("dataset") == "unknown"
    assert classify_publication_status(None) == "unknown"
