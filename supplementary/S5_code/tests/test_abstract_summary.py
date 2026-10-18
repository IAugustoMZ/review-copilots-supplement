from __future__ import annotations

from review_pipeline.src.validation.abstract_summary import (
    NOT_AVAILABLE,
    generate_abstract_summary,
    lint_summary,
)


def test_generate_abstract_summary_takes_first_two_sentences() -> None:
    abstract = "First sentence. Second sentence. Third sentence that should be dropped."
    summary = generate_abstract_summary(abstract)
    assert summary == "First sentence. Second sentence."


def test_generate_abstract_summary_respects_max_sentences() -> None:
    abstract = "One. Two. Three. Four."
    assert generate_abstract_summary(abstract, max_sentences=1) == "One."


def test_generate_abstract_summary_missing_abstract_returns_not_available() -> None:
    assert generate_abstract_summary(None) == NOT_AVAILABLE
    assert generate_abstract_summary("") == NOT_AVAILABLE
    assert generate_abstract_summary("   ") == NOT_AVAILABLE


def test_generate_abstract_summary_falls_back_when_lint_fails() -> None:
    abstract = "This should clearly be included in the review. More text follows here."
    assert generate_abstract_summary(abstract) == NOT_AVAILABLE


def test_lint_summary_detects_banned_phrases_case_insensitively() -> None:
    assert lint_summary("This paper CLEARLY demonstrates something.") == ["clearly"]
    assert lint_summary("A neutral, descriptive sentence.") == []


def test_lint_summary_detects_capability_level_language() -> None:
    assert "level 3" in lint_summary("The system is at Level 3 capability.")
