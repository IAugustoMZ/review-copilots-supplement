from __future__ import annotations

from review_pipeline.src.common.text_normalize import normalize_doi, normalize_title


def test_normalize_title_case_and_whitespace() -> None:
    assert normalize_title("  LLM Agents   for Process Control  ") == "llm agents for process control"


def test_normalize_title_punctuation_and_subtitle() -> None:
    a = normalize_title("Digital Twins in Chemical Engineering: A Review")
    b = normalize_title("Digital Twins in Chemical Engineering — A Review")
    assert a == "digital twins in chemical engineering a review"
    assert a == b  # colon and em-dash subtitle separators both strip to the same form


def test_normalize_title_html_entities() -> None:
    assert normalize_title("Process &amp; Plant Safety") == "process plant safety"


def test_normalize_title_unicode_nfkc() -> None:
    # full-width vs regular characters should normalize to the same form
    assert normalize_title("Ｐｒｏｃｅｓｓ") == normalize_title("Process")


def test_normalize_title_hyphenation() -> None:
    assert normalize_title("multi-\nagent") == normalize_title("multiagent") == "multiagent"


def test_normalize_doi_strips_url_and_case() -> None:
    assert normalize_doi("https://doi.org/10.1016/J.DCHE.2026.100123") == "10.1016/j.dche.2026.100123"
    assert normalize_doi("http://dx.doi.org/10.1016/j.dche.2026.100123") == "10.1016/j.dche.2026.100123"
    assert normalize_doi("  10.1016/j.dche.2026.100123.  ") == "10.1016/j.dche.2026.100123"
    assert normalize_doi(None) is None
    assert normalize_doi("") is None
