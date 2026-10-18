from __future__ import annotations

from review_pipeline.src.search.connectors.base import RawRecord
from review_pipeline.src.search.query_builder import (
    load_search_queries_config,
    matches_family,
    matches_genai_domain,
    render_query_string,
)

GENAI = ["large language model", "LLM"]
DOMAIN = ["chemical engineering", "process industry"]
THEME = ["HAZOP", "process safety"]


def test_render_query_string_quotes_multiword_terms() -> None:
    rendered = render_query_string(GENAI, DOMAIN, THEME)
    assert '"large language model"' in rendered
    assert "LLM" in rendered
    assert '"chemical engineering"' in rendered
    assert '"process safety"' in rendered


def test_matches_family_requires_all_three_blocks() -> None:
    record = RawRecord(
        title="An LLM agent for HAZOP review in the chemical engineering domain",
        abstract="We use a large language model to assist HAZOP studies.",
        raw_source_id="x",
    )
    assert matches_family(record, GENAI, DOMAIN, THEME) is True


def test_matches_family_fails_if_theme_block_absent() -> None:
    record = RawRecord(
        title="An LLM agent for chemical engineering education",
        abstract="We use a large language model to help students.",
        raw_source_id="x",
    )
    # matches genai + domain, but not the HAZOP/process-safety theme block
    assert matches_family(record, GENAI, DOMAIN, THEME) is False


def test_matches_family_fails_if_domain_block_absent() -> None:
    record = RawRecord(
        title="An LLM agent for HAZOP-style safety review in aviation",
        abstract="We use a large language model to assist safety studies.",
        raw_source_id="x",
    )
    assert matches_family(record, GENAI, DOMAIN, THEME) is False


def test_matches_genai_domain_ignores_theme_block() -> None:
    # matches genai + domain but has none of the HAZOP/process-safety theme
    # terms -- the strict matches_family() rejects this (see the test above),
    # but the broader D0 gate must accept it.
    record = RawRecord(
        title="An LLM agent for chemical engineering education",
        abstract="We use a large language model to help students.",
        raw_source_id="x",
    )
    assert matches_genai_domain(record, GENAI, DOMAIN) is True


def test_matches_genai_domain_still_requires_domain_block() -> None:
    record = RawRecord(title="An LLM agent for aviation safety", abstract="Nothing chemical here.", raw_source_id="x")
    assert matches_genai_domain(record, GENAI, DOMAIN) is False


def test_load_search_queries_config_families_are_contiguous_and_populated() -> None:
    """Asserts the config's invariants rather than a family count: the count is
    expected to grow as the protocol adds themes (CE15 for benchmark datasets,
    CE16-CE18 for benchmark generation), and a hard-coded number turns every
    legitimate expansion into a spurious failure."""
    config = load_search_queries_config()
    assert config.families, "no CE families loaded"
    assert set(config.families) == {f"CE{i}" for i in range(1, len(config.families) + 1)}
    assert config.genai_block
    assert config.domain_block
    for key, family in config.families.items():
        assert family.theme_block, f"{key} has an empty theme_block"
        assert family.theme_label, f"{key} has no theme_label"
