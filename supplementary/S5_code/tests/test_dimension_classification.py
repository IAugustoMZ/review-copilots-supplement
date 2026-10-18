from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.validation.dimension_classification import (
    answered_dimensions,
    load_classification_schema,
    save_dimension_classification,
    save_single_value_classification,
)


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record(engine, record_id: str = "r1") -> str:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    title = "A Paper"
    with engine.begin() as conn:
        conn.execute(
            records_table.insert().values(
                record_id=record_id,
                title=title,
                title_normalized=normalize_title(title),
                d1=1,
                created_at=now,
                updated_at=now,
            )
        )
    return record_id


def test_load_classification_schema_has_six_dimensions_in_order() -> None:
    schema = load_classification_schema()
    names = [d.name for d in schema]
    assert names == [
        "domain",
        "technology",
        "application",
        "architectural_role",
        "validation_focus",
        "evidence_type",
    ]
    domain = next(d for d in schema if d.name == "domain")
    assert domain.multi_value is True
    assert "chemical_engineering" in domain.categories
    evidence_type = next(d for d in schema if d.name == "evidence_type")
    assert evidence_type.multi_value is False


def test_save_dimension_classification_writes_label_array_and_levels(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    record_id = _insert_record(engine)

    save_dimension_classification(
        engine,
        record_id,
        "alice",
        "domain",
        {"high": ["chemical_engineering", "process_systems_engineering"], "medium": ["manufacturing"], "low": []},
    )

    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        row = dict(conn.execute(classifications_human.select()).mappings().one())

    assert row["dimension"] == "domain"
    assert json.loads(row["label"]) == ["chemical_engineering", "process_systems_engineering", "manufacturing"]
    levels = json.loads(row["label_levels_json"])
    assert levels == {
        "chemical_engineering": "high",
        "process_systems_engineering": "high",
        "manufacturing": "medium",
    }
    assert row["coder"] == "alice"


def test_a_category_named_at_two_levels_keeps_the_highest_and_appears_once(tmp_path: Path) -> None:
    """Nothing stops the reviewer typing the same number under two levels,
    and one live record did exactly that. The old code disagreed with itself
    about the result: the label array listed the category twice while the
    levels map silently kept whichever level came last -- the LOWEST. Since
    P(high) is the training target (ml_assist._parse_high_categories), that
    quietly downgraded a category the reviewer had in fact marked high."""
    engine = _db(tmp_path)
    record_id = _insert_record(engine)

    save_dimension_classification(
        engine, record_id, "alice", "domain", {"high": ["operations"], "medium": [], "low": ["operations"]}
    )

    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        row = dict(conn.execute(classifications_human.select()).mappings().one())

    assert json.loads(row["label"]) == ["operations"]  # once, not twice
    assert json.loads(row["label_levels_json"]) == {"operations": "high"}  # highest wins


def test_save_dimension_classification_with_no_selections_still_writes_a_row(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    record_id = _insert_record(engine)

    save_dimension_classification(engine, record_id, "alice", "domain", {"high": [], "medium": [], "low": []})

    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        row = dict(conn.execute(classifications_human.select()).mappings().one())
    assert json.loads(row["label"]) == []
    assert row["label_levels_json"] is None


def test_save_single_value_classification_writes_plain_label(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    record_id = _insert_record(engine)

    save_single_value_classification(engine, record_id, "alice", "evidence_type", "case_study")

    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        row = dict(conn.execute(classifications_human.select()).mappings().one())
    assert row["dimension"] == "evidence_type"
    assert row["label"] == "case_study"
    assert row["label_levels_json"] is None


def test_answered_dimensions_tracks_per_reviewer_progress(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    record_id = _insert_record(engine)

    assert answered_dimensions(engine, record_id, "alice") == set()

    save_dimension_classification(engine, record_id, "alice", "domain", {"high": ["chemical_engineering"], "medium": [], "low": []})
    assert answered_dimensions(engine, record_id, "alice") == {"domain"}

    save_dimension_classification(engine, record_id, "alice", "technology", {"high": ["llm"], "medium": [], "low": []})
    assert answered_dimensions(engine, record_id, "alice") == {"domain", "technology"}

    # a different reviewer's progress on the same record is tracked separately
    assert answered_dimensions(engine, record_id, "bob") == set()
