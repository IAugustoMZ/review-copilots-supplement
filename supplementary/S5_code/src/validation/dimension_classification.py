"""Phase 11 extension: per-dimension, multi-category human classification
(domain, technology, application, architectural_role, validation_focus,
evidence_type), grouped H/M/L entry -- a reviewer sees a numbered category
list once per dimension and says which numbers are high/medium/low, instead
of being asked for a confidence one category at a time.

Reuses classifications_human's existing one-row-per-(record_id, dimension,
coder) + JSON-array-label convention (src/validation/agreement.py's
`_is_multi_label` already expects this shape for multi-label dimensions, and
src/validation/adjudication.py's `_latest_classification` assumes exactly one
row per dimension) -- so per-category level is carried in a sibling
label_levels_json column (migration 005), not as extra rows.

A dimension is recorded as "answered" the moment a row exists for it, even if
the reviewer selected zero categories (an explicit "not applicable"). That is
what lets classify-human resume mid-record without re-asking dimensions
already handled -- see answered_dimensions().
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Engine, select

from review_pipeline.src.common.config import load_yaml
from review_pipeline.src.common.ids import new_id
from review_pipeline.src.db.tables import reflect_table

LEVELS = ("high", "medium", "low")


@dataclass
class DimensionSchema:
    name: str
    multi_value: bool
    categories: list[str]


def load_classification_schema() -> list[DimensionSchema]:
    """Dimensions in config file order (domain, technology, application,
    architectural_role, validation_focus, evidence_type)."""
    raw = load_yaml("classification_schema.yaml")["dimensions"]
    return [
        DimensionSchema(name=name, multi_value=spec["multi_value"], categories=list(spec["categories"]))
        for name, spec in raw.items()
    ]


def answered_dimensions(engine: Engine, record_id: str, reviewer_id: str) -> set[str]:
    """Dimensions this reviewer has already recorded a row for on this
    record -- including 'topical_inclusion'. Used to skip already-answered
    dimensions on resume after a mid-record quit."""
    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        rows = conn.execute(
            select(classifications_human.c.dimension).where(
                classifications_human.c.record_id == record_id,
                classifications_human.c.coder == reviewer_id,
            )
        ).all()
    return {r[0] for r in rows}


def save_dimension_classification(
    engine: Engine,
    record_id: str,
    reviewer_id: str,
    dimension: str,
    categories_by_level: dict[str, list[str]],
) -> None:
    """categories_by_level: {"high": [...], "medium": [...], "low": [...]}.
    Writes one row: label = JSON array of every selected category (high then
    medium then low), label_levels_json = {category: level}. An entirely
    empty selection still writes a row (label='[]') so the dimension counts
    as answered.

    A category named at more than one level keeps only its HIGHEST, and
    appears once in the label array. Nothing stopped the reviewer typing the
    same number under two levels, and the previous code disagreed with
    itself about the result: the label array listed the category twice while
    levels_map silently kept whichever level came last (i.e. the lowest).
    Since P(high) is the training target (ml_assist._parse_high_categories),
    that discrepancy quietly downgraded a category the reviewer had in fact
    marked high."""
    classifications_human = reflect_table(engine, "classifications_human")
    now = datetime.now(UTC).isoformat()

    levels_map: dict[str, str] = {}
    ordered_categories: list[str] = []
    for level in LEVELS:  # high first, so the first assignment seen is the highest
        for category in categories_by_level.get(level, []):
            if category in levels_map:
                continue
            levels_map[category] = level
            ordered_categories.append(category)

    with engine.begin() as conn:
        conn.execute(
            classifications_human.insert().values(
                classification_id=new_id(),
                record_id=record_id,
                dimension=dimension,
                label=json.dumps(ordered_categories),
                confidence=None,
                label_levels_json=json.dumps(levels_map) if levels_map else None,
                coder=reviewer_id,
                timestamp=now,
            )
        )


def save_single_value_classification(
    engine: Engine, record_id: str, reviewer_id: str, dimension: str, category: str | None
) -> None:
    """For single-value dimensions (evidence_type): one row, label = the
    chosen category, or '' if the reviewer chose none."""
    classifications_human = reflect_table(engine, "classifications_human")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            classifications_human.insert().values(
                classification_id=new_id(),
                record_id=record_id,
                dimension=dimension,
                label=category or "",
                confidence=None,
                coder=reviewer_id,
                timestamp=now,
            )
        )
