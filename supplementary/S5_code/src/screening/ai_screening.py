"""Phase 10: AI screening, run through a Claude Code subagent rather than a
direct Anthropic API call (no ANTHROPIC_API_KEY is available in this
environment). This module owns the mechanical halves only -- exporting a
batch of D1-eligible, not-yet-AI-screened records for a subagent to judge,
and ingesting the subagent's structured decisions back into the DB. The
judgment itself happens outside this module, in the subagent's own reasoning
against config/screening_criteria.yaml.

Structural safety property: ingest_decisions() only ever writes
screening_decisions rows with decision_type='ai'. It never touches
final_screening_decision. A record only becomes excluded "for real" once a
human/adjudication step (Phase 11-13, not yet built) reviews it -- so a
mis-scored or low-confidence AI call can never silently and irreversibly
remove a record from the corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError, model_validator
from sqlalchemy import Engine, select

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.db.tables import reflect_table

Decision = Literal["include", "exclude", "uncertain"]
Confidence = Literal["high", "medium", "low"]


class ScreeningDecision(BaseModel):
    record_id: str
    decision: Decision
    confidence: Confidence
    rationale: str
    exclusion_reason: str | None = None
    uncertainty_reason: str | None = None

    @model_validator(mode="after")
    def _reason_matches_decision(self) -> ScreeningDecision:
        if self.decision == "exclude" and not self.exclusion_reason:
            raise ValueError("decision='exclude' requires exclusion_reason")
        if self.decision == "uncertain" and not self.uncertainty_reason:
            raise ValueError("decision='uncertain' requires uncertainty_reason")
        return self


def export_batch(engine: Engine, limit: int | None = None) -> list[dict]:
    """D1-eligible, canonical records with no existing AI screening decision yet."""
    records_table = reflect_table(engine, "records")
    screening_decisions_table = reflect_table(engine, "screening_decisions")

    with engine.connect() as conn:
        already_screened = {
            row[0]
            for row in conn.execute(
                select(screening_decisions_table.c.record_id).where(
                    screening_decisions_table.c.decision_type == "ai"
                )
            )
        }
        query = records_table.select().where(
            records_table.c.d1 == 1, records_table.c.canonical_record_id.is_(None)
        )
        rows = [dict(r) for r in conn.execute(query).mappings().all()]

    pending = [r for r in rows if r["record_id"] not in already_screened]
    if limit is not None:
        pending = pending[:limit]

    return [
        {
            "record_id": r["record_id"],
            "title": r["title"],
            "abstract": r["abstract"],
            "venue": r["venue"],
            "year": r["year"],
            "publication_status": r["publication_status"],
            "core_or_contextual": r["core_or_contextual"],
        }
        for r in pending
    ]


def write_export(batch: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class LoadDecisionsResult:
    decisions: list[ScreeningDecision] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def load_decisions(path: Path) -> LoadDecisionsResult:
    """Parses and validates every entry independently -- one malformed entry
    is reported, not silently dropped, and does not block valid entries."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = LoadDecisionsResult()
    for i, entry in enumerate(raw):
        try:
            result.decisions.append(ScreeningDecision.model_validate(entry))
        except ValidationError as exc:
            record_id = entry.get("record_id", f"<index {i}, no record_id>") if isinstance(entry, dict) else f"<index {i}>"
            result.errors.append(f"{record_id}: {exc}")
    return result


@dataclass
class IngestSummary:
    ingested: int = 0
    skipped_unknown_record_id: list[str] = field(default_factory=list)
    decision_counts: dict[str, int] = field(default_factory=dict)


def ingest_decisions(
    engine: Engine, decisions: list[ScreeningDecision], coder: str, prompt_version: str
) -> IngestSummary:
    records_table = reflect_table(engine, "records")
    classifications_ai_table = reflect_table(engine, "classifications_ai")
    screening_decisions_table = reflect_table(engine, "screening_decisions")
    now = datetime.now(UTC).isoformat()

    with engine.connect() as conn:
        known_ids = {row[0] for row in conn.execute(select(records_table.c.record_id))}

    summary = IngestSummary()
    with engine.begin() as conn:
        for d in decisions:
            if d.record_id not in known_ids:
                summary.skipped_unknown_record_id.append(d.record_id)
                continue

            conn.execute(
                classifications_ai_table.insert().values(
                    classification_id=new_id(),
                    record_id=d.record_id,
                    dimension="topical_inclusion",
                    label=d.decision,
                    confidence=d.confidence,
                    rationale=d.rationale,
                    uncertainty_reason=d.uncertainty_reason,
                    coder=coder,
                    prompt_version=prompt_version,
                    timestamp=now,
                )
            )
            conn.execute(
                screening_decisions_table.insert().values(
                    screening_decision_id=new_id(),
                    record_id=d.record_id,
                    decision_type="ai",
                    decision=d.decision,
                    exclusion_reason=d.exclusion_reason,
                    confidence=d.confidence,
                    coder=coder,
                    timestamp=now,
                )
            )
            summary.ingested += 1
            summary.decision_counts[d.decision] = summary.decision_counts.get(d.decision, 0) + 1

    return summary
