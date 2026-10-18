"""Phase 13: adjudication.

docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 19: queues records where AI and
human disagree, or either is low-confidence. Every action -- manual or the
mechanical auto-finalize path below -- INSERTS a new classifications_final
row plus an audit_events row; nothing is ever updated in place, so the full
label history is reconstructable by replaying audit_events for a record.

Two paths write classifications_final:
  1. apply_adjudication() -- a human adjudicator's explicit resolution of a
     disagreement/uncertainty (accept_human/accept_ai/new_value/unresolved/
     request_fulltext).
  2. auto_finalize_agreements() -- a purely mechanical rule for the common
     case where AI and human already agree at reasonable confidence, so a
     human adjudicator doesn't need to look at every single record, only the
     disagreements. Still fully audited (actor='system:auto_agreement' in
     audit_events), and only ever chooses source_of_truth='human' (human
     review is the authoritative check on AI in this pipeline's design, not
     the reverse) -- never silently promotes an AI-only label to final.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import Engine

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.validation.agreement import paired_decisions

AdjudicationAction = Literal["accept_human", "accept_ai", "new_value", "unresolved", "request_fulltext"]


@dataclass
class AdjudicationCandidate:
    record_id: str
    dimension: str
    ai_label: str
    ai_confidence: str | None
    ai_rationale: str | None
    human_label: str
    human_confidence: str | None
    human_rationale: str | None


def _latest_classification(engine: Engine, table_name: str, record_id: str, dimension: str) -> dict | None:
    table = reflect_table(engine, table_name)
    with engine.connect() as conn:
        rows = conn.execute(
            table.select()
            .where(table.c.record_id == record_id, table.c.dimension == dimension)
            .order_by(table.c.timestamp.desc())
        ).mappings().all()
    return dict(rows[0]) if rows else None


def queue_for_adjudication(engine: Engine, dimension: str = "topical_inclusion") -> list[AdjudicationCandidate]:
    """Paired AI/human records that disagree, or where either is low-confidence
    -- and that don't already have a classifications_final row for this
    dimension. Without that second check, an already-resolved record would
    reappear in the queue forever (queue membership is derived from the raw
    AI/human rows, which are never modified by adjudication -- that's
    intentional, for the audit trail, but means "already resolved" has to be
    checked separately)."""
    candidates = []
    for pair in paired_decisions(engine, dimension):
        if _latest_classification(engine, "classifications_final", pair.record_id, dimension) is not None:
            continue
        ai_row = _latest_classification(engine, "classifications_ai", pair.record_id, dimension)
        human_row = _latest_classification(engine, "classifications_human", pair.record_id, dimension)
        if ai_row is None or human_row is None:
            continue
        needs_adjudication = (
            pair.ai_label != pair.human_label
            or ai_row.get("confidence") == "low"
            or human_row.get("confidence") == "low"
        )
        if needs_adjudication:
            candidates.append(
                AdjudicationCandidate(
                    record_id=pair.record_id,
                    dimension=dimension,
                    ai_label=pair.ai_label,
                    ai_confidence=ai_row.get("confidence"),
                    ai_rationale=ai_row.get("rationale"),
                    human_label=pair.human_label,
                    human_confidence=human_row.get("confidence"),
                    human_rationale=human_row.get("rationale"),
                )
            )
    return candidates


def _write_final(
    engine: Engine,
    record_id: str,
    dimension: str,
    label: str,
    confidence: str | None,
    source_of_truth: str,
    adjudicator: str,
    reason: str,
    before: dict,
    action: str,
) -> str:
    classifications_final = reflect_table(engine, "classifications_final")
    screening_decisions = reflect_table(engine, "screening_decisions")
    audit_events = reflect_table(engine, "audit_events")
    now = datetime.now(UTC).isoformat()

    previous = _latest_classification(engine, "classifications_final", record_id, dimension)
    classification_id = new_id()

    with engine.begin() as conn:
        conn.execute(
            classifications_final.insert().values(
                classification_id=classification_id,
                record_id=record_id,
                dimension=dimension,
                label=label,
                confidence=confidence,
                source_of_truth=source_of_truth,
                adjudicator=adjudicator,
                reason=reason,
                supersedes_classification_id=previous["classification_id"] if previous else None,
                timestamp=now,
            )
        )
        if dimension == "topical_inclusion" and label in ("include", "exclude", "uncertain"):
            conn.execute(
                screening_decisions.insert().values(
                    screening_decision_id=new_id(),
                    record_id=record_id,
                    decision_type="final",
                    decision=label,
                    exclusion_reason=None,
                    confidence=confidence,
                    coder=adjudicator,
                    timestamp=now,
                )
            )
        conn.execute(
            audit_events.insert().values(
                event_id=new_id(),
                entity_type="classifications_final",
                entity_id=classification_id,
                actor=adjudicator,
                action=action,
                before_json=json.dumps(before),
                after_json=json.dumps({"label": label, "confidence": confidence, "source_of_truth": source_of_truth}),
                timestamp=now,
            )
        )
    return classification_id


def apply_adjudication(
    engine: Engine,
    record_id: str,
    dimension: str,
    action: AdjudicationAction,
    adjudicator: str,
    reason: str,
    new_decision: str | None = None,
    new_confidence: str | None = None,
) -> str:
    ai_row = _latest_classification(engine, "classifications_ai", record_id, dimension)
    human_row = _latest_classification(engine, "classifications_human", record_id, dimension)
    if ai_row is None or human_row is None:
        raise ValueError(f"record {record_id} has no paired AI+human decision for dimension {dimension!r}")

    before = {
        "ai": {"label": ai_row["label"], "confidence": ai_row.get("confidence")},
        "human": {"label": human_row["label"], "confidence": human_row.get("confidence")},
    }

    if action == "accept_human":
        label, confidence, source = human_row["label"], human_row.get("confidence"), "adjudicated"
    elif action == "accept_ai":
        label, confidence, source = ai_row["label"], ai_row.get("confidence"), "adjudicated"
    elif action == "new_value":
        if new_decision is None:
            raise ValueError("action='new_value' requires new_decision")
        label, confidence, source = new_decision, new_confidence, "adjudicated"
    elif action == "unresolved":
        label, confidence, source = "uncertain", "low", "adjudicated"
    elif action == "request_fulltext":
        label, confidence, source = "uncertain", "low", "adjudicated"
        reason = f"{reason} [awaiting full-text verification]"
    else:  # pragma: no cover -- Literal type keeps this unreachable in practice
        raise ValueError(f"unknown action: {action}")

    return _write_final(
        engine, record_id, dimension, label, confidence, source, adjudicator, reason, before, action=f"adjudicate:{action}"
    )


@dataclass
class AutoFinalizeSummary:
    finalized: int = 0
    skipped_already_final: int = 0


def auto_finalize_agreements(
    engine: Engine, dimension: str = "topical_inclusion", adjudicator: str = "system:auto_agreement"
) -> AutoFinalizeSummary:
    """Mechanically finalizes the common case: AI and human already agree at
    non-low confidence. Never runs on a record that already has a
    classifications_final row for this dimension (idempotent)."""
    summary = AutoFinalizeSummary()
    for pair in paired_decisions(engine, dimension):
        if _latest_classification(engine, "classifications_final", pair.record_id, dimension) is not None:
            summary.skipped_already_final += 1
            continue

        ai_row = _latest_classification(engine, "classifications_ai", pair.record_id, dimension)
        human_row = _latest_classification(engine, "classifications_human", pair.record_id, dimension)
        if ai_row is None or human_row is None:
            continue
        agrees = pair.ai_label == pair.human_label
        confident = ai_row.get("confidence") != "low" and human_row.get("confidence") != "low"
        if not (agrees and confident):
            continue

        before = {
            "ai": {"label": ai_row["label"], "confidence": ai_row.get("confidence")},
            "human": {"label": human_row["label"], "confidence": human_row.get("confidence")},
        }
        _write_final(
            engine,
            pair.record_id,
            dimension,
            human_row["label"],
            human_row.get("confidence"),
            "human",
            adjudicator,
            "AI and human agreed at non-low confidence; auto-finalized without manual adjudication.",
            before,
            action="auto_finalize_agreement",
        )
        summary.finalized += 1
    return summary
