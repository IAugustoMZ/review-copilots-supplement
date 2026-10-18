"""Phase 11: persisting a human reviewer's decision.

Reuses ScreeningDecision (src/screening/ai_screening.py) for the decision
shape -- same fields, same exclusion/uncertainty-reason validation -- since a
human decision and an AI decision are structurally the same claim about a
record, just from a different coder. Writes to classifications_human /
screening_decisions(decision_type='human'), never to the AI tables and never
directly to classifications_final (Phase 13 adjudication owns that).

save_human_decision() only records the topical_inclusion verdict -- it does
NOT touch reviewer_sessions (records_completed / current_record_id). An
'include' decision still has the 6-dimension classification step ahead of it
(src/validation/dimension_classification.py), so the record isn't actually
done yet; call finalize_record_review() once it truly is (immediately for a
non-include decision, or after the last dimension is answered).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Engine

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.screening.ai_screening import ScreeningDecision
from review_pipeline.src.validation.blinded_queue import set_current_record


def save_human_decision(
    engine: Engine, decision: ScreeningDecision, reviewer_id: str, sample_membership: str | None = None
) -> None:
    classifications_human = reflect_table(engine, "classifications_human")
    screening_decisions = reflect_table(engine, "screening_decisions")
    now = datetime.now(UTC).isoformat()

    with engine.begin() as conn:
        conn.execute(
            classifications_human.insert().values(
                classification_id=new_id(),
                record_id=decision.record_id,
                dimension="topical_inclusion",
                label=decision.decision,
                confidence=decision.confidence,
                rationale=decision.rationale,
                uncertainty_reason=decision.uncertainty_reason,
                coder=reviewer_id,
                sample_membership=sample_membership,
                timestamp=now,
            )
        )
        conn.execute(
            screening_decisions.insert().values(
                screening_decision_id=new_id(),
                record_id=decision.record_id,
                decision_type="human",
                decision=decision.decision,
                exclusion_reason=decision.exclusion_reason,
                confidence=decision.confidence,
                coder=reviewer_id,
                timestamp=now,
            )
        )


def finalize_record_review(engine: Engine, reviewer_id: str) -> None:
    """Marks a record as fully done for this reviewer: increments
    records_completed and clears current_record_id. Call this once, at the
    true end of a record's review -- right after a non-include decision, or
    after the last classification dimension is answered for an 'include'."""
    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        existing = conn.execute(
            reviewer_sessions.select().where(reviewer_sessions.c.reviewer_id == reviewer_id)
        ).first()
        if existing:
            conn.execute(
                reviewer_sessions.update()
                .where(reviewer_sessions.c.reviewer_id == reviewer_id)
                .values(
                    records_completed=existing.records_completed + 1,
                    current_record_id=None,
                    last_session_at=now,
                )
            )
        else:
            conn.execute(
                reviewer_sessions.insert().values(
                    reviewer_id=reviewer_id,
                    records_completed=1,
                    current_record_id=None,
                    last_session_at=now,
                )
            )


def skip_current_record(engine: Engine, reviewer_id: str) -> None:
    """Clears the reviewer's in-progress record without recording a decision --
    it stays in the queue for later (their own or another reviewer's session)."""
    set_current_record(engine, reviewer_id, None)
