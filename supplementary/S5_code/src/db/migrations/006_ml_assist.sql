-- Phase 11 semi-supervised assist: logs every ML-suggested topical_inclusion
-- decision shown to a reviewer in `classify assisted`, plus what the reviewer
-- actually decided, so override/agreement rates over time are auditable --
-- the evidence needed to judge whether this is solid enough to extend from
-- the required-review set (596 records) to the full D1 pool (1,839).
-- Deliberately independent of classifications_ai: this table's suggestions
-- come from a model trained only on classifications_human, never on the
-- Phase 10 AI screening labels (see src/validation/ml_assist.py).
CREATE TABLE ml_assist_suggestions (
    suggestion_id TEXT PRIMARY KEY,
    record_id            TEXT NOT NULL REFERENCES records(record_id),
    reviewer_id             TEXT NOT NULL,
    suggested_label           TEXT NOT NULL,
    suggested_probability       REAL NOT NULL,
    training_set_size             INTEGER NOT NULL,
    final_label                     TEXT NOT NULL,
    was_override                     INTEGER NOT NULL CHECK (was_override IN (0, 1)),
    timestamp                          TEXT NOT NULL
);
CREATE INDEX idx_ml_assist_suggestions_record ON ml_assist_suggestions(record_id);
