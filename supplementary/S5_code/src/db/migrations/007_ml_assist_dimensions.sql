-- Extends ml_assist_suggestions to also log suggestions for the 6-dimension
-- classification step (domain, technology, application, architectural_role,
-- validation_focus, evidence_type), not just topical_inclusion. Existing rows
-- (all topical_inclusion suggestions) backfill to that value so `classify
-- ml-stats` can report override rates per dimension without a data migration.
ALTER TABLE ml_assist_suggestions ADD COLUMN dimension TEXT NOT NULL DEFAULT 'topical_inclusion';
