-- Phase 7 (Enrichment): track which source produced each mergeable field on
-- `records`, so the enricher can apply source-priority rules and never
-- silently overwrite higher-quality metadata with lower-quality metadata.
-- See docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 8.

ALTER TABLE records ADD COLUMN field_provenance_json TEXT;
