-- Iteration tracking: tags every record with the discovery iteration (search
-- wave) it was first found in, and lets canonical (post-dedup) records carry
-- the earliest iteration any of their duplicate-group members were found in.
-- Powers the search-saturation curve: cumulative unique papers vs. iteration,
-- and the "elbow" where new iterations stop contributing a substantial
-- number of new unique papers (src/analysis/saturation.py).

ALTER TABLE records ADD COLUMN discovery_iteration INTEGER NOT NULL DEFAULT 1;
ALTER TABLE records ADD COLUMN earliest_seen_iteration INTEGER;

CREATE TABLE search_iterations (
    iteration_number INTEGER PRIMARY KEY,
    label             TEXT NOT NULL,
    description       TEXT,
    connectors_used   TEXT,               -- comma-separated connector names
    started_at        TEXT NOT NULL,
    notes             TEXT
);
