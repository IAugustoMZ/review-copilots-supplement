-- Initial schema for review_pipeline.
-- SQLite. Timestamps are ISO-8601 UTC strings (SQLite has no native datetime type).
-- This file is the single source of truth for the schema; SQLAlchemy accesses tables
-- via reflection (src/db/tables.py) rather than a parallel ORM model, so there is only
-- one place schema changes are made. See docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 2.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- records: the corpus. One row per distinct scholarly work (post-deduplication
-- non-canonical duplicates stay as rows with canonical_record_id set, never deleted).
-- ---------------------------------------------------------------------------
CREATE TABLE records (
    record_id            TEXT PRIMARY KEY,             -- ULID
    doi                  TEXT,
    title                TEXT NOT NULL,
    title_normalized     TEXT NOT NULL,
    year                 INTEGER,
    venue                TEXT,
    publication_type     TEXT,                          -- raw connector-reported type
    publication_status   TEXT NOT NULL DEFAULT 'unknown'
        CHECK (publication_status IN (
            'peer_reviewed_journal', 'peer_reviewed_conference',
            'book_chapter_peer_reviewed', 'preprint', 'technical_report',
            'standard', 'documentation', 'unknown'
        )),
    abstract              TEXT,
    abstract_source       TEXT,                          -- e.g. 'crossref', 'openalex'
    keywords              TEXT,                           -- JSON array
    publisher             TEXT,
    issn                  TEXT,
    citation_count         INTEGER,
    oa_status               TEXT,
    oa_url                  TEXT,
    fulltext_surrogate_url  TEXT,                          -- set only for HIGH-confidence arXiv links
    preprint_doi_or_id      TEXT,
    preprint_version         TEXT,
    canonical_record_id       TEXT REFERENCES records(record_id),  -- set when merged away
    corpus_type                TEXT NOT NULL DEFAULT 'primary'
        CHECK (corpus_type IN ('primary', 'snowball')),
    retrieval_source            TEXT,                       -- e.g. 'crossref', 'venue_search'
    query_id                     TEXT,
    retrieval_timestamp            TEXT,
    raw_source_id                   TEXT,
    raw_metadata_json                 TEXT,                  -- JSON, kept where practical
    d0                                 INTEGER NOT NULL DEFAULT 1,  -- boolean: in discovery pool
    d1                                 INTEGER NOT NULL DEFAULT 0,  -- boolean: eligible peer-reviewed corpus
    d2                                 INTEGER NOT NULL DEFAULT 0,  -- boolean: system evidence corpus
    d3                                 INTEGER NOT NULL DEFAULT 0,  -- boolean: verified evidence set
    core_or_contextual                 TEXT
        CHECK (core_or_contextual IN ('core', 'contextual') OR core_or_contextual IS NULL),
    d1_exception_rule_id                 TEXT,               -- documented-exception id, §8
    created_at                            TEXT NOT NULL,
    updated_at                            TEXT NOT NULL
);
CREATE INDEX idx_records_doi ON records(doi);
CREATE INDEX idx_records_title_normalized ON records(title_normalized);
CREATE INDEX idx_records_canonical ON records(canonical_record_id);

-- ---------------------------------------------------------------------------
-- search_runs / query_hits: every search execution, success or failure.
-- ---------------------------------------------------------------------------
CREATE TABLE search_runs (
    search_run_id     TEXT PRIMARY KEY,
    query_id          TEXT NOT NULL,
    query_text        TEXT NOT NULL,
    database          TEXT NOT NULL,               -- connector name, e.g. 'crossref'
    timestamp         TEXT NOT NULL,
    filters_json      TEXT,
    page_count        INTEGER,
    record_count      INTEGER,
    api_version       TEXT,
    status            TEXT NOT NULL
        CHECK (status IN ('ok', 'zero_results', 'error')),
    error_detail      TEXT,
    rate_limit_events INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_search_runs_query_id ON search_runs(query_id);

CREATE TABLE query_hits (
    search_run_id   TEXT NOT NULL REFERENCES search_runs(search_run_id),
    record_id       TEXT NOT NULL REFERENCES records(record_id),
    rank_in_results INTEGER,
    PRIMARY KEY (search_run_id, record_id)
);

-- ---------------------------------------------------------------------------
-- authors / record_authors
-- ---------------------------------------------------------------------------
CREATE TABLE authors (
    author_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    surname   TEXT NOT NULL,
    orcid     TEXT
);
CREATE INDEX idx_authors_surname ON authors(surname);

CREATE TABLE record_authors (
    record_id TEXT NOT NULL REFERENCES records(record_id),
    author_id TEXT NOT NULL REFERENCES authors(author_id),
    position  INTEGER NOT NULL,
    PRIMARY KEY (record_id, author_id)
);

-- ---------------------------------------------------------------------------
-- fulltext_locations: every resolution attempt, not just the winner.
-- ---------------------------------------------------------------------------
CREATE TABLE fulltext_locations (
    fulltext_location_id TEXT PRIMARY KEY,
    record_id             TEXT NOT NULL REFERENCES records(record_id),
    source                 TEXT NOT NULL,   -- publisher/unpaywall/openalex/semantic_scholar/
                                             -- institutional_repository/author_manuscript/pmc/arxiv
    url                    TEXT,
    license                TEXT,
    resolved_at             TEXT NOT NULL,
    accessible               INTEGER NOT NULL,  -- boolean
    access_method             TEXT
);
CREATE INDEX idx_fulltext_locations_record ON fulltext_locations(record_id);

-- ---------------------------------------------------------------------------
-- preprint_matches: arXiv <-> published linkage evidence (§9/§10 of the plan).
-- ---------------------------------------------------------------------------
CREATE TABLE preprint_matches (
    preprint_match_id TEXT PRIMARY KEY,
    journal_record_id  TEXT NOT NULL REFERENCES records(record_id),
    preprint_record_id  TEXT NOT NULL REFERENCES records(record_id),
    match_confidence      TEXT NOT NULL CHECK (match_confidence IN ('high', 'medium', 'low')),
    evidence_json          TEXT NOT NULL,  -- title_sim, author_overlap, year_proximity, doi_or_jref_match, abstract_sim
    status                   TEXT NOT NULL DEFAULT 'pending_human_review'
        CHECK (status IN ('auto_linked', 'pending_human_review', 'rejected')),
    created_at                TEXT NOT NULL,
    decided_at                 TEXT,
    decided_by                  TEXT
);

-- ---------------------------------------------------------------------------
-- duplicate_groups
-- ---------------------------------------------------------------------------
CREATE TABLE duplicate_groups (
    duplicate_group_id  TEXT PRIMARY KEY,
    canonical_record_id  TEXT NOT NULL REFERENCES records(record_id),
    merged_record_ids_json TEXT NOT NULL,
    merge_reason           TEXT NOT NULL
        CHECK (merge_reason IN ('doi_exact', 'title_exact', 'title_fuzzy', 'preprint_link')),
    merge_confidence         REAL,
    created_at                 TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- classifications_ai / classifications_human / classifications_final
-- Kept as three separate tables (not one table with a `source` column) so that a
-- query executed during the blinded human-review window structurally cannot join
-- human and AI labels. See plan section 15.
-- ---------------------------------------------------------------------------
CREATE TABLE classifications_ai (
    classification_id TEXT PRIMARY KEY,
    record_id          TEXT NOT NULL REFERENCES records(record_id),
    dimension            TEXT NOT NULL,       -- e.g. 'domain', 'technology'
    label                  TEXT NOT NULL,       -- JSON array if multi-label
    confidence               TEXT,
    rationale                 TEXT,
    uncertainty_reason          TEXT,
    coder                        TEXT NOT NULL,   -- model name, e.g. 'claude-sonnet-5'
    prompt_version                 TEXT,
    timestamp                        TEXT NOT NULL
);
CREATE INDEX idx_classifications_ai_record ON classifications_ai(record_id);

CREATE TABLE classifications_human (
    classification_id TEXT PRIMARY KEY,
    record_id          TEXT NOT NULL REFERENCES records(record_id),
    dimension            TEXT NOT NULL,
    label                  TEXT NOT NULL,
    confidence               TEXT CHECK (confidence IN ('high', 'medium', 'low') OR confidence IS NULL),
    rationale                 TEXT,
    uncertainty_reason          TEXT,
    coder                        TEXT NOT NULL,    -- reviewer id
    sample_membership              TEXT,             -- e.g. 'validation', 'random', NULL
    timestamp                        TEXT NOT NULL
);
CREATE INDEX idx_classifications_human_record ON classifications_human(record_id);

CREATE TABLE classifications_final (
    classification_id TEXT PRIMARY KEY,
    record_id          TEXT NOT NULL REFERENCES records(record_id),
    dimension            TEXT NOT NULL,
    label                  TEXT NOT NULL,
    confidence               TEXT,
    source_of_truth            TEXT NOT NULL
        CHECK (source_of_truth IN ('human', 'ai', 'adjudicated')),
    adjudicator                  TEXT,
    reason                         TEXT,
    supersedes_classification_id     TEXT REFERENCES classifications_final(classification_id),
    timestamp                          TEXT NOT NULL
);
CREATE INDEX idx_classifications_final_record ON classifications_final(record_id);

-- ---------------------------------------------------------------------------
-- screening decisions
-- ---------------------------------------------------------------------------
CREATE TABLE screening_decisions (
    screening_decision_id TEXT PRIMARY KEY,
    record_id               TEXT NOT NULL REFERENCES records(record_id),
    decision_type              TEXT NOT NULL CHECK (decision_type IN ('ai', 'human', 'final')),
    decision                     TEXT NOT NULL CHECK (decision IN ('include', 'exclude', 'uncertain')),
    exclusion_reason                TEXT,
    confidence                        TEXT,
    coder                                TEXT NOT NULL,
    timestamp                             TEXT NOT NULL
);
CREATE INDEX idx_screening_decisions_record ON screening_decisions(record_id);

-- ---------------------------------------------------------------------------
-- verified_evidence: D3 per-claim extraction notes
-- ---------------------------------------------------------------------------
CREATE TABLE verified_evidence (
    verified_evidence_id TEXT PRIMARY KEY,
    record_id              TEXT NOT NULL REFERENCES records(record_id),
    claim_id                  TEXT,
    verified_text_span          TEXT,
    evidence_source                TEXT NOT NULL
        CHECK (evidence_source IN ('verified_fulltext', 'oa_fulltext', 'abstract', 'metadata_only')),
    coding_confidence                TEXT NOT NULL CHECK (coding_confidence IN ('high', 'medium', 'low')),
    created_at                         TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- sentinel_records: known-item set
-- ---------------------------------------------------------------------------
CREATE TABLE sentinel_records (
    sentinel_id           TEXT PRIMARY KEY,
    doi                     TEXT,
    title                     TEXT NOT NULL,
    venue                      TEXT,
    year                         INTEGER,
    ce_theme                       TEXT,
    why_known_relevant                TEXT NOT NULL,
    matched_record_id                   TEXT REFERENCES records(record_id),
    retrieved_generic                     INTEGER NOT NULL DEFAULT 0,
    retrieved_ce_search                     INTEGER NOT NULL DEFAULT 0,
    retrieved_venue_search                    INTEGER NOT NULL DEFAULT 0,
    retrieval_sources_json                      TEXT,
    last_checked_at                               TEXT
);

-- ---------------------------------------------------------------------------
-- gap_searches: targeted absence-claim verification
-- ---------------------------------------------------------------------------
CREATE TABLE gap_searches (
    gap_id            TEXT PRIMARY KEY,
    claim              TEXT NOT NULL,
    queries_json         TEXT NOT NULL,
    sources_json            TEXT NOT NULL,
    date                       TEXT NOT NULL,
    candidate_count               INTEGER,
    records_reviewed                 INTEGER,
    counterexamples_json                TEXT,
    status                                TEXT NOT NULL DEFAULT 'uncertain'
        CHECK (status IN ('SUPPORTED_WITHIN_UPDATED_SEARCH', 'COUNTEREXAMPLE_FOUND', 'UNCERTAIN'))
);

-- ---------------------------------------------------------------------------
-- audit_events: append-only history of every mutating decision.
-- ---------------------------------------------------------------------------
CREATE TABLE audit_events (
    event_id     TEXT PRIMARY KEY,
    entity_type   TEXT NOT NULL,      -- e.g. 'classifications_final', 'screening_decisions'
    entity_id       TEXT NOT NULL,
    actor             TEXT NOT NULL,    -- 'ai' | 'human:<name>' | 'system'
    action              TEXT NOT NULL,
    before_json           TEXT,
    after_json              TEXT,
    timestamp                 TEXT NOT NULL
);
CREATE INDEX idx_audit_events_entity ON audit_events(entity_type, entity_id);

-- ---------------------------------------------------------------------------
-- reviewer_sessions: resumable human-coding progress (§15/§18 of the plan).
-- ---------------------------------------------------------------------------
CREATE TABLE reviewer_sessions (
    reviewer_id         TEXT PRIMARY KEY,
    records_completed     INTEGER NOT NULL DEFAULT 0,
    current_record_id       TEXT REFERENCES records(record_id),
    last_session_at            TEXT
);

-- ---------------------------------------------------------------------------
-- corpus_snapshots: Phase 16 freeze points referenced by analysis (§21).
-- ---------------------------------------------------------------------------
CREATE TABLE corpus_snapshots (
    snapshot_id     TEXT PRIMARY KEY,
    label             TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    record_ids_json        TEXT NOT NULL,  -- frozen list of record_ids in this snapshot
    notes                     TEXT
);

-- ---------------------------------------------------------------------------
-- documented_exceptions: D1-membership exceptions for non-peer-reviewed records (§8).
-- ---------------------------------------------------------------------------
CREATE TABLE documented_exceptions (
    exception_id TEXT PRIMARY KEY,
    record_id      TEXT NOT NULL REFERENCES records(record_id),
    rule_text        TEXT NOT NULL,
    justification       TEXT NOT NULL,
    approved_by            TEXT NOT NULL,
    created_at                TEXT NOT NULL
);
