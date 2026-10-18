-- Phase 14: arXiv-as-fulltext-access-surrogate candidates.
--
-- Deliberately NOT reusing preprint_matches: that table's preprint_record_id
-- is NOT NULL with an FK to records(record_id), which is correct for its own
-- job (Phase 8 corpus dedup, where both sides are real, already-discovered
-- records). An arXiv candidate found here via Phase 14's fallback search is
-- not a corpus record and must never become one just to satisfy that FK --
-- doing so would silently inflate D0/search counts with an item nobody
-- actually searched for. This table holds the candidate's own identity
-- directly instead.
CREATE TABLE fulltext_surrogate_candidates (
    candidate_id    TEXT PRIMARY KEY,
    record_id        TEXT NOT NULL REFERENCES records(record_id),
    arxiv_id          TEXT NOT NULL,
    arxiv_title        TEXT NOT NULL,
    arxiv_pdf_url        TEXT,
    match_confidence      TEXT NOT NULL CHECK (match_confidence IN ('high', 'medium', 'low')),
    evidence_json            TEXT NOT NULL,  -- title_sim, author_overlap, year_proximity, doi_or_jref_match
    status                     TEXT NOT NULL DEFAULT 'pending_human_review'
        CHECK (status IN ('auto_linked', 'pending_human_review', 'rejected')),
    created_at                   TEXT NOT NULL,
    decided_at                     TEXT,
    decided_by                       TEXT
);
CREATE INDEX idx_fulltext_surrogate_candidates_record ON fulltext_surrogate_candidates(record_id);
