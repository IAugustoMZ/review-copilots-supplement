"""Phase 14, second half: arXiv-as-fulltext-access-surrogate matching.

Only ever called after resolve_fulltext() has exhausted every other OA
source and found nothing accessible -- this resolves an ACCESS problem for
an already-known, already-peer-reviewed record, never a peer-review problem
and never a new discovery path (see connectors/arxiv.py's docstring).

Confidence rule, exactly as specced (docs/TECHNICAL_IMPLEMENTATION_PLAN.md
section 10): HIGH if doi_or_jref_match OR (title_sim >= 0.97 AND
author_overlap >= 0.5); MEDIUM if title_sim >= 0.90 AND author_overlap >=
0.25; LOW if title_sim >= 0.50 (keyword-level similarity only); otherwise
not a plausible candidate at all (None). Only HIGH auto-sets
records.fulltext_surrogate_url; MEDIUM is logged to
fulltext_surrogate_candidates as pending_human_review; LOW is logged but
never linked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from sqlalchemy import Engine

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_doi, normalize_title
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.deduplication.matching import fuzzy_title_score, years_close
from review_pipeline.src.search.connectors.arxiv import ArxivCandidate, search_arxiv_by_title

HIGH_TITLE_SIM = 0.97
HIGH_AUTHOR_OVERLAP = 0.5
MEDIUM_TITLE_SIM = 0.90
MEDIUM_AUTHOR_OVERLAP = 0.25
LOW_TITLE_SIM = 0.50


@dataclass
class MatchEvidence:
    title_sim: float
    author_overlap: float
    year_proximity: bool
    doi_or_jref_match: bool


def _author_overlap(record_surnames: list[str], candidate_surnames: list[str]) -> float:
    if not record_surnames or not candidate_surnames:
        return 0.0
    a = {s.lower() for s in record_surnames}
    b = {s.lower() for s in candidate_surnames}
    return len(a & b) / min(len(a), len(b))


def _doi_or_jref_match(record_doi: str | None, candidate: ArxivCandidate) -> bool:
    record_doi_norm = normalize_doi(record_doi)
    if record_doi_norm and candidate.doi and normalize_doi(candidate.doi) == record_doi_norm:
        return True
    return bool(record_doi_norm and candidate.journal_ref and record_doi_norm in candidate.journal_ref.lower())


def score_match(
    record_title: str, record_year: int | None, record_doi: str | None, record_author_surnames: list[str], candidate: ArxivCandidate
) -> tuple[str | None, MatchEvidence]:
    title_sim = fuzzy_title_score(normalize_title(record_title), normalize_title(candidate.title))
    author_overlap = _author_overlap(record_author_surnames, candidate.author_surnames)
    year_proximity = years_close(record_year, candidate.year)
    doi_or_jref_match = _doi_or_jref_match(record_doi, candidate)
    evidence = MatchEvidence(
        title_sim=title_sim, author_overlap=author_overlap, year_proximity=year_proximity, doi_or_jref_match=doi_or_jref_match
    )

    if doi_or_jref_match or (title_sim >= HIGH_TITLE_SIM and author_overlap >= HIGH_AUTHOR_OVERLAP):
        return "high", evidence
    if title_sim >= MEDIUM_TITLE_SIM and author_overlap >= MEDIUM_AUTHOR_OVERLAP:
        return "medium", evidence
    if title_sim >= LOW_TITLE_SIM:
        return "low", evidence
    return None, evidence


@dataclass
class ArxivFallbackResult:
    record_id: str
    candidate: ArxivCandidate | None
    confidence: str | None
    evidence: MatchEvidence | None
    auto_linked: bool


def find_arxiv_fallback(engine: Engine, record_id: str, client: httpx.Client) -> ArxivFallbackResult:
    records_table = reflect_table(engine, "records")
    authors_table = reflect_table(engine, "authors")
    record_authors_table = reflect_table(engine, "record_authors")
    candidates_table = reflect_table(engine, "fulltext_surrogate_candidates")

    with engine.connect() as conn:
        record = dict(conn.execute(records_table.select().where(records_table.c.record_id == record_id)).mappings().one())
        author_rows = conn.execute(
            record_authors_table.join(authors_table, record_authors_table.c.author_id == authors_table.c.author_id)
            .select()
            .where(record_authors_table.c.record_id == record_id)
        ).mappings().all()
    record_surnames = [a["surname"] for a in author_rows]

    arxiv_candidates = search_arxiv_by_title(client, record["title"])
    scored = [(*score_match(record["title"], record["year"], record["doi"], record_surnames, c), c) for c in arxiv_candidates]
    scored = [(conf, ev, c) for conf, ev, c in scored if conf is not None]
    if not scored:
        return ArxivFallbackResult(record_id=record_id, candidate=None, confidence=None, evidence=None, auto_linked=False)

    rank = {"high": 0, "medium": 1, "low": 2}
    best_conf, best_evidence, best_candidate = min(scored, key=lambda t: (rank[t[0]], -t[1].title_sim))

    auto_linked = best_conf == "high"
    now = datetime.now(UTC).isoformat()

    with engine.begin() as conn:
        conn.execute(
            candidates_table.insert().values(
                candidate_id=new_id(),
                record_id=record_id,
                arxiv_id=best_candidate.arxiv_id,
                arxiv_title=best_candidate.title,
                arxiv_pdf_url=best_candidate.pdf_url,
                match_confidence=best_conf,
                evidence_json=json.dumps(vars(best_evidence)),
                status="auto_linked" if auto_linked else "pending_human_review",
                created_at=now,
                decided_at=now if auto_linked else None,
                decided_by="arxiv_matching_phase14" if auto_linked else None,
            )
        )
        if auto_linked:
            conn.execute(
                records_table.update()
                .where(records_table.c.record_id == record_id)
                .values(
                    fulltext_surrogate_url=best_candidate.pdf_url,
                    preprint_doi_or_id=best_candidate.arxiv_id,
                    preprint_version=None,
                    updated_at=now,
                )
            )

    return ArxivFallbackResult(
        record_id=record_id, candidate=best_candidate, confidence=best_conf, evidence=best_evidence, auto_linked=auto_linked
    )
