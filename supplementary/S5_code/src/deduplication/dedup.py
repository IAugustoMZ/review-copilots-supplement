"""Phase 8 orchestration: load all non-canonical records, find duplicate/
preprint-published groups, and persist merges + the audit trail
(duplicate_groups, preprint_matches, records.canonical_record_id).

Merge cascade (docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 7):
  1. DOI exact (normalized)                         -> auto-merge, confidence 1.0
  2. Title exact (normalized)                        -> auto-merge, confidence 1.0
  3. Fuzzy title + (year proximity OR author overlap) -> auto-merge, confidence = fuzzy score
  4. Preprint <-> peer-reviewed linkage:
       high confidence   -> auto-merge (also logged to preprint_matches, status=auto_linked)
       medium/low         -> logged to preprint_matches only, NOT merged
                             (status=pending_human_review) -- never silently
                             conflate two possibly-distinct works.

Every merge is logged to duplicate_groups with merged_record_ids_json,
merge_reason (the strongest reason among the group's edges), and
merge_confidence. Non-canonical rows are never deleted, only marked via
canonical_record_id, per the `records` table's own documented invariant.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rapidfuzz import fuzz, process
from sqlalchemy import Engine, bindparam, select

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.deduplication.canonical import pick_canonical_id
from review_pipeline.src.deduplication.disjoint_set import DisjointSet
from review_pipeline.src.deduplication.matching import (
    FUZZY_TITLE_THRESHOLD,
    PREPRINT_LOW_THRESHOLD,
    DedupRecord,
    authors_overlap,
    preprint_published_confidence,
    title_first_token,
    years_close,
)

# Strongest-first, used to pick the group's headline merge_reason when a group
# was formed by more than one kind of edge.
_REASON_STRENGTH = {"doi_exact": 0, "title_exact": 1, "preprint_link": 2, "title_fuzzy": 3}


@dataclass
class MergeEdge:
    record_a: str
    record_b: str
    reason: str
    confidence: float


@dataclass
class PreprintMatchOutcome:
    preprint_record_id: str
    journal_record_id: str
    confidence: str  # high/medium/low
    evidence: dict
    merged: bool


@dataclass
class DedupSummary:
    records_considered: int = 0
    duplicate_groups_created: int = 0
    records_merged_away: int = 0
    preprint_matches_logged: int = 0
    edges: list[MergeEdge] = field(default_factory=list)
    preprint_outcomes: list[PreprintMatchOutcome] = field(default_factory=list)


def _load_dedup_records(engine: Engine) -> tuple[list[DedupRecord], dict[str, dict]]:
    records_table = reflect_table(engine, "records")
    authors_table = reflect_table(engine, "authors")
    record_authors_table = reflect_table(engine, "record_authors")

    with engine.connect() as conn:
        # Every column except the provenance blobs: raw_metadata_json alone is
        # ~320MB across the corpus and nothing in matching or canonical
        # selection reads it, but materialising it for every row is most of
        # what made this function's footprint fatal.
        wanted = [c for c in records_table.c if c.name not in ("raw_metadata_json", "field_provenance_json")]
        rows = [
            dict(r)
            for r in conn.execute(
                select(*wanted).where(records_table.c.canonical_record_id.is_(None))
            )
            .mappings()
            .all()
        ]
        author_rows = conn.execute(
            record_authors_table.join(authors_table, record_authors_table.c.author_id == authors_table.c.author_id).select()
        ).mappings().all()

    surnames_by_record: dict[str, list[str]] = defaultdict(list)
    for ar in author_rows:
        surnames_by_record[ar["record_id"]].append(ar["surname"])

    row_by_id = {r["record_id"]: r for r in rows}
    dedup_records = [
        DedupRecord(
            record_id=r["record_id"],
            doi=r["doi"],
            title=r["title"],
            title_normalized=r["title_normalized"],
            year=r["year"],
            publication_status=r["publication_status"],
            author_surnames=tuple(surnames_by_record.get(r["record_id"], [])),
        )
        for r in rows
    ]
    return dedup_records, row_by_id


# Step C compares two records iff their year buckets overlap, i.e. iff their
# years differ by at most 2 (year_bucket_keys() puts each record in year-1,
# year, year+1) or both years are unknown. Generating those pairs directly,
# once each, replaces the set of already-seen pairs the earlier implementation
# kept: at 22k records that set holds >10^8 frozensets, which OOM-killed the
# process at 12.4GB of RSS on 2026-08-29 and took the editor down with it.
_YEAR_BLOCK_SPAN = 2
# Query rows per cdist call. The score matrix is len(chunk) x len(right)
# float32, so this caps a block at a few MB regardless of corpus size.
_CDIST_CHUNK = 256


def _score_block(
    left: list[DedupRecord], right: list[DedupRecord], threshold: float, same_block: bool
):
    """Yield (a, b, score) for pairs scoring >= threshold.

    rapidfuzz's cdist runs the same token_sort_ratio as fuzzy_title_score, in
    C++ across threads, and zeroes everything below the cutoff -- so Python
    only ever sees the handful of pairs that can match. Scoring semantics and
    the threshold are unchanged; only the loop moved out of Python.
    """
    if not left or not right:
        return
    titles_left = [r.title_normalized for r in left]
    titles_right = [r.title_normalized for r in right]
    for start in range(0, len(left), _CDIST_CHUNK):
        chunk = titles_left[start : start + _CDIST_CHUNK]
        scores = process.cdist(
            chunk, titles_right, scorer=fuzz.token_sort_ratio, score_cutoff=threshold * 100, workers=-1
        )
        for i, j in zip(*scores.nonzero(), strict=True):
            left_index = start + int(i)
            # Within one block every pair appears twice in the matrix; keep the
            # upper triangle so each is yielded once and never against itself.
            if same_block and int(j) <= left_index:
                continue
            yield left[left_index], right[int(j)], float(scores[int(i), int(j)]) / 100.0


def _fuzzy_candidate_pairs(dedup_records: list[DedupRecord], threshold: float):
    """Year-blocked title-similarity candidates, each pair yielded once."""
    # An empty normalized title scores 0.0 by fuzzy_title_score's own guard;
    # cdist has no such guard, so drop those records before scoring.
    titled = [r for r in dedup_records if r.title_normalized]
    by_year: dict[int | None, list[DedupRecord]] = defaultdict(list)
    for r in titled:
        by_year[r.year].append(r)

    known_years = sorted(y for y in by_year if y is not None)
    for year in known_years:
        left = by_year[year]
        yield from _score_block(left, left, threshold, same_block=True)
        neighbours = [r for offset in range(1, _YEAR_BLOCK_SPAN + 1) for r in by_year.get(year + offset, [])]
        yield from _score_block(left, neighbours, threshold, same_block=False)

    unknown = by_year.get(None, [])
    yield from _score_block(unknown, unknown, threshold, same_block=True)


def _find_pairwise_edges(dedup_records: list[DedupRecord]) -> tuple[list[MergeEdge], list[PreprintMatchOutcome]]:
    edges: list[MergeEdge] = []
    preprint_outcomes: list[PreprintMatchOutcome] = []
    # Tracks connectivity as edges are added, so Steps A-C never re-add a pair
    # that is already in one component.
    ds = DisjointSet([r.record_id for r in dedup_records])

    # --- Step A: DOI exact (O(n), no blocking needed -- DOI equality doesn't
    # depend on year/title matching at all).
    doi_groups: dict[str, list[str]] = defaultdict(list)
    for r in dedup_records:
        if r.doi:
            doi_groups[r.doi].append(r.record_id)
    for ids in doi_groups.values():
        for other in ids[1:]:
            ds.union(ids[0], other)
            edges.append(MergeEdge(ids[0], other, "doi_exact", 1.0))

    # --- Step B: title exact (normalized), O(n).
    title_groups: dict[str, list[str]] = defaultdict(list)
    for r in dedup_records:
        if r.title_normalized:
            title_groups[r.title_normalized].append(r.record_id)
    for ids in title_groups.values():
        for other in ids[1:]:
            if ds.find(ids[0]) == ds.find(other):
                continue
            ds.union(ids[0], other)
            edges.append(MergeEdge(ids[0], other, "title_exact", 1.0))

    # --- Step C: year-blocked fuzzy title match, same-vs-same publication
    # status only (ordinary near-duplicates from different connectors share a
    # year; a preprint and its eventual publication routinely do not, so
    # cross-status pairs are handled separately in Step D with different
    # blocking -- see title_first_token()).
    #
    # Pairs already connected by Step A or B are skipped through the union-find
    # rather than through a set of seen pairs. That is equivalent for grouping
    # (they are already one component) and for the recorded merge_reason
    # (title_fuzzy is the weakest reason, so it never wins a group that also
    # carries a doi_exact or title_exact edge).
    for a, b, score in _fuzzy_candidate_pairs(dedup_records, FUZZY_TITLE_THRESHOLD):
        if (a.publication_status == "preprint") != (b.publication_status == "preprint"):
            continue
        if ds.find(a.record_id) == ds.find(b.record_id):
            continue
        if not (years_close(a.year, b.year) or authors_overlap(a.author_surnames, b.author_surnames)):
            continue
        ds.union(a.record_id, b.record_id)
        edges.append(MergeEdge(a.record_id, b.record_id, "title_fuzzy", score))

    # --- Step D: preprint <-> peer-reviewed linkage, blocked by shared first
    # title token (NOT year -- a preprint can predate its publication by years).
    # Each record has exactly one first token, so a pair can only arise in one
    # bucket and no cross-bucket bookkeeping is needed.
    prefix_buckets: dict[str, list[DedupRecord]] = defaultdict(list)
    for r in dedup_records:
        token = title_first_token(r.title_normalized)
        if token:
            prefix_buckets[token].append(r)

    for bucket in prefix_buckets.values():
        preprints = [r for r in bucket if r.publication_status == "preprint"]
        published = [r for r in bucket if r.publication_status != "preprint"]
        # Below the low threshold preprint_published_confidence returns None,
        # so cutting there drops only pairs that were never going to be logged.
        for preprint_rec, journal_rec, _score in _score_block(
            preprints, published, PREPRINT_LOW_THRESHOLD, same_block=False
        ):
            confidence, evidence = preprint_published_confidence(preprint_rec, journal_rec)
            if confidence is None:
                continue
            merged = confidence == "high"
            if merged:
                edges.append(
                    MergeEdge(preprint_rec.record_id, journal_rec.record_id, "preprint_link", evidence["title_similarity"])
                )
            preprint_outcomes.append(
                PreprintMatchOutcome(
                    preprint_record_id=preprint_rec.record_id,
                    journal_record_id=journal_rec.record_id,
                    confidence=confidence,
                    evidence=evidence,
                    merged=merged,
                )
            )

    return edges, preprint_outcomes


def _backfill_earliest_seen_iteration(engine: Engine) -> None:
    """For every canonical record, sets earliest_seen_iteration = the minimum
    discovery_iteration across itself and every record merged into it (current
    duplicate-group membership, read from records.canonical_record_id -- not
    the in-memory union-find, so this stays correct across repeated dedup runs
    over multiple search iterations, not just the run that just executed).
    A record found again by a later iteration's search must still count
    toward the iteration that found it FIRST, or the saturation curve would
    overstate later iterations' real contribution."""
    records_table = reflect_table(engine, "records")
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                records_table.c.record_id,
                records_table.c.canonical_record_id,
                records_table.c.discovery_iteration,
            )
        ).all()

    # One pass over the membership instead of a query per canonical record:
    # at 22k canonical records the per-record form issued 22k round trips.
    earliest_by_canonical: dict[str, int] = {}
    for record_id, canonical_record_id, iteration in rows:
        if iteration is None:
            continue
        group_id = canonical_record_id or record_id
        current = earliest_by_canonical.get(group_id)
        if current is None or iteration < current:
            earliest_by_canonical[group_id] = iteration

    if not earliest_by_canonical:
        return
    with engine.begin() as conn:
        conn.execute(
            records_table.update()
            .where(records_table.c.record_id == bindparam("target_record_id"))
            .values(earliest_seen_iteration=bindparam("earliest")),
            [{"target_record_id": rid, "earliest": it} for rid, it in earliest_by_canonical.items()],
        )


def run_deduplication(engine: Engine, run_logger: RunLogger) -> DedupSummary:
    dedup_records, row_by_id = _load_dedup_records(engine)
    summary = DedupSummary(records_considered=len(dedup_records))

    edges, preprint_outcomes = _find_pairwise_edges(dedup_records)
    summary.edges = edges
    summary.preprint_outcomes = preprint_outcomes

    ds = DisjointSet([r.record_id for r in dedup_records])
    for edge in edges:
        ds.union(edge.record_a, edge.record_b)

    edges_by_group: dict[str, list[MergeEdge]] = defaultdict(list)
    for edge in edges:
        edges_by_group[ds.find(edge.record_a)].append(edge)

    records_table = reflect_table(engine, "records")
    duplicate_groups_table = reflect_table(engine, "duplicate_groups")
    preprint_matches_table = reflect_table(engine, "preprint_matches")
    now = datetime.now(UTC).isoformat()

    for root, members in ds.groups().items():
        if len(members) < 2:
            continue
        member_rows = [row_by_id[m] for m in members]
        canonical_id = pick_canonical_id(member_rows)
        group_edges = edges_by_group.get(root, [])
        best_reason = min((e.reason for e in group_edges), key=lambda r: _REASON_STRENGTH[r], default="title_fuzzy")
        confidences = [e.confidence for e in group_edges if e.reason == best_reason] or [1.0]
        merge_confidence = min(confidences)

        with engine.begin() as conn:
            conn.execute(
                duplicate_groups_table.insert().values(
                    duplicate_group_id=new_id(),
                    canonical_record_id=canonical_id,
                    merged_record_ids_json=json.dumps(sorted(members)),
                    merge_reason=best_reason,
                    merge_confidence=merge_confidence,
                    created_at=now,
                )
            )
            for member_id in members:
                if member_id == canonical_id:
                    continue
                conn.execute(
                    records_table.update()
                    .where(records_table.c.record_id == member_id)
                    .values(canonical_record_id=canonical_id, updated_at=now)
                )
        summary.duplicate_groups_created += 1
        summary.records_merged_away += len(members) - 1
        run_logger.log_event(
            event="duplicate_group_merged",
            canonical_record_id=canonical_id,
            merged_record_ids=sorted(members),
            merge_reason=best_reason,
            merge_confidence=merge_confidence,
        )

    with engine.begin() as conn:
        for outcome in preprint_outcomes:
            conn.execute(
                preprint_matches_table.insert().values(
                    preprint_match_id=new_id(),
                    journal_record_id=outcome.journal_record_id,
                    preprint_record_id=outcome.preprint_record_id,
                    match_confidence=outcome.confidence,
                    evidence_json=json.dumps(outcome.evidence),
                    status="auto_linked" if outcome.merged else "pending_human_review",
                    created_at=now,
                    decided_at=now if outcome.merged else None,
                    decided_by="dedup_pipeline_phase8" if outcome.merged else None,
                )
            )
    summary.preprint_matches_logged = len(preprint_outcomes)
    if preprint_outcomes:
        run_logger.log_event(
            event="preprint_matches_logged",
            count=len(preprint_outcomes),
            auto_linked=sum(1 for o in preprint_outcomes if o.merged),
            pending_human_review=sum(1 for o in preprint_outcomes if not o.merged),
        )

    _backfill_earliest_seen_iteration(engine)

    return summary
