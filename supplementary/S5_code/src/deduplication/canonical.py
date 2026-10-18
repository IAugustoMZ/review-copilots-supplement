"""Canonical-record selection within a duplicate group.

Prefers the most authoritative publication_status, then the most complete
metadata, so the merged record downstream synthesis reads from is the best
available version -- never an arbitrary "first found" pick.
"""

from __future__ import annotations

from review_pipeline.src.deduplication.matching import PUBLICATION_STATUS_RANK


def _completeness_score(row: dict) -> int:
    score = 0
    if row.get("doi"):
        score += 1
    if row.get("abstract"):
        score += 1
    if row.get("venue"):
        score += 1
    if row.get("citation_count") is not None:
        score += 1
    return score


def pick_canonical_id(rows: list[dict]) -> str:
    """rows: the full `records` row dicts for one duplicate group (>= 2)."""
    if not rows:
        raise ValueError("pick_canonical_id requires at least one row")

    def sort_key(row: dict) -> tuple[int, int, str]:
        status_rank = PUBLICATION_STATUS_RANK.get(row.get("publication_status") or "unknown", 99)
        return (status_rank, -_completeness_score(row), row["record_id"])

    return min(rows, key=sort_key)["record_id"]
