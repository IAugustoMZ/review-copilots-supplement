"""Phase 6: sentinel / known-item workflow.

The sentinel set is hand-curated in config/sentinel_set.yaml (source of truth);
`sentinel_records` is a materialized, check-annotated copy of it -- `load`
replaces the table wholesale from the YAML, `check` matches each entry against
`records` and annotates how (if at all) it was retrieved. This is a debugging
tool for the search strategy, never a recall percentage over all literature
(docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel
from sqlalchemy import Engine

from review_pipeline.src.common.config import load_yaml
from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_doi, normalize_title
from review_pipeline.src.db.tables import reflect_table


class SentinelSeed(BaseModel):
    doi: str | None = None
    title: str
    venue: str | None = None
    year: int | None = None
    ce_theme: str | None = None
    why_known_relevant: str


def load_sentinel_config() -> list[SentinelSeed]:
    raw = load_yaml("sentinel_set.yaml")
    return [SentinelSeed(**entry) for entry in raw.get("sentinels", [])]


def load_sentinels_to_db(engine: Engine) -> int:
    """Replace sentinel_records wholesale from config/sentinel_set.yaml."""
    seeds = load_sentinel_config()
    sentinel_table = reflect_table(engine, "sentinel_records")
    with engine.begin() as conn:
        conn.execute(sentinel_table.delete())
        for seed in seeds:
            conn.execute(
                sentinel_table.insert().values(
                    sentinel_id=new_id(),
                    doi=seed.doi,
                    title=seed.title,
                    venue=seed.venue,
                    year=seed.year,
                    ce_theme=seed.ce_theme,
                    why_known_relevant=seed.why_known_relevant,
                    retrieved_generic=0,
                    retrieved_ce_search=0,
                    retrieved_venue_search=0,
                )
            )
    return len(seeds)


def _is_match(record_row: dict, sentinel_row: dict) -> bool:
    sentinel_doi = normalize_doi(sentinel_row.get("doi"))
    if sentinel_doi and normalize_doi(record_row.get("doi")) == sentinel_doi:
        return True
    return normalize_title(record_row.get("title", "")) == normalize_title(sentinel_row.get("title", ""))


@dataclass
class SentinelCheckResult:
    sentinel_id: str
    title: str
    matched: bool
    retrieved_generic: bool
    retrieved_ce_search: bool
    retrieved_venue_search: bool
    retrieval_sources: list[str]


def check_sentinels(engine: Engine) -> list[SentinelCheckResult]:
    """Match every sentinel_records row against the current `records` table and
    persist the outcome. A query_id starting with 'CE' came from a CE-family
    search (Phase 4); one starting with 'venue.' came from venue search
    (Phase 5); anything else counts as a generic search."""
    sentinel_table = reflect_table(engine, "sentinel_records")
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()

    with engine.begin() as conn:
        seed_rows = [dict(r) for r in conn.execute(sentinel_table.select()).mappings().all()]
        all_records = [dict(r) for r in conn.execute(records_table.select()).mappings().all()]

        results = []
        for seed_row in seed_rows:
            matches = [r for r in all_records if _is_match(r, seed_row)]
            query_ids = [r.get("query_id") or "" for r in matches]
            retrieved_ce = any(qid.startswith("CE") for qid in query_ids)
            retrieved_venue = any(qid.startswith("venue.") for qid in query_ids)
            retrieved_generic = any(not qid.startswith(("CE", "venue.")) for qid in query_ids if qid)
            sources = sorted({r["retrieval_source"] for r in matches if r.get("retrieval_source")})

            conn.execute(
                sentinel_table.update()
                .where(sentinel_table.c.sentinel_id == seed_row["sentinel_id"])
                .values(
                    matched_record_id=matches[0]["record_id"] if matches else None,
                    retrieved_generic=int(retrieved_generic),
                    retrieved_ce_search=int(retrieved_ce),
                    retrieved_venue_search=int(retrieved_venue),
                    retrieval_sources_json=json.dumps(sources),
                    last_checked_at=now,
                )
            )
            results.append(
                SentinelCheckResult(
                    sentinel_id=seed_row["sentinel_id"],
                    title=seed_row["title"],
                    matched=bool(matches),
                    retrieved_generic=retrieved_generic,
                    retrieved_ce_search=retrieved_ce,
                    retrieved_venue_search=retrieved_venue,
                    retrieval_sources=sources,
                )
            )
    return results


def known_item_retrieval_rate(results: list[SentinelCheckResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.matched) / len(results)
