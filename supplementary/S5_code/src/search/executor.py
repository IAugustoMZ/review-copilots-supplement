"""Runs a connector search and persists the outcome: one search_runs row
(always, regardless of ok/zero_results/error), one query_hits row per record,
and one records row per raw hit -- deduplication (src/deduplication) runs as a
separate, later step, so D0 here genuinely means "everything anyone's search
turned up," duplicates included.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import Engine

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.common.publication_status import classify_publication_status
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.search.connectors.base import (
    AuthorStub,
    ConnectorResponse,
    QueryFilters,
    SearchConnector,
)


def _get_or_create_author_id(conn, authors_table, record_authors_table, record_id: str, author: AuthorStub) -> None:
    """Reuses an existing author row on an exact full_name match (case-insensitive);
    otherwise creates one. No fuzzy author disambiguation -- exact match only, so
    two distinct same-named authors are a known, accepted limitation rather than a
    silent wrong merge."""
    existing = conn.execute(
        authors_table.select().where(authors_table.c.full_name == author.full_name)
    ).mappings().first()
    author_id = existing["author_id"] if existing else new_id()
    if not existing:
        conn.execute(
            authors_table.insert().values(author_id=author_id, full_name=author.full_name, surname=author.surname)
        )
    conn.execute(
        record_authors_table.insert().values(record_id=record_id, author_id=author_id, position=author.position)
    )


def run_search(
    engine: Engine,
    connector: SearchConnector,
    query_id: str,
    query_text: str,
    filters: QueryFilters,
    corpus_type: str = "primary",
    run_logger: RunLogger | None = None,
    iteration: int = 1,
) -> tuple[str, ConnectorResponse]:
    """Execute one search, log it, and persist results. Returns (search_run_id, response)."""
    run_logger = run_logger or RunLogger(component="search", run_id=query_id)
    search_run_id = new_id()
    now = datetime.now(UTC).isoformat()

    response = connector.search(query_text, filters)

    run_logger.log_event(
        event="search_run",
        search_run_id=search_run_id,
        connector=connector.name,
        query_id=query_id,
        query_text=query_text,
        status=response.status,
        record_count=len(response.records),
        error_detail=response.error_detail,
        rate_limit_events=response.rate_limit_events,
    )

    search_runs = reflect_table(engine, "search_runs")
    records_table = reflect_table(engine, "records")
    query_hits = reflect_table(engine, "query_hits")
    authors_table = reflect_table(engine, "authors")
    record_authors_table = reflect_table(engine, "record_authors")

    with engine.begin() as conn:
        conn.execute(
            search_runs.insert().values(
                search_run_id=search_run_id,
                query_id=query_id,
                query_text=query_text,
                database=connector.name,
                timestamp=now,
                filters_json=json.dumps(vars(filters), default=str),
                page_count=response.page_count,
                record_count=len(response.records),
                api_version=response.api_version,
                status=response.status,
                error_detail=response.error_detail,
                rate_limit_events=response.rate_limit_events,
            )
        )

        for rank, raw in enumerate(response.records, start=1):
            record_id = new_id()
            conn.execute(
                records_table.insert().values(
                    record_id=record_id,
                    doi=raw.doi,
                    title=raw.title,
                    title_normalized=normalize_title(raw.title),
                    year=raw.year,
                    venue=raw.venue,
                    publication_type=raw.publication_type,
                    publication_status=classify_publication_status(raw.publication_type, raw.doi),
                    abstract=raw.abstract,
                    abstract_source=raw.abstract_source,
                    keywords=json.dumps(raw.keywords),
                    publisher=raw.publisher,
                    issn=raw.issn,
                    citation_count=raw.citation_count,
                    oa_status=raw.oa_status,
                    oa_url=raw.oa_url,
                    corpus_type=corpus_type,
                    retrieval_source=connector.name,
                    query_id=query_id,
                    retrieval_timestamp=now,
                    raw_source_id=raw.raw_source_id,
                    raw_metadata_json=json.dumps(raw.raw_metadata, default=str),
                    discovery_iteration=iteration,
                    d0=1,
                    d1=0,
                    d2=0,
                    d3=0,
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                query_hits.insert().values(
                    search_run_id=search_run_id,
                    record_id=record_id,
                    rank_in_results=rank,
                )
            )
            seen_author_names: set[str] = set()
            for author in raw.authors:
                # A source can legitimately list the same person twice on one
                # work (e.g. duplicated corresponding-author entry) -- since
                # authors are matched by exact full_name, that would try to
                # link the same (record_id, author_id) pair twice and violate
                # record_authors' uniqueness constraint. Keep the first
                # (correct) position, drop the repeat.
                if author.full_name in seen_author_names:
                    continue
                seen_author_names.add(author.full_name)
                _get_or_create_author_id(conn, authors_table, record_authors_table, record_id, author)

    return search_run_id, response
