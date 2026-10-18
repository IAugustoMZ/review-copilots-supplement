from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.search.connectors.base import (
    AuthorStub,
    ConnectorResponse,
    QueryFilters,
    RawRecord,
    SearchConnector,
)
from review_pipeline.src.search.executor import run_search


class _FakeOkConnector(SearchConnector):
    name = "fake_ok"

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        return ConnectorResponse(
            status="ok",
            records=[
                RawRecord(
                    doi="10.1000/fake1",
                    title="A Fake LLM Copilot Paper",
                    year=2026,
                    venue="Fake Journal",
                    authors=[AuthorStub(full_name="A. Uthor", surname="Uthor", position=1)],
                    raw_source_id="fake1",
                )
            ],
            page_count=1,
        )


class _FakeErrorConnector(SearchConnector):
    name = "fake_error"

    def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
        return ConnectorResponse(status="error", error_detail="simulated connector failure")


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def test_run_search_persists_records_and_query_hits(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    logger = RunLogger(component="search", run_id="Q1", logs_dir=tmp_path / "logs")
    search_run_id, response = run_search(
        engine,
        _FakeOkConnector(),
        query_id="Q1",
        query_text="llm copilot",
        filters=QueryFilters(),
        run_logger=logger,
    )

    assert response.status == "ok"
    with engine.connect() as conn:
        run_row = conn.execute(
            text("SELECT status, record_count, error_detail FROM search_runs WHERE search_run_id=:id"),
            {"id": search_run_id},
        ).one()
        assert run_row.status == "ok"
        assert run_row.record_count == 1
        assert run_row.error_detail is None

        record_row = conn.execute(text("SELECT title, doi, title_normalized, d0 FROM records")).one()
        assert record_row.title == "A Fake LLM Copilot Paper"
        assert record_row.doi == "10.1000/fake1"
        assert record_row.title_normalized == "a fake llm copilot paper"
        assert record_row.d0 == 1

        hit_count = conn.execute(text("SELECT COUNT(*) FROM query_hits")).scalar_one()
        assert hit_count == 1

        author_row = conn.execute(text("SELECT full_name, surname FROM authors")).one()
        assert author_row.full_name == "A. Uthor"
        assert author_row.surname == "Uthor"
        record_author_count = conn.execute(text("SELECT COUNT(*) FROM record_authors")).scalar_one()
        assert record_author_count == 1


def test_run_search_dedupes_repeated_author_name_within_one_record(tmp_path: Path) -> None:
    class _DuplicateAuthorOnSameRecordConnector(SearchConnector):
        name = "fake_dup_author"

        def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
            duped = AuthorStub(full_name="A. Uthor", surname="Uthor", position=1)
            return ConnectorResponse(
                status="ok",
                records=[
                    RawRecord(
                        doi="10.1/dup",
                        title="A Paper With A Duplicated Author Entry",
                        authors=[duped, AuthorStub(full_name="A. Uthor", surname="Uthor", position=10)],
                        raw_source_id="dup",
                    )
                ],
            )

    engine = _db(tmp_path)
    logger = RunLogger(component="search", run_id="Q7", logs_dir=tmp_path / "logs")
    # Must not raise a UNIQUE constraint violation on record_authors.
    _search_run_id, response = run_search(
        engine,
        _DuplicateAuthorOnSameRecordConnector(),
        query_id="Q7",
        query_text="anything",
        filters=QueryFilters(),
        run_logger=logger,
    )
    assert response.status == "ok"

    with engine.connect() as conn:
        author_count = conn.execute(text("SELECT COUNT(*) FROM authors")).scalar_one()
        record_author_count = conn.execute(text("SELECT COUNT(*) FROM record_authors")).scalar_one()
    assert author_count == 1
    assert record_author_count == 1  # only the first occurrence is linked


def test_run_search_reuses_existing_author_on_exact_name_match(tmp_path: Path) -> None:
    class _TwoRecordsSameAuthorConnector(SearchConnector):
        name = "fake_shared_author"

        def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
            shared_author = AuthorStub(full_name="A. Uthor", surname="Uthor", position=1)
            return ConnectorResponse(
                status="ok",
                records=[
                    RawRecord(doi="10.1/a", title="Paper A", authors=[shared_author], raw_source_id="a"),
                    RawRecord(doi="10.1/b", title="Paper B", authors=[shared_author], raw_source_id="b"),
                ],
            )

    engine = _db(tmp_path)
    logger = RunLogger(component="search", run_id="Q3", logs_dir=tmp_path / "logs")
    run_search(
        engine,
        _TwoRecordsSameAuthorConnector(),
        query_id="Q3",
        query_text="anything",
        filters=QueryFilters(),
        run_logger=logger,
    )

    with engine.connect() as conn:
        author_count = conn.execute(text("SELECT COUNT(*) FROM authors")).scalar_one()
        assert author_count == 1  # reused, not duplicated
        record_author_count = conn.execute(text("SELECT COUNT(*) FROM record_authors")).scalar_one()
        assert record_author_count == 2


def test_run_search_classifies_publication_status_from_publication_type(tmp_path: Path) -> None:
    class _KnownTypesConnector(SearchConnector):
        name = "fake_types"

        def search(self, query: str, filters: QueryFilters) -> ConnectorResponse:
            return ConnectorResponse(
                status="ok",
                records=[
                    RawRecord(title="Journal Paper", publication_type="peer_reviewed_journal", raw_source_id="j"),
                    RawRecord(title="A Preprint", publication_type="posted-content", raw_source_id="p"),
                    RawRecord(title="Something Else", publication_type="dataset", raw_source_id="u"),
                ],
            )

    engine = _db(tmp_path)
    logger = RunLogger(component="search", run_id="Q4", logs_dir=tmp_path / "logs")
    run_search(
        engine,
        _KnownTypesConnector(),
        query_id="Q4",
        query_text="anything",
        filters=QueryFilters(),
        run_logger=logger,
    )

    with engine.connect() as conn:
        rows = {
            r.title: r.publication_status
            for r in conn.execute(text("SELECT title, publication_status FROM records"))
        }
    assert rows["Journal Paper"] == "peer_reviewed_journal"
    assert rows["A Preprint"] == "preprint"
    assert rows["Something Else"] == "unknown"


def test_run_search_stamps_discovery_iteration(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    logger = RunLogger(component="search", run_id="Q5", logs_dir=tmp_path / "logs")
    run_search(
        engine,
        _FakeOkConnector(),
        query_id="Q5",
        query_text="llm copilot",
        filters=QueryFilters(),
        run_logger=logger,
        iteration=3,
    )

    with engine.connect() as conn:
        iteration = conn.execute(text("SELECT discovery_iteration FROM records")).scalar_one()
    assert iteration == 3


def test_run_search_defaults_discovery_iteration_to_one(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    logger = RunLogger(component="search", run_id="Q6", logs_dir=tmp_path / "logs")
    run_search(
        engine,
        _FakeOkConnector(),
        query_id="Q6",
        query_text="llm copilot",
        filters=QueryFilters(),
        run_logger=logger,
    )

    with engine.connect() as conn:
        iteration = conn.execute(text("SELECT discovery_iteration FROM records")).scalar_one()
    assert iteration == 1


def test_run_search_persists_error_without_writing_records(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    logger = RunLogger(component="search", run_id="Q2", logs_dir=tmp_path / "logs")
    search_run_id, response = run_search(
        engine,
        _FakeErrorConnector(),
        query_id="Q2",
        query_text="anything",
        filters=QueryFilters(),
        run_logger=logger,
    )

    assert response.status == "error"
    with engine.connect() as conn:
        run_row = conn.execute(
            text("SELECT status, error_detail FROM search_runs WHERE search_run_id=:id"),
            {"id": search_run_id},
        ).one()
        assert run_row.status == "error"
        assert run_row.error_detail == "simulated connector failure"

        record_count = conn.execute(text("SELECT COUNT(*) FROM records")).scalar_one()
        assert record_count == 0
