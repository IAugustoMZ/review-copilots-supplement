from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from review_pipeline.src.analysis.saturation import (
    IterationStat,
    compute_saturation_curve,
    detect_elbow,
    log_iteration,
    plot_saturation_curve,
    write_saturation_report,
)
from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_canonical(engine, *, earliest_seen_iteration: int, d1: int = 0, **overrides) -> str:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    record_id = overrides.pop("record_id", new_id())
    title = overrides.pop("title", f"Paper {record_id}")
    values = {
        "record_id": record_id,
        "title": title,
        "title_normalized": normalize_title(title),
        "publication_status": "unknown",
        "discovery_iteration": earliest_seen_iteration,
        "earliest_seen_iteration": earliest_seen_iteration,
        "d1": d1,
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    with engine.begin() as conn:
        conn.execute(records_table.insert().values(**values))
    return record_id


def test_compute_saturation_curve_counts_new_and_cumulative(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_canonical(engine, earliest_seen_iteration=1, d1=1)
    _insert_canonical(engine, earliest_seen_iteration=1, d1=0)
    _insert_canonical(engine, earliest_seen_iteration=2, d1=1)

    curve = compute_saturation_curve(engine)
    assert len(curve) == 2
    it1, it2 = curve
    assert it1.iteration == 1
    assert it1.new_unique_records == 2
    assert it1.cumulative_unique_records == 2
    assert it1.new_d1_eligible == 1
    assert it1.marginal_growth_pct == 100.0  # first iteration always counts as full growth

    assert it2.iteration == 2
    assert it2.new_unique_records == 1
    assert it2.cumulative_unique_records == 3
    assert it2.cumulative_d1_eligible == 2
    assert it2.marginal_growth_pct == 50.0  # 1 new / 2 previous cumulative


def test_compute_saturation_curve_excludes_merged_away_records(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    canonical_id = _insert_canonical(engine, earliest_seen_iteration=1)
    _insert_canonical(engine, earliest_seen_iteration=2, canonical_record_id=canonical_id)

    curve = compute_saturation_curve(engine)
    # the merged-away record has canonical_record_id set -- it must not count
    # as a separate unique record in either iteration's bucket
    assert sum(s.new_unique_records for s in curve) == 1


def test_compute_saturation_curve_uses_logged_labels(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_canonical(engine, earliest_seen_iteration=1)
    log_iteration(
        engine,
        iteration_number=1,
        label="Initial production sweep",
        description="CE1-14 + venues, Crossref+OpenAlex",
        connectors_used=["crossref", "openalex"],
        started_at=datetime.now(UTC).isoformat(),
    )

    curve = compute_saturation_curve(engine)
    assert curve[0].label == "Initial production sweep"


def test_compute_saturation_curve_shows_zero_yield_logged_iteration(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_canonical(engine, earliest_seen_iteration=1)
    log_iteration(
        engine,
        iteration_number=1,
        label="Initial sweep",
        description=None,
        connectors_used=["crossref"],
        started_at=datetime.now(UTC).isoformat(),
    )
    log_iteration(
        engine,
        iteration_number=2,
        label="Crossref-only retry",
        description=None,
        connectors_used=["crossref"],
        started_at=datetime.now(UTC).isoformat(),
    )
    # iteration 2 found nothing new -- every candidate merged into an
    # iteration-1 canonical record, so no canonical row carries
    # earliest_seen_iteration=2. It must still show up as its own row.

    curve = compute_saturation_curve(engine)
    assert [s.iteration for s in curve] == [1, 2]
    it2 = curve[1]
    assert it2.new_unique_records == 0
    assert it2.cumulative_unique_records == 1
    assert it2.marginal_growth_pct == 0.0
    assert it2.label == "Crossref-only retry"


def test_detect_elbow_requires_consecutive_low_growth_iterations() -> None:
    curve = [
        IterationStat(1, "a", 100, 100, 50, 50, 100.0),
        IterationStat(2, "b", 40, 140, 20, 70, 40.0),
        IterationStat(3, "c", 2, 142, 1, 71, 1.43),
        IterationStat(4, "d", 1, 143, 0, 71, 0.7),
    ]
    assert detect_elbow(curve, min_marginal_pct=5.0, consecutive=2) == 3


def test_detect_elbow_returns_none_when_still_growing() -> None:
    curve = [
        IterationStat(1, "a", 100, 100, 50, 50, 100.0),
        IterationStat(2, "b", 40, 140, 20, 70, 40.0),
    ]
    assert detect_elbow(curve, min_marginal_pct=5.0, consecutive=2) is None


def test_detect_elbow_resets_run_on_a_rebound(tmp_path: Path) -> None:
    curve = [
        IterationStat(1, "a", 100, 100, 50, 50, 100.0),
        IterationStat(2, "b", 2, 102, 1, 51, 2.0),  # single low iteration, not consecutive
        IterationStat(3, "c", 50, 152, 25, 76, 49.0),  # rebound -- e.g. snowballing found a new vein
        IterationStat(4, "d", 1, 153, 0, 76, 0.66),
        IterationStat(5, "e", 1, 154, 0, 76, 0.65),
    ]
    assert detect_elbow(curve, min_marginal_pct=5.0, consecutive=2) == 4


def test_write_saturation_report_and_plot(tmp_path: Path) -> None:
    curve = [
        IterationStat(1, "Initial sweep", 66, 66, 40, 40, 100.0),
        IterationStat(2, "S2 retry", 3, 69, 2, 42, 4.5),
        IterationStat(3, "Snowball", 1, 70, 1, 43, 1.4),
    ]
    elbow = detect_elbow(curve)
    report_path = tmp_path / "saturation_curve.md"
    plot_path = tmp_path / "saturation_curve.png"

    write_saturation_report(curve, elbow, report_path)
    plot_saturation_curve(curve, elbow, plot_path)

    text = report_path.read_text(encoding="utf-8")
    assert "Elbow detected at iteration 2" in text
    assert "Initial sweep" in text
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
