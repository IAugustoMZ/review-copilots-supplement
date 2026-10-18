from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.validation.sampling import (
    build_label_stratified_sample,
    build_validation_sample,
    extend_sample_with_random_excludes,
    load_validation_sample,
    mandatory_review_ids,
    write_sample_files,
)


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record(engine, record_id: str, title: str = "A Paper", abstract: str | None = None) -> str:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            records_table.insert().values(
                record_id=record_id,
                title=title,
                title_normalized=normalize_title(title),
                abstract=abstract,
                d1=1,
                created_at=now,
                updated_at=now,
            )
        )
    return record_id


def _insert_ai_topical_decision(engine, record_id: str, decision: str, confidence: str) -> None:
    classifications_ai = reflect_table(engine, "classifications_ai")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            classifications_ai.insert().values(
                classification_id=new_id(),
                record_id=record_id,
                dimension="topical_inclusion",
                label=decision,
                confidence=confidence,
                coder="claude-sonnet-5",
                timestamp=now,
            )
        )


def test_build_validation_sample_includes_every_record_when_pool_smaller_than_target(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, "r1")
    _insert_record(engine, "r2")

    result = build_validation_sample(engine, seed=1, target_size=90)

    assert result.pool_size == 2
    assert {row.record_id for row in result.rows} == {"r1", "r2"}


def test_build_validation_sample_excludes_non_d1_and_merged_records(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    core = _insert_record(engine, "r1")
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            records_table.insert().values(
                record_id="r2", title="x", title_normalized="x", d1=0, created_at=now, updated_at=now
            )
        )
        conn.execute(
            records_table.insert().values(
                record_id="r3",
                title="y",
                title_normalized="y",
                d1=1,
                canonical_record_id=core,
                created_at=now,
                updated_at=now,
            )
        )

    result = build_validation_sample(engine, seed=1, target_size=90)

    assert {row.record_id for row in result.rows} == {"r1"}


def test_build_validation_sample_is_deterministic_for_a_given_seed(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    for i in range(20):
        _insert_record(engine, f"r{i}")

    first = build_validation_sample(engine, seed=7, target_size=5)
    second = build_validation_sample(engine, seed=7, target_size=5)

    assert [row.record_id for row in first.rows] == [row.record_id for row in second.rows]


def test_build_validation_sample_stratifies_low_confidence_ai_decisions(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    low_conf = _insert_record(engine, "r1")
    _insert_ai_topical_decision(engine, low_conf, "include", "low")
    for i in range(2, 10):
        rid = _insert_record(engine, f"r{i}")
        _insert_ai_topical_decision(engine, rid, "include", "high")

    result = build_validation_sample(engine, seed=1, target_size=3)

    strata_by_record = {row.record_id: row.stratum for row in result.rows}
    assert strata_by_record[low_conf] == "low_confidence"


def test_build_validation_sample_stratifies_chemical_pse_keyword_matches(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    ce_record = _insert_record(engine, "r1", title="A distillation column control study")
    for i in range(2, 10):
        _insert_record(engine, f"r{i}", title="Something else entirely")

    result = build_validation_sample(engine, seed=1, target_size=3)

    strata_by_record = {row.record_id: row.stratum for row in result.rows}
    assert strata_by_record[ce_record] == "chemical_pse"


def test_write_and_load_sample_round_trips(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, "r1")
    _insert_record(engine, "r2")
    result = build_validation_sample(engine, seed=1, target_size=90)

    csv_path = tmp_path / "sample.csv"
    manifest_path = tmp_path / "manifest.json"
    write_sample_files(result, csv_path, manifest_path)

    loaded = load_validation_sample(csv_path)
    assert loaded == {row.record_id: row.stratum for row in result.rows}
    assert manifest_path.exists()


def test_load_validation_sample_returns_none_when_not_built(tmp_path: Path) -> None:
    assert load_validation_sample(tmp_path / "missing.csv") is None


def test_mandatory_review_ids_flags_uncertain_and_non_high_confidence(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    uncertain = _insert_record(engine, "r1")
    _insert_ai_topical_decision(engine, uncertain, "uncertain", "low")
    medium_conf = _insert_record(engine, "r2")
    _insert_ai_topical_decision(engine, medium_conf, "include", "medium")
    high_conf = _insert_record(engine, "r3")
    _insert_ai_topical_decision(engine, high_conf, "exclude", "high")
    no_decision = _insert_record(engine, "r4")

    mandatory = mandatory_review_ids(engine)

    assert mandatory[uncertain] == "ai_uncertain"
    assert mandatory[medium_conf] == "ai_non_high_confidence"
    assert mandatory[no_decision] == "no_ai_decision"
    assert high_conf not in mandatory


def _seed_label_pool(engine, n_include: int, n_exclude: int, n_uncertain: int) -> None:
    """Bulk-insert a labelled D1 pool in two statements -- one row at a time is
    too slow at these sizes to keep the suite fast."""
    records_table = reflect_table(engine, "records")
    classifications_ai = reflect_table(engine, "classifications_ai")
    now = datetime.now(UTC).isoformat()

    specs: list[tuple[str, str, str]] = []
    specs += [(f"inc-{i}", "include", "high") for i in range(n_include)]
    specs += [(f"exc-{i}", "exclude", "low" if i < 5 else "high") for i in range(n_exclude)]
    specs += [(f"unc-{i}", "uncertain", "low") for i in range(n_uncertain)]

    with engine.begin() as conn:
        conn.execute(
            records_table.insert(),
            [
                {
                    "record_id": record_id,
                    "title": "A Paper",
                    "title_normalized": normalize_title("A Paper"),
                    "abstract": None,
                    "d1": 1,
                    "created_at": now,
                    "updated_at": now,
                }
                for record_id, _decision, _confidence in specs
            ],
        )
        conn.execute(
            classifications_ai.insert(),
            [
                {
                    "classification_id": new_id(),
                    "record_id": record_id,
                    "dimension": "topical_inclusion",
                    "label": decision,
                    "confidence": confidence,
                    "coder": "claude-opus-5",
                    "timestamp": now,
                }
                for record_id, decision, confidence in specs
            ],
        )


def test_label_stratified_sample_hits_the_quota_fractions(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_label_pool(engine, n_include=500, n_exclude=1200, n_uncertain=120)

    result = build_label_stratified_sample(engine, seed=42, target_size=400)

    assert len(result.rows) == 400
    assert result.stratum_counts == {"ai_include": 200, "ai_exclude": 130, "ai_uncertain": 70}
    assert result.population_counts == {"ai_include": 500, "ai_exclude": 1200, "ai_uncertain": 120}


def test_label_stratified_sample_draws_only_from_its_own_label(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_label_pool(engine, n_include=500, n_exclude=1200, n_uncertain=120)

    result = build_label_stratified_sample(engine, seed=42, target_size=400)

    prefix_for = {"ai_include": "inc-", "ai_exclude": "exc-", "ai_uncertain": "unc-"}
    for row in result.rows:
        assert row.record_id.startswith(prefix_for[row.stratum])


def test_label_stratified_sample_is_deterministic_for_a_given_seed(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_label_pool(engine, n_include=500, n_exclude=1200, n_uncertain=120)

    first = build_label_stratified_sample(engine, seed=42, target_size=400)
    second = build_label_stratified_sample(engine, seed=42, target_size=400)
    third = build_label_stratified_sample(engine, seed=7, target_size=400)

    assert [r.record_id for r in first.rows] == [r.record_id for r in second.rows]
    assert [r.record_id for r in first.rows] != [r.record_id for r in third.rows]


def test_label_stratified_sample_prefers_lower_confidence_excludes(tmp_path: Path) -> None:
    """The exclude cell should still point at the records most likely to be wrong."""
    engine = _db(tmp_path)
    _seed_label_pool(engine, n_include=500, n_exclude=1200, n_uncertain=120)

    result = build_label_stratified_sample(engine, seed=42, target_size=400)

    sampled_excludes = {r.record_id for r in result.rows if r.stratum == "ai_exclude"}
    assert {f"exc-{i}" for i in range(5)} <= sampled_excludes


def test_label_stratified_shortfall_spills_into_the_exclude_stratum(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_label_pool(engine, n_include=10, n_exclude=1200, n_uncertain=4)

    result = build_label_stratified_sample(engine, seed=42, target_size=400)

    assert len(result.rows) == 400
    assert result.stratum_counts["ai_include"] == 10
    assert result.stratum_counts["ai_uncertain"] == 4
    assert result.stratum_counts["ai_exclude"] == 386


def test_manifest_records_sampling_weights(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_label_pool(engine, n_include=500, n_exclude=1200, n_uncertain=120)
    result = build_label_stratified_sample(engine, seed=42, target_size=400)

    csv_path = tmp_path / "sample.csv"
    manifest_path = tmp_path / "manifest.json"
    write_sample_files(result, csv_path, manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sampling_weights"]["ai_include"] == 500 / 200
    assert manifest["sampling_weights"]["ai_exclude"] == 1200 / 130
    assert manifest["sampling_weights"]["ai_uncertain"] == 120 / 70


# --- D-35: the exclude stratum is two frames --------------------------------


def test_extend_sample_relabels_existing_excludes_by_their_real_frame(tmp_path: Path) -> None:
    """The original draw took 5 low-confidence excludes and 125 high-confidence
    ones. Those are two different populations and must not share a weight."""
    engine = _db(tmp_path)
    _seed_label_pool(engine, n_include=500, n_exclude=1200, n_uncertain=120)
    original = build_label_stratified_sample(engine, seed=42, target_size=400)
    existing = {row.record_id: row.stratum for row in original.rows}

    result = extend_sample_with_random_excludes(engine, existing, size=100, seed=43)

    strata = {row.record_id: row.stratum for row in result.rows}
    assert "ai_exclude" not in set(strata.values())
    assert result.stratum_counts["ai_exclude_borderline"] == 5
    assert result.stratum_counts["ai_exclude_high"] == 125 + 100


def test_extend_sample_keeps_every_already_adjudicated_record(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_label_pool(engine, n_include=500, n_exclude=1200, n_uncertain=120)
    original = build_label_stratified_sample(engine, seed=42, target_size=400)
    existing = {row.record_id: row.stratum for row in original.rows}

    result = extend_sample_with_random_excludes(engine, existing, size=100, seed=43)

    assert set(existing) <= {row.record_id for row in result.rows}
    assert len(result.rows) == 500


def test_extend_sample_draws_new_records_only_from_high_confidence_excludes(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_label_pool(engine, n_include=500, n_exclude=1200, n_uncertain=120)
    original = build_label_stratified_sample(engine, seed=42, target_size=400)
    existing = {row.record_id: row.stratum for row in original.rows}

    result = extend_sample_with_random_excludes(engine, existing, size=100, seed=43)

    new_ids = {row.record_id for row in result.rows} - set(existing)
    assert len(new_ids) == 100
    assert all(record_id.startswith("exc-") for record_id in new_ids)
    # exc-0..4 are the low-confidence ones, and were already taken.
    assert not any(record_id in {f"exc-{i}" for i in range(5)} for record_id in new_ids)


def test_extend_sample_new_draw_is_not_confidence_ordered(tmp_path: Path) -> None:
    """The whole point of the extension: this draw is random, so a different
    seed reaches a different set of records."""
    engine = _db(tmp_path)
    _seed_label_pool(engine, n_include=500, n_exclude=1200, n_uncertain=120)
    original = build_label_stratified_sample(engine, seed=42, target_size=400)
    existing = {row.record_id: row.stratum for row in original.rows}

    first = extend_sample_with_random_excludes(engine, existing, size=100, seed=43)
    same = extend_sample_with_random_excludes(engine, existing, size=100, seed=43)
    other = extend_sample_with_random_excludes(engine, existing, size=100, seed=7)

    assert [r.record_id for r in first.rows] == [r.record_id for r in same.rows]
    assert {r.record_id for r in first.rows} != {r.record_id for r in other.rows}


def test_extend_sample_manifest_weights_describe_the_frames_actually_drawn(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_label_pool(engine, n_include=500, n_exclude=1200, n_uncertain=120)
    original = build_label_stratified_sample(engine, seed=42, target_size=400)
    existing = {row.record_id: row.stratum for row in original.rows}
    result = extend_sample_with_random_excludes(engine, existing, size=100, seed=43)

    csv_path = tmp_path / "sample.csv"
    manifest_path = tmp_path / "manifest.json"
    write_sample_files(result, csv_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 5 low-confidence excludes exist in this pool, 1195 high-confidence ones.
    assert manifest["population_counts"]["ai_exclude_borderline"] == 5
    assert manifest["population_counts"]["ai_exclude_high"] == 1195
    assert manifest["sampling_weights"]["ai_exclude_borderline"] == 5 / 5
    assert manifest["sampling_weights"]["ai_exclude_high"] == 1195 / 225
    # The 50x weight that the single lumped exclude stratum implied is gone.
    assert "ai_exclude" not in manifest["sampling_weights"]
