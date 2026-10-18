from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.validation import ml_assist, ml_model_metrics
from review_pipeline.src.validation.dimension_classification import (
    DimensionSchema,
    save_dimension_classification,
    save_single_value_classification,
)
from review_pipeline.src.validation.ml_assist import Suggestion


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


class _FakeModel:
    """Predicts whatever the test wired up per-text, independent of any real
    classifier -- lets evaluate_model()'s metrics be pinned against known
    inputs rather than depending on what a real TF-IDF/classifier happens to
    learn from a tiny sample."""

    def __init__(self, predictions: dict[str, str]) -> None:
        self._predictions = predictions

    def predict(self, text: str) -> Suggestion:
        return Suggestion(label=self._predictions[text], probability=1.0, training_size=0)


def test_evaluate_model_computes_expected_metrics_against_known_labels() -> None:
    model = _FakeModel({"a": "include", "b": "include", "c": "exclude", "d": "include"})
    texts = ["a", "b", "c", "d"]
    true_labels = ["include", "exclude", "exclude", "include"]  # model gets b wrong (predicts include, true exclude)

    metrics = ml_model_metrics.evaluate_model(model, texts, true_labels)

    assert metrics.n == 4
    assert metrics.accuracy == 0.75
    assert 0.0 < metrics.cohen_kappa < 1.0


def test_evaluate_model_returns_zero_kappa_when_only_one_label_present() -> None:
    """Cohen's kappa is undefined when true labels and predictions only
    share one class in common -- sklearn returns NaN for this; evaluate_model
    stores 0.0 instead so the metrics file never has to hold a non-standard
    JSON `NaN` literal."""
    model = _FakeModel({"a": "include", "b": "include"})
    metrics = ml_model_metrics.evaluate_model(model, ["a", "b"], ["include", "include"])

    assert metrics.accuracy == 1.0
    assert metrics.cohen_kappa == 0.0


def test_evaluate_models_scores_every_model_type_independently() -> None:
    models = {
        "logistic_regression": _FakeModel({"a": "include", "b": "exclude"}),
        "decision_tree": _FakeModel({"a": "exclude", "b": "exclude"}),
    }
    metrics = ml_model_metrics.evaluate_models(models, ["a", "b"], ["include", "exclude"])

    assert set(metrics) == {"logistic_regression", "decision_tree"}
    assert metrics["logistic_regression"].accuracy == 1.0
    assert metrics["decision_tree"].accuracy == 0.5


def _sample_batch_record(batch: int, active_model: str = "logistic_regression") -> ml_model_metrics.BatchRecord:
    metrics = {
        "logistic_regression": ml_model_metrics.ModelMetrics(
            accuracy=0.9, precision_macro=0.9, recall_macro=0.9, f1_macro=0.9, cohen_kappa=0.8, n=10
        ),
        "decision_tree": ml_model_metrics.ModelMetrics(
            accuracy=0.8, precision_macro=0.8, recall_macro=0.8, f1_macro=0.8, cohen_kappa=0.6, n=10
        ),
    }
    return ml_model_metrics.BatchRecord(
        batch=batch,
        timestamp=datetime.now(UTC).isoformat(),
        n_evaluated=10,
        n_trained_on=20,
        metrics=metrics,
        active_model=active_model,
    )


def test_record_batch_writes_json_and_plot_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    plot_path = tmp_path / "metrics.png"

    history = ml_model_metrics.record_batch(_sample_batch_record(1), path=path, plot_path=plot_path)

    assert len(history) == 1
    assert path.exists()
    assert plot_path.exists()

    loaded = ml_model_metrics.load_history(path)
    assert len(loaded) == 1
    assert loaded[0].batch == 1
    assert loaded[0].metrics["logistic_regression"].accuracy == 0.9
    assert loaded[0].active_model == "logistic_regression"


def test_record_batch_overwrites_same_batch_number_instead_of_duplicating(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    ml_model_metrics.record_batch(_sample_batch_record(1), path=path, plot_path=tmp_path / "metrics.png")
    ml_model_metrics.record_batch(
        _sample_batch_record(1, active_model="random_forest"), path=path, plot_path=tmp_path / "metrics.png"
    )

    history = ml_model_metrics.load_history(path)
    assert len(history) == 1
    assert history[0].active_model == "random_forest"


def test_record_batch_appends_a_new_batch_number(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    plot_path = tmp_path / "metrics.png"
    ml_model_metrics.record_batch(_sample_batch_record(1), path=path, plot_path=plot_path)
    ml_model_metrics.record_batch(_sample_batch_record(2), path=path, plot_path=plot_path)

    history = ml_model_metrics.load_history(path)
    assert [b.batch for b in history] == [1, 2]


def test_current_active_model_defaults_to_none_and_persists_when_saved(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    assert ml_model_metrics.load_current_active_model(path) is None

    ml_model_metrics.save_current_active_model("random_forest", path)

    assert ml_model_metrics.load_current_active_model(path) == "random_forest"


def test_save_current_active_model_preserves_existing_batch_history(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    ml_model_metrics.record_batch(_sample_batch_record(1), path=path, plot_path=tmp_path / "metrics.png")

    ml_model_metrics.save_current_active_model("decision_tree", path)

    assert ml_model_metrics.load_current_active_model(path) == "decision_tree"
    assert len(ml_model_metrics.load_history(path)) == 1


def test_record_batch_preserves_current_active_model(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    plot_path = tmp_path / "metrics.png"
    ml_model_metrics.save_current_active_model("decision_tree", path)

    ml_model_metrics.record_batch(_sample_batch_record(1), path=path, plot_path=plot_path)

    assert ml_model_metrics.load_current_active_model(path) == "decision_tree"


def test_pending_batch_defaults_to_empty_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    assert ml_model_metrics.load_pending_batch(path) == []

    ml_model_metrics.save_pending_batch([("some text", "include"), ("other text", "exclude")], path)

    assert ml_model_metrics.load_pending_batch(path) == [("some text", "include"), ("other text", "exclude")]


def test_save_pending_batch_preserves_existing_history_and_active_model(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    plot_path = tmp_path / "metrics.png"
    ml_model_metrics.record_batch(_sample_batch_record(1), path=path, plot_path=plot_path)
    ml_model_metrics.save_current_active_model("decision_tree", path)

    ml_model_metrics.save_pending_batch([("pending", "include")], path)

    assert ml_model_metrics.load_pending_batch(path) == [("pending", "include")]
    assert ml_model_metrics.load_current_active_model(path) == "decision_tree"
    assert len(ml_model_metrics.load_history(path)) == 1


def test_record_batch_preserves_pending_batch(tmp_path: Path) -> None:
    """record_batch() is called at retrain time, after evaluation but before
    the caller has necessarily cleared the pending batch -- it must not
    stomp on whatever's still recorded there."""
    path = tmp_path / "metrics.json"
    plot_path = tmp_path / "metrics.png"
    ml_model_metrics.save_pending_batch([("still pending", "exclude")], path)

    ml_model_metrics.record_batch(_sample_batch_record(1), path=path, plot_path=plot_path)

    assert ml_model_metrics.load_pending_batch(path) == [("still pending", "exclude")]


INCLUDE_TEXTS = [
    "A large language model copilot for chemical process troubleshooting",
    "Generative AI agent for industrial plant operations support",
    "LLM-based assistant for distillation column monitoring",
    "Retrieval-augmented copilot for manufacturing knowledge transfer",
]
EXCLUDE_TEXTS = [
    "A survey of medieval agricultural irrigation techniques",
    "Ceramic glaze chemistry in Bronze Age pottery",
    "Folk music preservation in rural communities",
    "History of cartography in the 18th century",
]


def _insert_record(engine, record_id: str, title: str) -> str:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            records_table.insert().values(
                record_id=record_id, title=title, title_normalized=normalize_title(title), d1=1,
                created_at=now, updated_at=now,
            )
        )
    return record_id


def _insert_human_decision(engine, record_id: str, label: str, timestamp: str) -> None:
    classifications_human = reflect_table(engine, "classifications_human")
    with engine.begin() as conn:
        conn.execute(
            classifications_human.insert().values(
                classification_id=new_id(), record_id=record_id, dimension="topical_inclusion", label=label,
                confidence="high", rationale="x", coder="alice", timestamp=timestamp,
            )
        )


def _seed_chronological_decisions(engine, n_repeats: int) -> None:
    """n_repeats copies of INCLUDE_TEXTS/EXCLUDE_TEXTS (interleaved so every
    4-record chunk is class-balanced), each batch strictly later than the
    last -- exercises backfill_batches' chronological chunking."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    i = 0
    for rep in range(n_repeats):
        for inc_title, exc_title in zip(INCLUDE_TEXTS, EXCLUDE_TEXTS, strict=True):
            ts = base.replace(minute=i % 60).isoformat()
            _insert_human_decision(engine, _insert_record(engine, f"inc{i}", inc_title), "include", ts)
            i += 1
            ts = base.replace(minute=i % 60).isoformat()
            _insert_human_decision(engine, _insert_record(engine, f"exc{i}", exc_title), "exclude", ts)
            i += 1


def test_backfill_batches_skips_batches_without_enough_prior_examples(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    seed = [(INCLUDE_TEXTS[0], "include"), (INCLUDE_TEXTS[1], "include"), (EXCLUDE_TEXTS[0], "exclude"), (EXCLUDE_TEXTS[1], "exclude")]
    for i, (title, label) in enumerate(seed):
        ts = base.replace(minute=i).isoformat()
        _insert_human_decision(engine, _insert_record(engine, f"r{i}", title), label, ts)

    history = ml_model_metrics.backfill_batches(
        engine, batch_size=4, path=tmp_path / "m.json", plot_path=tmp_path / "m.png"
    )

    # The only chunk's cumulative count is 4, below MIN_TOTAL_EXAMPLES (6) -- skipped.
    assert history == []


def test_backfill_batches_reconstructs_batches_once_enough_prior_data_exists(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_chronological_decisions(engine, n_repeats=3)  # 24 examples, batch_size=4 -> 6 chunks

    history = ml_model_metrics.backfill_batches(
        engine, batch_size=4, path=tmp_path / "m.json", plot_path=tmp_path / "m.png"
    )

    # Chunk 1's cumulative count is 4, below MIN_TOTAL_EXAMPLES (6) -- skipped.
    # Chunks 2-6 have cumulative counts 8, 12, 16, 20, 24 -- each evaluated by
    # out-of-fold CV over everything decided up to and including that chunk
    # (ml_model_metrics.cumulative_metrics), not just the chunk itself --
    # every example both trains (in some folds) and is evaluated (in its own
    # held-out fold), so n_trained_on == n_evaluated == the cumulative count.
    assert [b.n_trained_on for b in history] == [8, 12, 16, 20, 24]
    for b in history:
        assert set(b.metrics) == set(ml_assist.MODEL_SPECS)
        assert b.active_model == ml_assist.DEFAULT_ACTIVE_MODEL
        assert b.n_trained_on == b.n_evaluated

    assert (tmp_path / "m.json").exists()
    assert (tmp_path / "m.png").exists()


def test_plot_history_uses_solid_for_tfidf_and_dashed_for_embedding_with_shared_classifier_color(
    tmp_path: Path, monkeypatch
) -> None:
    """With 8 model_types on one plot, feature kind (tfidf/embedding) is
    encoded as linestyle and classifier kind as color -- so the same
    classifier's two feature variants share a color but are visually
    distinguishable, rather than 8 arbitrarily-colored lines."""
    import matplotlib.pyplot as plt

    captured_figs = []
    monkeypatch.setattr(plt, "close", lambda fig=None: captured_figs.append(fig))

    history = [
        ml_model_metrics.BatchRecord(
            batch=1,
            timestamp=datetime.now(UTC).isoformat(),
            n_evaluated=4,
            n_trained_on=8,
            metrics={
                "tfidf_logistic_regression": ml_model_metrics.ModelMetrics(
                    accuracy=0.9, precision_macro=0.9, recall_macro=0.9, f1_macro=0.9, cohen_kappa=0.8, n=4
                ),
                "embedding_logistic_regression": ml_model_metrics.ModelMetrics(
                    accuracy=0.8, precision_macro=0.8, recall_macro=0.8, f1_macro=0.8, cohen_kappa=0.6, n=4
                ),
            },
            active_model="tfidf_logistic_regression",
        )
    ]

    ml_model_metrics.plot_history(history, tmp_path / "m.png")

    assert len(captured_figs) == 1
    lines_by_label = {line.get_label(): line for line in captured_figs[0].axes[0].get_lines()}
    tfidf_line = lines_by_label["tfidf logistic regression"]
    embedding_line = lines_by_label["embedding logistic regression"]
    assert tfidf_line.get_linestyle() == "-"
    assert embedding_line.get_linestyle() == "--"
    assert tfidf_line.get_color() == embedding_line.get_color()


def test_backfill_batches_is_idempotent_and_overwrites_not_appends(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_chronological_decisions(engine, n_repeats=3)
    path = tmp_path / "m.json"
    plot_path = tmp_path / "m.png"

    first = ml_model_metrics.backfill_batches(engine, batch_size=4, path=path, plot_path=plot_path)
    second = ml_model_metrics.backfill_batches(engine, batch_size=4, path=path, plot_path=plot_path)

    assert [b.batch for b in first] == [b.batch for b in second]
    assert len(ml_model_metrics.load_history(path)) == len(second)


# --- 6-dimension model comparison (cumulative_dimension_metrics et al.) ---


def test_average_model_metrics_averages_each_field_and_sums_n() -> None:
    m1 = ml_model_metrics.ModelMetrics(
        accuracy=0.8, precision_macro=0.7, recall_macro=0.6, f1_macro=0.65, cohen_kappa=0.5, n=10
    )
    m2 = ml_model_metrics.ModelMetrics(
        accuracy=0.6, precision_macro=0.5, recall_macro=0.4, f1_macro=0.45, cohen_kappa=0.3, n=20
    )

    avg = ml_model_metrics._average_model_metrics([m1, m2])

    assert avg.accuracy == pytest.approx(0.7)
    assert avg.precision_macro == pytest.approx(0.6)
    assert avg.f1_macro == pytest.approx(0.55)
    assert avg.n == 30  # summed, not averaged -- still reflects total examples scored


def test_evaluate_dimension_model_multi_value_from_oof_pinned_against_perfect_predictions() -> None:
    """`labels` are {category: level} maps, not presence arrays -- the target
    is "rated high" (ml_assist._parse_high_categories)."""
    high = json.dumps({"chemical_engineering": "high"})
    low = json.dumps({"chemical_engineering": "low"})
    labels = [high, low, high, low]
    oof_by_category = {"chemical_engineering": [1, 0, 1, 0]}  # perfectly matches the true high/not-high target

    metrics = ml_model_metrics.evaluate_dimension_model_multi_value_from_oof(labels, oof_by_category)

    assert metrics is not None
    assert metrics.accuracy == 1.0
    assert metrics.n == 4


def test_evaluate_dimension_model_multi_value_from_oof_scores_the_level_not_mere_presence() -> None:
    """The regression guard for the defect this target replaced: a category
    the reviewer named on every record but rated 'high' on only half is
    scored on the half, not treated as present-everywhere. Under the old
    presence target these labels were a constant-1 vector, which is single
    class and therefore unscoreable."""
    high = json.dumps({"chemical_engineering": "high"})
    medium = json.dumps({"chemical_engineering": "medium"})
    labels = [high, medium, high, medium]
    oof_by_category = {"chemical_engineering": [1, 0, 1, 0]}

    metrics = ml_model_metrics.evaluate_dimension_model_multi_value_from_oof(labels, oof_by_category)

    assert metrics is not None
    assert metrics.accuracy == 1.0
    assert metrics.n == 4


def test_evaluate_dimension_model_multi_value_from_oof_skips_single_class_categories() -> None:
    """A category never rated high across every example in `labels` (by
    chance, or because it's genuinely marginal) can't be scored -- skipped,
    not counted as a 0."""
    labels = ["{}", "{}", "{}"]
    oof_by_category = {"rare_cat": [0, 0, 0]}

    assert ml_model_metrics.evaluate_dimension_model_multi_value_from_oof(labels, oof_by_category) is None


def test_evaluate_dimension_model_multi_value_from_oof_returns_none_without_categories() -> None:
    assert ml_model_metrics.evaluate_dimension_model_multi_value_from_oof(["{}"], {}) is None


def _seed_dimension_multi(engine, n_per_category: int = 6) -> None:
    """Every record names BOTH categories -- as the grouped H/M/L prompt
    always does -- and only the level separates them, so the fixture exercises
    the high/not-high target rather than a presence signal that no longer
    exists (ml_assist._parse_high_categories)."""
    for i in range(n_per_category):
        rid = _insert_record(engine, f"dinc{i}", f"{INCLUDE_TEXTS[i % len(INCLUDE_TEXTS)]} variant {i}")
        save_dimension_classification(
            engine, rid, "alice", "domain", {"high": ["chemical_engineering"], "medium": [], "low": ["safety"]}
        )
    for i in range(n_per_category):
        rid = _insert_record(engine, f"dexc{i}", f"{EXCLUDE_TEXTS[i % len(EXCLUDE_TEXTS)]} variant {i}")
        save_dimension_classification(
            engine, rid, "alice", "domain", {"high": ["safety"], "medium": [], "low": ["chemical_engineering"]}
        )


def _seed_dimension_single(engine, n_per_class: int = 6) -> None:
    for i in range(n_per_class):
        rid = _insert_record(engine, f"single_inc{i}", f"{INCLUDE_TEXTS[i % len(INCLUDE_TEXTS)]} variant {i}")
        save_single_value_classification(engine, rid, "alice", "evidence_type", "prototype")
    for i in range(n_per_class):
        rid = _insert_record(engine, f"single_exc{i}", f"{EXCLUDE_TEXTS[i % len(EXCLUDE_TEXTS)]} variant {i}")
        save_single_value_classification(engine, rid, "alice", "evidence_type", "conceptual_discussion")


_DIMENSION_SCHEMA = [
    DimensionSchema(name="domain", multi_value=True, categories=["chemical_engineering", "safety", "manufacturing"]),
    DimensionSchema(name="evidence_type", multi_value=False, categories=["prototype", "conceptual_discussion", "review"]),
]


def test_cumulative_dimension_metrics_averages_across_scored_dimensions(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_dimension_multi(engine)
    _seed_dimension_single(engine)

    evaluation = ml_model_metrics.cumulative_dimension_metrics(engine, _DIMENSION_SCHEMA)

    assert evaluation is not None
    # evidence_type is single-value/stratified -- always scoreable at this size.
    assert "evidence_type" in evaluation.per_dimension_metrics
    assert set(evaluation.aggregate_metrics) <= set(ml_assist.MODEL_SPECS)
    for model_type, aggregate in evaluation.aggregate_metrics.items():
        scored = [d[model_type] for d in evaluation.per_dimension_metrics.values() if model_type in d]
        assert scored  # every aggregate entry is backed by at least one dimension's score
        assert aggregate.n == sum(d.n for d in scored)
        assert aggregate.f1_macro == pytest.approx(sum(d.f1_macro for d in scored) / len(scored))


def test_a_dimension_where_every_category_is_selected_everywhere_is_still_scoreable(tmp_path: Path) -> None:
    """The regression guard for D-33. The grouped H/M/L prompt assigns every
    category of a dimension a level, so classifications_human.label lists
    every category on every record and category PRESENCE is constant-1. When
    that was the training target, all 5 multi-value dimensions silently
    produced no model and no metrics -- the reviewer saw only the
    topical_inclusion table and could never pick a dimension model.

    _seed_dimension_multi reproduces exactly that shape: both categories on
    every record, separated only by level."""
    engine = _db(tmp_path)
    _seed_dimension_multi(engine, n_per_category=10)

    # Precondition: presence really is constant across the seeded corpus, so
    # this test would fail against the old target rather than pass vacuously.
    labels = [label for _text, label, _ts in ml_assist._dimension_examples_full(engine, "domain")]
    assert all(set(ml_assist._parse_categories(m)) == {"chemical_engineering", "safety"} for m in labels)

    evaluation = ml_model_metrics.cumulative_dimension_metrics(engine, _DIMENSION_SCHEMA)

    assert evaluation is not None
    assert "domain" in evaluation.per_dimension_metrics


def test_a_single_value_dimension_with_a_singleton_class_is_still_scoreable(tmp_path: Path) -> None:
    """The regression guard for D-34. evidence_type was scored through
    cumulative_metrics(), which refuses the whole dimension if ANY class
    falls below MIN_EXAMPLES_PER_CLASS, while the fit it mirrored drops such
    classes and trains on the rest. One record with a one-of-a-kind label
    made the dimension permanently unscoreable while its suggestions kept
    being shown."""
    engine = _db(tmp_path)
    _seed_dimension_single(engine)
    rid = _insert_record(engine, "lonely", "A single one-off review of copilot literature")
    save_single_value_classification(engine, rid, "alice", "evidence_type", "review")

    evaluation = ml_model_metrics.cumulative_dimension_metrics(engine, _DIMENSION_SCHEMA)

    assert evaluation is not None
    assert "evidence_type" in evaluation.per_dimension_metrics
    # Scored over the filtered set the model actually trained on -- the
    # singleton class is dropped, not silently counted.
    assert all(m.n == 12 for m in evaluation.per_dimension_metrics["evidence_type"].values())


def test_dimension_scoring_diagnostics_names_every_unscoreable_dimension(tmp_path: Path) -> None:
    """A dimension that cannot be scored has to say so. Both defects above
    hid behind a bare `if evaluation is not None:`, which skipped the table,
    the file write and the model-selection prompt together -- indistinguishable
    from the feature working."""
    engine = _db(tmp_path)
    rid = _insert_record(engine, "d1", "A GenAI copilot for reactor monitoring")
    save_dimension_classification(engine, rid, "alice", "domain", {"high": ["chemical_engineering"], "medium": [], "low": []})

    reasons = dict(ml_model_metrics.dimension_scoring_diagnostics(engine, _DIMENSION_SCHEMA))

    assert set(reasons) == {"domain", "evidence_type"}
    assert "1 classified record" in reasons["domain"]
    assert str(ml_assist.MIN_TOTAL_EXAMPLES) in reasons["domain"]


def test_dimension_scoring_diagnostics_is_empty_when_everything_scores(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_dimension_multi(engine, n_per_category=10)
    _seed_dimension_single(engine)

    assert ml_model_metrics.dimension_scoring_diagnostics(engine, _DIMENSION_SCHEMA) == []


def test_dimension_scoring_diagnostics_reports_how_far_the_closest_category_is(tmp_path: Path) -> None:
    """The reviewer needs to know how much more classifying it would take,
    not merely that nothing scored."""
    engine = _db(tmp_path)
    for i in range(ml_assist.MIN_TOTAL_EXAMPLES + 2):
        rid = _insert_record(engine, f"d{i}", f"{INCLUDE_TEXTS[i % len(INCLUDE_TEXTS)]} variant {i}")
        # chemical_engineering high on 2 records only -- over MIN_EXAMPLES_PER_CLASS,
        # under MIN_POSITIVES_PER_CATEGORY, so no category can be fit.
        level = "high" if i < 2 else "low"
        save_dimension_classification(
            engine, rid, "alice", "domain", {"high": [], "medium": [], "low": [], level: ["chemical_engineering"]}
        )

    reasons = dict(ml_model_metrics.dimension_scoring_diagnostics(engine, _DIMENSION_SCHEMA))

    assert "domain" in reasons
    assert "chemical_engineering" in reasons["domain"]
    assert "2 high" in reasons["domain"]


def test_cumulative_dimension_metrics_returns_none_below_minimum(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    rid = _insert_record(engine, "d1", "A GenAI copilot for reactor monitoring")
    save_dimension_classification(engine, rid, "alice", "domain", {"high": ["chemical_engineering"], "medium": [], "low": []})

    assert ml_model_metrics.cumulative_dimension_metrics(engine, _DIMENSION_SCHEMA) is None


def _sample_dimension_batch_record(
    batch: int, active: str = "tfidf_logistic_regression"
) -> ml_model_metrics.DimensionBatchRecord:
    m = ml_model_metrics.ModelMetrics(accuracy=0.9, precision_macro=0.9, recall_macro=0.9, f1_macro=0.9, cohen_kappa=0.8, n=10)
    return ml_model_metrics.DimensionBatchRecord(
        batch=batch,
        timestamp=datetime.now(UTC).isoformat(),
        per_dimension_metrics={"domain": {active: m}},
        aggregate_metrics={active: m},
        active_dimension_model=active,
    )


def test_record_dimension_batch_writes_json_and_plot_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "dm.json"
    plot_path = tmp_path / "dm.png"

    history = ml_model_metrics.record_dimension_batch(_sample_dimension_batch_record(1), path=path, plot_path=plot_path)

    assert len(history) == 1
    assert path.exists()
    assert plot_path.exists()

    loaded = ml_model_metrics.load_dimension_history(path)
    assert len(loaded) == 1
    assert loaded[0].per_dimension_metrics["domain"]["tfidf_logistic_regression"].accuracy == 0.9
    assert loaded[0].aggregate_metrics["tfidf_logistic_regression"].accuracy == 0.9
    assert loaded[0].active_dimension_model == "tfidf_logistic_regression"


def test_record_dimension_batch_overwrites_same_batch_number_instead_of_duplicating(tmp_path: Path) -> None:
    path = tmp_path / "dm.json"
    plot_path = tmp_path / "dm.png"
    ml_model_metrics.record_dimension_batch(_sample_dimension_batch_record(1), path=path, plot_path=plot_path)
    ml_model_metrics.record_dimension_batch(
        _sample_dimension_batch_record(1, active="embedding_random_forest"), path=path, plot_path=plot_path
    )

    history = ml_model_metrics.load_dimension_history(path)
    assert len(history) == 1
    assert history[0].active_dimension_model == "embedding_random_forest"


def test_current_active_dimension_model_defaults_to_none_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "dm.json"
    assert ml_model_metrics.load_current_active_dimension_model(path) is None

    ml_model_metrics.save_current_active_dimension_model("embedding_random_forest", path)

    assert ml_model_metrics.load_current_active_dimension_model(path) == "embedding_random_forest"


def test_save_current_active_dimension_model_preserves_existing_batch_history(tmp_path: Path) -> None:
    path = tmp_path / "dm.json"
    ml_model_metrics.record_dimension_batch(_sample_dimension_batch_record(1), path=path, plot_path=tmp_path / "dm.png")

    ml_model_metrics.save_current_active_dimension_model("tfidf_decision_tree", path)

    assert ml_model_metrics.load_current_active_dimension_model(path) == "tfidf_decision_tree"
    assert len(ml_model_metrics.load_dimension_history(path)) == 1


def test_plot_dimension_history_uses_solid_for_tfidf_and_dashed_for_embedding_with_shared_classifier_color(
    tmp_path: Path, monkeypatch
) -> None:
    import matplotlib.pyplot as plt

    captured_figs = []
    monkeypatch.setattr(plt, "close", lambda fig=None: captured_figs.append(fig))

    history = [
        ml_model_metrics.DimensionBatchRecord(
            batch=1,
            timestamp=datetime.now(UTC).isoformat(),
            per_dimension_metrics={},
            aggregate_metrics={
                "tfidf_logistic_regression": ml_model_metrics.ModelMetrics(
                    accuracy=0.9, precision_macro=0.9, recall_macro=0.9, f1_macro=0.9, cohen_kappa=0.8, n=4
                ),
                "embedding_logistic_regression": ml_model_metrics.ModelMetrics(
                    accuracy=0.8, precision_macro=0.8, recall_macro=0.8, f1_macro=0.8, cohen_kappa=0.6, n=4
                ),
            },
            active_dimension_model="tfidf_logistic_regression",
        )
    ]

    ml_model_metrics.plot_dimension_history(history, tmp_path / "dm.png")

    assert len(captured_figs) == 1
    lines_by_label = {line.get_label(): line for line in captured_figs[0].axes[0].get_lines()}
    tfidf_line = lines_by_label["tfidf logistic regression"]
    embedding_line = lines_by_label["embedding logistic regression"]
    assert tfidf_line.get_linestyle() == "-"
    assert embedding_line.get_linestyle() == "--"
    assert tfidf_line.get_color() == embedding_line.get_color()


def _seed_topical_and_dimension_chronology(engine) -> None:
    """24 topical_inclusion decisions (minutes 0-23, alternating include/
    exclude -- same shape _seed_chronological_decisions uses, defining the
    cutoff timestamps backfill_dimension_batches replays) where every
    'include' one also gets a domain + evidence_type classification a
    minute later, alternating between two labels each so both classes are
    present for backfill to actually fit something -- dimension data trails
    topical data by construction, the way it does in real review sessions."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    classifications_human = reflect_table(engine, "classifications_human")

    def _insert_dimension_row(record_id: str, dimension: str, label: str, minute: int) -> None:
        with engine.begin() as conn:
            conn.execute(
                classifications_human.insert().values(
                    classification_id=new_id(), record_id=record_id, dimension=dimension, label=label,
                    confidence=None, coder="alice", timestamp=base.replace(minute=min(minute, 59)).isoformat(),
                )
            )

    minute = 0
    include_index = 0
    for _rep in range(3):
        for inc_title, exc_title in zip(INCLUDE_TEXTS, EXCLUDE_TEXTS, strict=True):
            inc_id = f"inc{minute}"
            _insert_human_decision(engine, _insert_record(engine, inc_id, inc_title), "include", base.replace(minute=minute).isoformat())
            if include_index % 2 == 0:
                domain_label, evidence_label = json.dumps(["chemical_engineering"]), "prototype"
            else:
                domain_label, evidence_label = json.dumps(["safety"]), "conceptual_discussion"
            _insert_dimension_row(inc_id, "domain", domain_label, minute + 1)
            _insert_dimension_row(inc_id, "evidence_type", evidence_label, minute + 1)
            include_index += 1
            minute += 1

            exc_id = f"exc{minute}"
            _insert_human_decision(engine, _insert_record(engine, exc_id, exc_title), "exclude", base.replace(minute=minute).isoformat())
            minute += 1


def test_backfill_dimension_batches_reconstructs_history_and_is_idempotent(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_topical_and_dimension_chronology(engine)
    path = tmp_path / "dm.json"
    plot_path = tmp_path / "dm.png"

    first = ml_model_metrics.backfill_dimension_batches(engine, _DIMENSION_SCHEMA, batch_size=4, path=path, plot_path=plot_path)
    second = ml_model_metrics.backfill_dimension_batches(engine, _DIMENSION_SCHEMA, batch_size=4, path=path, plot_path=plot_path)

    assert first  # at least one checkpoint had enough dimension data by the time it fires
    assert [b.batch for b in first] == list(range(1, len(first) + 1))  # densely, independently numbered
    for b in first:
        assert b.aggregate_metrics
        assert b.active_dimension_model == ml_assist.DEFAULT_ACTIVE_MODEL
    assert [b.batch for b in first] == [b.batch for b in second]  # idempotent, overwrites not appends
    assert len(ml_model_metrics.load_dimension_history(path)) == len(second)
    assert path.exists()
    assert plot_path.exists()
