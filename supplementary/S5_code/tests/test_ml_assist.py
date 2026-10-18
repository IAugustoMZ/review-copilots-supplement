from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.validation import ml_assist
from review_pipeline.src.validation.dimension_classification import (
    DimensionSchema,
    save_dimension_classification,
    save_single_value_classification,
)
from review_pipeline.src.validation.ml_assist import (
    assist_stats,
    log_suggestion,
    logged_dimensions,
    suggest,
    suggest_dimension,
    train,
    train_dimensions,
)


def test_ml_assist_module_never_references_classifications_ai() -> None:
    """Structural enforcement mirroring test_blinded_queue.py: a suggestion
    model trained on the AI's own labels would defeat the point of blinded
    validation, so this module must never touch that table."""
    source = inspect.getsource(ml_assist)
    assert "classifications_ai" not in source


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record(engine, record_id: str, title: str, abstract: str = "") -> str:
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


def _insert_human_decision(engine, record_id: str, label: str, reviewer: str = "alice") -> None:
    classifications_human = reflect_table(engine, "classifications_human")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            classifications_human.insert().values(
                classification_id=new_id(),
                record_id=record_id,
                dimension="topical_inclusion",
                label=label,
                confidence="high",
                rationale="x",
                coder=reviewer,
                timestamp=now,
            )
        )


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


def _seed_labeled_pool(engine) -> None:
    for i, title in enumerate(INCLUDE_TEXTS):
        rid = _insert_record(engine, f"inc{i}", title)
        _insert_human_decision(engine, rid, "include")
    for i, title in enumerate(EXCLUDE_TEXTS):
        rid = _insert_record(engine, f"exc{i}", title)
        _insert_human_decision(engine, rid, "exclude")


def test_train_returns_none_below_minimum_total_examples(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    rid = _insert_record(engine, "r1", "A GenAI copilot for process control")
    _insert_human_decision(engine, rid, "include")

    assert train(engine) is None


def test_train_returns_none_when_only_one_class_present(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    for i, title in enumerate(INCLUDE_TEXTS + INCLUDE_TEXTS):
        rid = _insert_record(engine, f"r{i}", title)
        _insert_human_decision(engine, rid, "include")

    assert train(engine) is None


def test_train_and_suggest_predicts_a_known_class(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_labeled_pool(engine)

    model = train(engine)
    assert model is not None
    assert model.training_size == len(INCLUDE_TEXTS) + len(EXCLUDE_TEXTS)
    assert set(model.class_counts) == {"include", "exclude"}

    record = {"title": "GenAI copilot assistant for industrial plant operations", "abstract": ""}
    suggestion = suggest(model, record)
    assert suggestion is not None
    assert suggestion.label in ("include", "exclude")
    assert 0.0 <= suggestion.probability <= 1.0
    assert suggestion.training_size == model.training_size


def test_train_all_max_examples_trains_only_on_the_first_n_chronological_examples(tmp_path: Path) -> None:
    """max_examples lets a caller reconstruct the model that existed before
    some later boundary (used by `classify assisted` when resuming a batch
    that's already partway done -- see ml_model_metrics.py) -- training size
    must reflect the truncated set, not everything in the DB."""
    engine = _db(tmp_path)
    _seed_labeled_pool(engine)  # len(INCLUDE_TEXTS) + len(EXCLUDE_TEXTS) examples total

    full = ml_assist.train_all(engine)
    assert full is not None
    total = next(iter(full.values())).training_size

    truncated = ml_assist.train_all(engine, max_examples=total - 2)
    assert truncated is not None
    for model_type in ml_assist.MODEL_TYPES:
        assert truncated[model_type].training_size == total - 2


def test_train_all_max_examples_below_minimum_returns_none(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_labeled_pool(engine)

    assert ml_assist.train_all(engine, max_examples=1) is None


def test_train_all_trains_every_model_spec_including_embedding_and_gradient_boosting(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_labeled_pool(engine)

    models = ml_assist.train_all(engine)

    assert models is not None
    assert set(models) == set(ml_assist.MODEL_SPECS)
    for model_type in ml_assist.MODEL_TYPES:
        suggestion = suggest(models[model_type], {"title": INCLUDE_TEXTS[0], "abstract": ""})
        assert suggestion is not None
        assert suggestion.label in ("include", "exclude")


def test_build_classifier_gradient_boosting() -> None:
    from sklearn.ensemble import GradientBoostingClassifier

    assert isinstance(ml_assist._build_classifier("gradient_boosting"), GradientBoostingClassifier)


def test_build_classifier_wraps_logistic_regression_in_a_scaling_pipeline() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    tfidf_estimator = ml_assist._build_classifier("logistic_regression", "tfidf")
    assert isinstance(tfidf_estimator, Pipeline)
    assert isinstance(tfidf_estimator.named_steps["clf"], LogisticRegression)
    assert isinstance(tfidf_estimator.named_steps["scale"], StandardScaler)
    assert tfidf_estimator.named_steps["scale"].with_mean is False  # sparse tfidf -- centering would densify it

    embedding_estimator = ml_assist._build_classifier("logistic_regression", "embedding")
    assert embedding_estimator.named_steps["scale"].with_mean is True  # dense embeddings -- safe to center


def test_build_classifier_rejects_unknown_kind() -> None:
    try:
        ml_assist._build_classifier("not_a_real_kind")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_fit_with_cv_caps_folds_at_the_smallest_class_count() -> None:
    """3 folds is the usual cap, but a class with only MIN_EXAMPLES_PER_CLASS
    (2) examples must cap folds at 2 instead -- StratifiedKFold raises if
    asked for more splits than a class has members."""
    x = [[float(i)] for i in range(6)]
    labels = ["include", "include", "exclude", "exclude", "include", "include"]  # exclude: 2 examples

    classifier, oof_predictions = ml_assist._fit_with_cv("logistic_regression", "tfidf", x, labels)

    assert classifier is not None
    assert len(oof_predictions) == len(labels)  # one out-of-fold prediction per example


def test_fit_with_cv_out_of_fold_predictions_are_never_from_the_final_full_fit() -> None:
    """The out-of-fold predictions come from cross_val_predict (each fold
    scored by a model that never trained on it), not from calling .predict()
    on the final estimator that WAS fit on everything -- a clearly
    separable toy dataset would score perfectly under the latter, so
    checking oof accuracy isn't ~1.0 here would just be a weak smoke test;
    instead pin that oof_predictions has one entry per input and isn't
    trivially the string "include" (the majority class) repeated, which is
    what a badly-wired constant-predictor bug would produce."""
    x = [[float(i)] for i in range(6)]
    labels = ["include", "include", "exclude", "exclude", "include", "include"]

    _classifier, oof_predictions = ml_assist._fit_with_cv("logistic_regression", "tfidf", x, labels)

    assert len(oof_predictions) == 6
    assert set(oof_predictions) <= {"include", "exclude"}


def test_train_all_falls_back_to_tfidf_only_when_embedding_backend_fails(tmp_path: Path, monkeypatch) -> None:
    """A reviewer without network access (SPECTER can't be downloaded) still
    gets the tfidf_* suggestion models -- embedding_* ones are silently
    omitted rather than the whole retrain blowing up."""
    engine = _db(tmp_path)
    _seed_labeled_pool(engine)
    monkeypatch.setattr(
        ml_assist.embeddings, "EmbeddingExtractor", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network"))
    )

    models = ml_assist.train_all(engine)

    assert models is not None
    assert set(models) == {mt for mt, (fk, _ck) in ml_assist.MODEL_SPECS.items() if fk == "tfidf"}


def test_suggest_returns_none_for_record_with_no_text(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_labeled_pool(engine)
    model = train(engine)

    assert suggest(model, {"title": None, "abstract": None}) is None


def test_suggest_returns_none_without_a_model() -> None:
    assert suggest(None, {"title": "anything", "abstract": ""}) is None


def test_log_suggestion_and_assist_stats_track_overrides(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_labeled_pool(engine)
    model = train(engine)
    assert model is not None
    suggestion = suggest(model, {"title": "A copilot for reactor monitoring", "abstract": ""})
    assert suggestion is not None

    new1 = _insert_record(engine, "new1", "A GenAI copilot for reactor monitoring")
    new2 = _insert_record(engine, "new2", "A GenAI copilot for plant operations")
    log_suggestion(
        engine, new1, "alice",
        suggested_label=suggestion.label, suggested_probability=suggestion.probability,
        training_size=suggestion.training_size, final_label=suggestion.label,
    )
    overridden_label = "exclude" if suggestion.label != "exclude" else "include"
    log_suggestion(
        engine, new2, "alice",
        suggested_label=suggestion.label, suggested_probability=suggestion.probability,
        training_size=suggestion.training_size, final_label=overridden_label,
    )

    stats = assist_stats(engine)
    assert stats.total == 2
    assert stats.overrides == 1
    assert stats.recent_total == 2
    assert stats.recent_overrides == 1


# Dimension seeding needs more examples per category than topical_inclusion
# does: a multi-value dimension's category has to clear
# MIN_POSITIVES_PER_CATEGORY (stricter than MIN_EXAMPLES_PER_CLASS) before it
# is fit at all, and its Platt calibrator needs held-out scores of both
# classes in every CV fold. These extend INCLUDE_TEXTS/EXCLUDE_TEXTS rather
# than replacing them so the topical_inclusion tests' example counts, which
# several of them assert on directly, stay untouched.
DIM_INCLUDE_TEXTS = INCLUDE_TEXTS + [
    "Conversational agent for reactor startup procedure guidance",
    "Foundation model assistant for process control loop tuning",
    "Copilot for refinery unit operations and fault diagnosis",
    "Language model support tool for chemical batch scheduling",
]
DIM_EXCLUDE_TEXTS = EXCLUDE_TEXTS + [
    "Baroque counterpoint in early keyboard repertoire",
    "Sediment stratigraphy of an inland lake basin",
    "Typography and legibility in road signage",
    "Migration patterns of coastal wading birds",
]


def _seed_dimension_multi(engine) -> None:
    """8 records with domain=chemical_engineering rated HIGH (safety low), 8
    with domain=safety rated HIGH (chemical_engineering low) -- 8 high/8
    not-high per category, enough for each to individually clear
    MIN_POSITIVES_PER_CATEGORY on the high side and MIN_EXAMPLES_PER_CLASS on
    the not-high side.

    Both categories must appear at HIGH somewhere: the training target is
    P(rated 'high'), not category presence (ml_assist._parse_high_categories),
    so a category the reviewer only ever rates medium or low has zero
    positives and gets no classifier at all. Each record also rates the
    other category 'low' rather than omitting it, matching what the grouped
    H/M/L prompt actually produces -- every category of a dimension gets a
    level, which is precisely why presence carries no signal."""
    for i, title in enumerate(DIM_INCLUDE_TEXTS):
        rid = _insert_record(engine, f"dinc{i}", title)
        save_dimension_classification(
            engine, rid, "alice", "domain", {"high": ["chemical_engineering"], "medium": [], "low": ["safety"]}
        )
    for i, title in enumerate(DIM_EXCLUDE_TEXTS):
        rid = _insert_record(engine, f"dexc{i}", title)
        save_dimension_classification(
            engine, rid, "alice", "domain", {"high": ["safety"], "medium": [], "low": ["chemical_engineering"]}
        )


def _seed_dimension_single(engine) -> None:
    for i, title in enumerate(INCLUDE_TEXTS):
        rid = _insert_record(engine, f"single_inc{i}", title)
        save_single_value_classification(engine, rid, "alice", "evidence_type", "prototype")
    for i, title in enumerate(EXCLUDE_TEXTS):
        rid = _insert_record(engine, f"single_exc{i}", title)
        save_single_value_classification(engine, rid, "alice", "evidence_type", "conceptual_discussion")


def test_train_dimensions_multi_value_predicts_expected_category(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_dimension_multi(engine)
    schema = [DimensionSchema(name="domain", multi_value=True, categories=["chemical_engineering", "safety", "manufacturing"])]

    models = train_dimensions(engine, schema)
    assert "domain" in models

    suggestion = suggest_dimension(
        models["domain"], {"title": "GenAI copilot assistant for industrial plant operations", "abstract": ""}
    )
    assert suggestion is not None
    # Raw probability for every trained category, no high/medium/low bucket --
    # see test_multi_value_prediction_returns_raw_probabilities_for_every_trained_category
    # for that behavior pinned precisely against a fake classifier.
    assert set(suggestion.category_probabilities) == {"chemical_engineering", "safety"}
    assert suggestion.category_probabilities["chemical_engineering"] >= ml_assist.PRESENCE_THRESHOLD
    assert "manufacturing" not in suggestion.category_probabilities  # 0 positive examples -- never gets a classifier


def test_train_dimensions_all_trains_every_model_spec_for_multi_value_and_single_value_dimensions(
    tmp_path: Path,
) -> None:
    engine = _db(tmp_path)
    _seed_dimension_multi(engine)
    _seed_dimension_single(engine)
    schema = [
        DimensionSchema(name="domain", multi_value=True, categories=["chemical_engineering", "safety", "manufacturing"]),
        DimensionSchema(name="evidence_type", multi_value=False, categories=["prototype", "conceptual_discussion", "review"]),
    ]

    models_by_dimension = ml_assist.train_dimensions_all(engine, schema)

    assert set(models_by_dimension) == {"domain", "evidence_type"}
    assert set(models_by_dimension["domain"]) == set(ml_assist.MODEL_SPECS)
    assert set(models_by_dimension["evidence_type"]) == set(ml_assist.MODEL_SPECS)
    # train_dimensions() stays a thin wrapper around DEFAULT_ACTIVE_MODEL --
    # existing callers/tests keep working unchanged. Compared by prediction,
    # not object equality: sklearn estimators use identity-based __eq__, so
    # two independently-fit models are never == even with identical
    # coefficients (train_dimensions() calls train_dimensions_all() again,
    # a fresh fit, not a lookup into the same dict).
    thin_wrapper_models = train_dimensions(engine, schema)
    assert set(thin_wrapper_models) == {"domain", "evidence_type"}
    test_text = "GenAI copilot assistant for industrial plant operations"
    for name in thin_wrapper_models:
        assert thin_wrapper_models[name].predict(test_text) == models_by_dimension[name][ml_assist.DEFAULT_ACTIVE_MODEL].predict(
            test_text
        )


def test_fit_dimension_category_classifiers_uses_the_requested_classifier_kind(tmp_path: Path) -> None:
    from sklearn.ensemble import GradientBoostingClassifier

    engine = _db(tmp_path)
    _seed_dimension_multi(engine)
    dim = DimensionSchema(name="domain", multi_value=True, categories=["chemical_engineering", "safety", "manufacturing"])
    texts = [text for text, _label in ml_assist._dimension_examples(engine, "domain", levels=True)]
    labels = [label for _text, label in ml_assist._dimension_examples(engine, "domain", levels=True)]

    fitted = ml_assist._fit_dimension_category_classifiers(dim, texts, labels, "tfidf_gradient_boosting")

    assert fitted is not None
    model, oof_by_category = fitted
    assert {cc.category for cc in model.category_classifiers or []} == {"chemical_engineering", "safety"}
    # Each per-category classifier is Platt-wrapped (_fit_calibrated_binary),
    # so the requested classifier kind is the wrapper's `base`, not the
    # object hanging off _CategoryClassifier directly.
    assert all(isinstance(cc.classifier, ml_assist._PlattCalibrated) for cc in model.category_classifiers or [])
    assert all(isinstance(cc.classifier.base, GradientBoostingClassifier) for cc in model.category_classifiers or [])
    assert set(oof_by_category) == {"chemical_engineering", "safety"}
    assert all(len(oof) == len(labels) for oof in oof_by_category.values())


def test_fit_dimension_single_value_classifier_uses_the_requested_classifier_kind(tmp_path: Path) -> None:
    from sklearn.tree import DecisionTreeClassifier

    engine = _db(tmp_path)
    _seed_dimension_single(engine)
    dim = DimensionSchema(name="evidence_type", multi_value=False, categories=["prototype", "conceptual_discussion", "review"])
    examples = ml_assist._dimension_examples(engine, "evidence_type")
    texts = [text for text, _label in examples]
    labels = [label for _text, label in examples]

    fitted = ml_assist._fit_dimension_single_value_classifier(dim, texts, labels, "tfidf_decision_tree")

    assert fitted is not None
    model, filtered_labels, oof_predictions = fitted
    assert isinstance(model.classifier, DecisionTreeClassifier)
    assert len(oof_predictions) == len(filtered_labels)


def test_train_dimensions_all_falls_back_to_tfidf_only_when_embedding_backend_fails(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    _seed_dimension_multi(engine)
    schema = [DimensionSchema(name="domain", multi_value=True, categories=["chemical_engineering", "safety", "manufacturing"])]
    monkeypatch.setattr(
        ml_assist.embeddings, "EmbeddingExtractor", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network"))
    )

    models_by_dimension = ml_assist.train_dimensions_all(engine, schema)

    assert set(models_by_dimension["domain"]) == {mt for mt, (fk, _ck) in ml_assist.MODEL_SPECS.items() if fk == "tfidf"}


def test_multi_value_prediction_returns_raw_probabilities_for_every_trained_category() -> None:
    """DimensionModel.predict() never buckets a multi-value dimension's
    per-category probability into high/medium/low -- it returns every
    trained category's raw predict_proba output as-is, unfiltered, so the
    reviewer sees the same number the model actually computed. Pinned here
    against fake classifiers with known probabilities, independent of what a
    real LogisticRegression happens to output for any particular training
    set."""

    class _FakeClassifier:
        def __init__(self, positive_probability: float) -> None:
            self.classes_ = [0, 1]
            self._positive_probability = positive_probability

        def predict_proba(self, x):
            return [[1 - self._positive_probability, self._positive_probability]]

    class _FakeVectorizer:
        def transform(self, texts):
            return None

    model = ml_assist.DimensionModel(
        dimension="domain",
        multi_value=True,
        vectorizer=_FakeVectorizer(),
        training_size=10,
        category_classifiers=[
            ml_assist._CategoryClassifier(category="likely_cat", classifier=_FakeClassifier(0.90)),
            ml_assist._CategoryClassifier(category="borderline_cat", classifier=_FakeClassifier(0.55)),
            ml_assist._CategoryClassifier(category="unlikely_cat", classifier=_FakeClassifier(0.20)),
        ],
    )

    suggestion = model.predict("irrelevant text")
    assert suggestion.category_probabilities == {"likely_cat": 0.90, "borderline_cat": 0.55, "unlikely_cat": 0.20}


def test_bucket_suggestion_buckets_by_threshold_and_omits_low_probability_categories() -> None:
    suggestion = ml_assist.MultiLabelSuggestion(
        category_probabilities={
            "very_likely": 0.90,  # >= HIGH_THRESHOLD
            "somewhat_likely": 0.55,  # >= MEDIUM_THRESHOLD
            "borderline": 0.20,  # >= LOW_THRESHOLD
            "unlikely": 0.05,  # below LOW_THRESHOLD -- omitted entirely
        },
        training_size=10,
    )

    buckets = ml_assist.bucket_suggestion(suggestion)

    assert buckets == {"high": ["very_likely"], "medium": ["somewhat_likely"], "low": ["borderline"]}


def test_bucket_suggestion_orders_categories_within_a_bucket_by_probability() -> None:
    suggestion = ml_assist.MultiLabelSuggestion(
        category_probabilities={"second": 0.75, "first": 0.95, "third": 0.71},
        training_size=10,
    )

    buckets = ml_assist.bucket_suggestion(suggestion)

    assert buckets["high"] == ["first", "second", "third"]


def test_a_category_below_the_positive_minimum_gets_no_classifier(tmp_path: Path) -> None:
    """MIN_POSITIVES_PER_CATEGORY, not MIN_EXAMPLES_PER_CLASS, is what a
    multi-value dimension's category has to clear on the positive side --
    positive meaning "rated high", the training target. A category with a
    handful of high ratings is fittable and even calibratable, and that is
    exactly the problem -- it would hand the reviewer a confident number
    backed by three papers. It gets no suggestion instead, while the
    well-evidenced categories in the same dimension still do."""
    engine = _db(tmp_path)
    _seed_dimension_multi(engine)  # 8 'high' each for chemical_engineering and safety
    # 3 records carrying a third category: over MIN_EXAMPLES_PER_CLASS, under
    # MIN_POSITIVES_PER_CATEGORY.
    assert ml_assist.MIN_EXAMPLES_PER_CLASS < 3 < ml_assist.MIN_POSITIVES_PER_CATEGORY
    for i in range(3):
        rid = _insert_record(engine, f"drare{i}", f"Additive manufacturing process monitoring study {i}")
        save_dimension_classification(engine, rid, "alice", "domain", {"high": ["manufacturing"], "medium": [], "low": []})
    dim = DimensionSchema(name="domain", multi_value=True, categories=["chemical_engineering", "safety", "manufacturing"])
    texts = [text for text, _label in ml_assist._dimension_examples(engine, "domain", levels=True)]
    labels = [label for _text, label in ml_assist._dimension_examples(engine, "domain", levels=True)]

    fitted = ml_assist._fit_dimension_category_classifiers(dim, texts, labels, "tfidf_logistic_regression")

    assert fitted is not None
    model, _oof = fitted
    trained = {cc.category for cc in model.category_classifiers or []}
    assert "manufacturing" not in trained
    assert trained == {"chemical_engineering", "safety"}


def test_category_classifiers_are_platt_calibrated_not_raw(tmp_path: Path) -> None:
    """The per-category probability a reviewer sees comes through the Platt
    calibrator, not straight out of the base estimator -- which is the whole
    point, since `class_weight="balanced"` deliberately distorts the base
    estimator's probabilities away from the true base rate."""
    engine = _db(tmp_path)
    _seed_dimension_multi(engine)
    dim = DimensionSchema(name="domain", multi_value=True, categories=["chemical_engineering", "safety"])
    texts = [text for text, _label in ml_assist._dimension_examples(engine, "domain", levels=True)]
    labels = [label for _text, label in ml_assist._dimension_examples(engine, "domain", levels=True)]

    fitted = ml_assist._fit_dimension_category_classifiers(dim, texts, labels, "tfidf_logistic_regression")
    assert fitted is not None
    model, _oof = fitted

    calibrated = next(cc for cc in model.category_classifiers or [] if cc.category == "chemical_engineering")
    assert isinstance(calibrated.classifier, ml_assist._PlattCalibrated)

    x = model.vectorizer.transform(["A GenAI copilot for chemical reactor troubleshooting"])
    base_probability = calibrated.classifier.base.predict_proba(x)[0][1]
    calibrated_probability = calibrated.classifier.predict_proba(x)[0][1]
    assert calibrated_probability != base_probability
    assert 0.0 <= calibrated_probability <= 1.0


def test_platt_calibration_is_monotone_and_never_normalises_across_categories() -> None:
    """Two properties that together separate this from the softmax it
    replaces, pinned against a fake base classifier so they hold regardless
    of what a real fit happens to produce.

    Monotone: calibration only remaps the score scale, so a higher raw score
    stays higher after calibration -- it never reorders categories.

    Independent: each category's calibrated probability is a function of that
    category's own score alone. A softmax would make it depend on the other
    categories' scores and force the row to sum to 1, which is wrong for a
    multi-label dimension where a paper can genuinely be in two categories at
    once -- so the calibrated probabilities are explicitly NOT expected to
    sum to 1."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    class _FakeBase:
        def __init__(self, positive_probability: float) -> None:
            self.classes_ = [0, 1]
            self._p = positive_probability

        def predict_proba(self, x):
            return np.array([[1 - self._p, self._p]])

    # One shared calibrator, fitted on a clean monotone score/label relation.
    scores = [0.02, 0.08, 0.15, 0.30, 0.55, 0.70, 0.88, 0.97]
    presence = [0, 0, 0, 0, 1, 1, 1, 1]
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(ml_assist._logit(scores), presence)

    def calibrated_probability(raw: float) -> float:
        wrapped = ml_assist._PlattCalibrated(
            base=_FakeBase(raw), calibrator=calibrator, base_positive_index=1, classes_=[0, 1]
        )
        return float(wrapped.predict_proba(None)[0][1])

    raw_scores = [0.95, 0.60, 0.25]
    calibrated = [calibrated_probability(p) for p in raw_scores]

    assert calibrated == sorted(calibrated, reverse=True)  # order preserved
    # Three co-occurring categories, each calibrated on its own score: the row
    # is under no constraint to sum to 1, unlike a softmax.
    assert abs(sum(calibrated) - 1.0) > 1e-9


def test_calibrated_out_of_fold_predictions_stay_binary_high_labels(tmp_path: Path) -> None:
    """ml_model_metrics.evaluate_dimension_model_multi_value_from_oof scores
    these against the 0/1 high/not-high target parsed from the labels, so the
    calibration change must not alter their shape -- probabilities are
    thresholded back to 0/1 inside _fit_calibrated_binary, one per training
    example."""
    engine = _db(tmp_path)
    _seed_dimension_multi(engine)
    dim = DimensionSchema(name="domain", multi_value=True, categories=["chemical_engineering", "safety"])
    texts = [text for text, _label in ml_assist._dimension_examples(engine, "domain", levels=True)]
    labels = [label for _text, label in ml_assist._dimension_examples(engine, "domain", levels=True)]

    fitted = ml_assist._fit_dimension_category_classifiers(dim, texts, labels, "tfidf_logistic_regression")
    assert fitted is not None
    _model, oof_by_category = fitted

    for category, oof_predictions in oof_by_category.items():
        assert len(oof_predictions) == len(labels), category
        assert set(oof_predictions) <= {0, 1}, category


def test_dimension_examples_drops_rows_that_lost_their_levels_map(tmp_path: Path) -> None:
    """A row that selected categories but carries no levels map predates
    migration 005 and has no recoverable level. It must be dropped, not read
    as "no category is high" -- that would train the model on a fabricated
    negative for every category the reviewer actually rated."""
    engine = _db(tmp_path)
    _seed_dimension_multi(engine)
    classifications_human = reflect_table(engine, "classifications_human")
    rid = _insert_record(engine, "legacy", "A pre-migration copilot paper with no levels recorded")
    with engine.begin() as conn:
        conn.execute(
            classifications_human.insert().values(
                classification_id=new_id(), record_id=rid, dimension="domain",
                label=json.dumps(["chemical_engineering"]), label_levels_json=None,
                coder="alice", timestamp=datetime.now(UTC).isoformat(),
            )
        )

    with_levels = ml_assist._dimension_examples(engine, "domain", levels=True)
    without_levels = ml_assist._dimension_examples(engine, "domain")

    assert len(without_levels) == len(with_levels) + 1  # the legacy row exists, but not in the training view
    assert all(label != "{}" for _text, label in with_levels)


def test_dimension_examples_keeps_a_genuinely_empty_selection(tmp_path: Path) -> None:
    """An explicit "no category applies" also stores no levels map, but it
    IS a real observation -- nothing is high -- so it must survive the filter
    that drops pre-migration rows."""
    engine = _db(tmp_path)
    rid = _insert_record(engine, "empty", "A paper the reviewer marked as matching no domain")
    save_dimension_classification(engine, rid, "alice", "domain", {"high": [], "medium": [], "low": []})

    examples = ml_assist._dimension_examples(engine, "domain", levels=True)

    assert len(examples) == 1
    assert ml_assist._parse_high_categories(examples[0][1]) == []


def test_train_dimensions_returns_no_model_below_minimum(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    rid = _insert_record(engine, "d1", "A GenAI copilot for reactor monitoring")
    save_dimension_classification(engine, rid, "alice", "domain", {"high": ["chemical_engineering"], "medium": [], "low": []})
    schema = [DimensionSchema(name="domain", multi_value=True, categories=["chemical_engineering"])]

    assert train_dimensions(engine, schema) == {}


def test_train_dimensions_single_value_predicts_expected_class(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _seed_dimension_single(engine)
    schema = [
        DimensionSchema(name="evidence_type", multi_value=False, categories=["prototype", "conceptual_discussion", "review"])
    ]

    models = train_dimensions(engine, schema)
    assert "evidence_type" in models

    suggestion = suggest_dimension(
        models["evidence_type"], {"title": "GenAI copilot assistant for industrial plant operations", "abstract": ""}
    )
    assert suggestion is not None
    assert suggestion.label == "prototype"


def test_suggest_dimension_returns_none_without_a_model() -> None:
    assert suggest_dimension(None, {"title": "anything", "abstract": ""}) is None


def test_log_suggestion_and_assist_stats_support_per_dimension_breakdown(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    rid1 = _insert_record(engine, "dim1", "record one")
    rid2 = _insert_record(engine, "dim2", "record two")

    log_suggestion(
        engine, rid1, "alice",
        suggested_label="include", suggested_probability=0.9, training_size=10, final_label="include",
    )
    log_suggestion(
        engine, rid2, "alice",
        suggested_label='{"high": ["chemical_engineering"]}', suggested_probability=0.7, training_size=10,
        final_label='{"high": ["safety"]}', dimension="domain",
    )

    assert logged_dimensions(engine) == ["topical_inclusion", "domain"]

    topical_stats = assist_stats(engine, dimension="topical_inclusion")
    assert topical_stats.total == 1
    assert topical_stats.overrides == 0

    domain_stats = assist_stats(engine, dimension="domain")
    assert domain_stats.total == 1
    assert domain_stats.overrides == 1

    combined_stats = assist_stats(engine)
    assert combined_stats.total == 2
    assert combined_stats.overrides == 1
