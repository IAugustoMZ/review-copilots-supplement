"""Phase 11 semi-supervised assist: a weak classical-ML suggestion for the
topical_inclusion decision and the 6-dimension classification step, to speed
up the human required-review pass without weakening it. The human reviewer
still makes and records every decision -- `classify assisted`
(src/cli/human_review_commands.py) just pre-fills a suggestion the reviewer
accepts (Enter) or overrides (c), same as a second opinion, not a
replacement.

Deliberately trained ONLY on human-recorded decisions -- this module never
reflects or queries the DB table that stores AI-generated screening labels,
full stop, checked the same way tests/test_blinded_queue.py checks
blinded_queue.py (see tests/test_ml_assist.py; that table name is
deliberately not spelled out in this docstring, so the grep only matches real
code references). Training this on the Phase 10 AI screening labels would
just launder the AI's own decisions back onto the human review screen under a
different name, defeating the entire point of blinded validation: measuring
a genuine, independent human-AI agreement rate in Phase 12. The only signal
these models ever see is record text (title/abstract) and prior human
decisions.

Models: 4 classifier families (logistic regression, decision tree, random
forest, gradient boosting -- scikit-learn) each over 2 feature
representations (TF-IDF over title+abstract; SPECTER sentence embeddings,
see src/validation/embeddings.py) -- 8 combinations total, see MODEL_SPECS.
Every one goes through a basic cross-validation + hyperparameter-selection
pass before being scored (_fit_with_cv). logistic_regression is additionally
scaled (StandardScaler, see _build_classifier) -- unlike the tree-based
classifiers, its L2 penalty and gradient-based solver are scale-sensitive.
Reported metrics (src/validation/ml_model_metrics.py) come from the same
fit's out-of-fold cross-validation predictions (_fit_with_cv's
cross_val_predict over the same StratifiedKFold used for hyperparameter
selection) over the full cumulative labeled set, not from the small
~--batch-size decisions of whichever batch just finished, and not from a
single random holdout split either -- both are noisy: a batch is small and,
given how imbalanced human decisions are, easily unrepresentative on its
own, and a fresh 80/20 split drawn every retrain is effectively an
independent random sample each time rather than a converging estimate.
Retrained from scratch each time a
train function is called -- the labeled set here tops out in the hundreds of
rows, so there's no reason to persist or incrementally update a model file
between sessions -- but unlike the original single-model version, a full
retrain now takes real seconds (randomized CV search x 8 models, plus
SPECTER encoding for embedding-backed ones), not milliseconds.

Two kinds of suggestion, both now compared across the same MODEL_SPECS grid:
- topical_inclusion (train_all/train/suggest): single-value, 2-3 classes.
- each of the 6 classification dimensions (train_dimensions_all/
  train_dimensions/suggest_dimension):
  - multi_value dimensions (domain, technology, application,
    architectural_role, validation_focus): one independent binary
    high/not-high classifier per category (a paper can be 'high' on more
    than one) per model_type (_fit_dimension_category_classifiers), each
    Platt-calibrated against its own out-of-fold scores
    (_fit_calibrated_binary) and reporting that calibrated probability.
    The target is P(the reviewer put this category in 'high'), read from
    classifications_human.label_levels_json -- NOT presence in
    classifications_human.label. Presence carries no signal at all under the
    grouped H/M/L entry design (dimension_classification.py): the reviewer
    assigns every category of a dimension to one of high/medium/low, so
    `label` lists every category on every record and the presence target is
    constant-1. See _parse_high_categories.
    Calibration is what makes the per-category numbers comparable to each
    other and to the absolute bucket cutoffs below; it is explicitly NOT a
    cross-category softmax, which would be wrong for multi-label dimensions
    -- see _PlattCalibrated. A category needs
    MIN_POSITIVES_PER_CATEGORY positives before it is fit at all; below that
    it makes no suggestion rather than a confident one backed by three
    papers. bucket_suggestion() turns those calibrated probabilities into a
    single proposed high/medium/low classification (see HIGH_THRESHOLD/
    MEDIUM_THRESHOLD/LOW_THRESHOLD) that the reviewer accepts as a whole
    (Enter) or rejects in favor of the normal manual prompt (c) -- same
    accept/override shape as every other item, not a per-category
    confirmation.
  - evidence_type: single-value, same shape as topical_inclusion per
    model_type (_fit_dimension_single_value_classifier), trained only on
    categories with enough examples (rare categories are dropped from
    training, not from the reviewer's manual option list).

Unlike topical_inclusion (one active model type for that one decision), the
6 dimensions share a single active model type applied uniformly to all of
them (`classify assisted`'s active_dimension_model_type) -- picked from each
model type's holdout performance averaged ACROSS the 6 dimensions
(src/validation/ml_model_metrics.py::cumulative_dimension_metrics), not one
choice per dimension.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeAlias

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import Engine, select

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.validation import embeddings
from review_pipeline.src.validation.dimension_classification import DimensionSchema

# Below these thresholds a fitted model would just be noise dressed up as a
# suggestion -- callers fall back to the unassisted review loop instead.
MIN_TOTAL_EXAMPLES = 6
MIN_EXAMPLES_PER_CLASS = 2

# Positive examples a multi-value dimension's category needs before it gets
# its own presence/absence classifier (_fit_dimension_category_classifiers).
# Deliberately stricter than MIN_EXAMPLES_PER_CLASS, which still governs the
# negative side and every single-value fit: a category with 2-3 positives can
# be fit, and can even be calibrated, but neither the fit nor the calibration
# curve means anything at that size -- the reviewer would be shown a confident
# number backed by three papers. Below this the category simply gets no
# suggestion; it stays on the reviewer's manual option list, exactly like a
# category the model has never seen. Early batches therefore suggest only the
# handful of categories that have actually accumulated evidence, which is the
# intended behaviour: silence beats calibrated noise.
MIN_POSITIVES_PER_CATEGORY = 8

Classifier: TypeAlias = LogisticRegression | DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier | Pipeline

# The topical_inclusion suggestion model comes in eight flavors, crossing
# feature representation (tfidf: the original TF-IDF-over-title+abstract
# features; embedding: SPECTER embeddings, see src/validation/embeddings.py)
# with classifier family, all trained side by side each retrain so their
# metrics are directly comparable (see src/validation/ml_model_metrics.py).
# Only one actually drives suggestions at a time (DEFAULT_ACTIVE_MODEL to
# start, then whichever the reviewer picks at each retrain in `classify
# assisted`) -- the rest are trained and scored purely so there's something to
# compare the active one against.
MODEL_SPECS: dict[str, tuple[str, str]] = {
    "tfidf_logistic_regression": ("tfidf", "logistic_regression"),
    "tfidf_decision_tree": ("tfidf", "decision_tree"),
    "tfidf_random_forest": ("tfidf", "random_forest"),
    "tfidf_gradient_boosting": ("tfidf", "gradient_boosting"),
    "embedding_logistic_regression": ("embedding", "logistic_regression"),
    "embedding_decision_tree": ("embedding", "decision_tree"),
    "embedding_random_forest": ("embedding", "random_forest"),
    "embedding_gradient_boosting": ("embedding", "gradient_boosting"),
}
MODEL_TYPES = tuple(MODEL_SPECS)
CLASSIFIER_KINDS = ("logistic_regression", "decision_tree", "random_forest", "gradient_boosting")
# tfidf_* only, so this always trains and never depends on the (heavier,
# network-dependent) embedding backend being available.
DEFAULT_ACTIVE_MODEL = "tfidf_logistic_regression"

# Each classifier's candidate space, genuinely larger than RANDOM_SEARCH_ITER
# (_fit_with_cv uses RandomizedSearchCV, not GridSearchCV, so cost stays
# bounded at RANDOM_SEARCH_ITER samples regardless of how rich a grid is --
# widening a grid explores more of the space without growing per-fit cost).
_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logistic_regression": {"C": [0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]},
    "decision_tree": {"max_depth": [None, 5, 10, 15, 20, 30], "min_samples_split": [2, 5]},
    "random_forest": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20, 30]},
    "gradient_boosting": {"n_estimators": [50, 100, 150, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2]},
}

# Number of hyperparameter settings RandomizedSearchCV samples per fit,
# regardless of how many combinations _PARAM_GRIDS actually has -- this is
# what keeps the dimension side's per-category x per-model-type x per-batch
# fit count bounded even as the pipeline scales to hundreds of records and
# dozens of categories (see _fit_with_cv).
RANDOM_SEARCH_ITER = 5


def _build_classifier(classifier_kind: str, feature_kind: str = "tfidf") -> Classifier:
    if classifier_kind == "logistic_regression":
        # Unlike the tree-based classifiers below, logistic regression's L2
        # penalty and gradient-based solver both treat every feature's raw
        # scale as meaningful -- a feature with a naturally larger numeric
        # range gets regularized less and dominates the gradient, regardless
        # of how informative it actually is. TF-IDF is sparse, so centering
        # it (subtracting the mean) would both densify the matrix and be
        # statistically wrong for a bag-of-words representation -- scale by
        # variance only (with_mean=False). SPECTER embeddings are dense and
        # already L2-normalized per row (embeddings.embed_texts), but that
        # normalizes each example's vector length, not each dimension's
        # scale across examples -- StandardScaler still standardizes each of
        # the 768 dimensions to comparable variance, which is what logistic
        # regression's regularization actually assumes.
        scaler = StandardScaler(with_mean=(feature_kind == "embedding"))
        return Pipeline([
            ("scale", scaler),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
    if classifier_kind == "decision_tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=42)
    if classifier_kind == "random_forest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    if classifier_kind == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    raise ValueError(f"Unknown classifier_kind: {classifier_kind!r} -- must be one of {CLASSIFIER_KINDS}")


def _fit_with_cv(
    classifier_kind: str, feature_kind: str, x, labels: list[str] | list[int] | tuple[str, ...]
) -> tuple[Classifier, list]:
    """Basic CV + hyperparameter-selection pipeline shared by every model,
    old and new: a RandomizedSearchCV sampling RANDOM_SEARCH_ITER settings
    from _PARAM_GRIDS[classifier_kind] (a genuinely larger space than
    RANDOM_SEARCH_ITER -- see its definition), scored by macro F1, refit on
    the FULL `x`/`labels` at the end (sklearn's default
    RandomizedSearchCV(refit=True)) -- this refit model is what's returned
    for making real suggestions, and it has seen every example.
    RandomizedSearchCV rather than an exhaustive GridSearchCV is what keeps
    this bounded-cost: the 6-classification-dimension side calls this once
    per category per model type per retrain (potentially dozens of
    categories x 8 model types), where an exhaustive search over a rich grid
    would multiply out to an impractical fit count -- capping at
    RANDOM_SEARCH_ITER settings keeps that product bounded regardless of how
    rich _PARAM_GRIDS gets. Folds are capped at the smallest class's example
    count (never more folds than a class has examples) and explicitly
    stratified -- human decisions skew heavily toward one label (e.g. far
    more "exclude" than "include"), so an unstratified split could hand a
    fold zero examples of the minority class and blow up scoring.

    Also returns out-of-fold predictions for every example in `x`/`labels`,
    from the SAME StratifiedKFold used above, refit with the winning
    hyperparameters (sklearn.model_selection.cross_val_predict) -- this is
    what callers report as the model's quality (see
    ml_model_metrics.cumulative_metrics), instead of scoring against a
    single random holdout split. A single 80/20 split is a fresh, unrelated
    random resample every time the cumulative pool grows by one batch, so
    even as the pool gets large the metric stays noisy batch-to-batch (it's
    only ever asking "how did this one particular unlucky-or-lucky 20%
    slice go"); out-of-fold CV predictions use every example exactly once,
    each scored by a model that never trained on it, which is both a
    tighter estimate (more data contributes) and stable release-over-release
    (the same fold assignment, not a fresh coin flip every batch).

    logistic_regression is wrapped in a scaling Pipeline (see
    _build_classifier) -- the param grid keys are prefixed "clf__" in that
    case so they address the classifier step, not the scaler.

    Used for topical_inclusion (_fit_all_models) only -- one global model
    per model type, so the search cost is paid once, not multiplied. The 6
    classification dimensions' per-category classifiers use
    _fit_without_search instead (see there for why)."""
    folds = _cv_folds(labels)
    estimator = _build_classifier(classifier_kind, feature_kind)
    param_grid = _PARAM_GRIDS[classifier_kind]
    if isinstance(estimator, Pipeline):
        param_grid = {f"clf__{k}": v for k, v in param_grid.items()}
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator, param_grid, n_iter=RANDOM_SEARCH_ITER, cv=skf, scoring="f1_macro", n_jobs=-1, random_state=42
    )
    search.fit(x, labels)
    oof_predictions = list(cross_val_predict(clone(search.best_estimator_), x, labels, cv=skf, n_jobs=-1))
    return search.best_estimator_, oof_predictions


def _cv_folds(labels: list[str] | list[int] | tuple[str, ...]) -> int:
    """Number of StratifiedKFold splits for `labels` -- capped at 3, and
    never more than the smallest class's example count (StratifiedKFold
    raises if asked for more splits than a class has members). Shared by
    _fit_with_cv and _fit_without_search."""
    class_counts: dict[str | int, int] = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    return min(3, min(class_counts.values()))


def _fit_without_search(
    classifier_kind: str, feature_kind: str, x, labels: list[str] | list[int] | tuple[str, ...]
) -> tuple[Classifier, list]:
    """Same shape as _fit_with_cv (fitted estimator + out-of-fold CV
    predictions from the same StratifiedKFold, see there for why
    out-of-fold beats a holdout split) but skips hyperparameter search
    entirely -- fits classifier_kind's fixed default hyperparameters
    (_build_classifier) directly. Used for the 6 classification dimensions'
    per-category classifiers: these exist to give the reviewer a fast,
    good-enough second opinion to accept or override, not a tuned model,
    and this is called once per category per model type per retrain --
    potentially dozens of categories x 8 model types -- so a hyperparameter
    search multiplied across all of them is cost this pipeline doesn't need
    to pay, unlike topical_inclusion's single global model (_fit_with_cv),
    where the search happens exactly once per retrain regardless of corpus
    size."""
    folds = _cv_folds(labels)
    estimator = _build_classifier(classifier_kind, feature_kind)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof_predictions = list(cross_val_predict(clone(estimator), x, labels, cv=skf, n_jobs=-1))
    estimator.fit(x, labels)
    return estimator, oof_predictions


# Probabilities are clipped this far from 0 and 1 before the logit transform
# in _logit() -- tree ensembles routinely emit an exact 0.0 or 1.0 when every
# leaf agrees, and log(p/(1-p)) is infinite there.
PROBABILITY_CLIP = 1e-6


def _logit(probabilities) -> np.ndarray:
    """Positive-class probabilities as an (n, 1) column of log-odds -- the
    scale Platt scaling actually operates on. Fitting the calibrator on raw
    probabilities instead would ask a sigmoid to model a variable already
    squashed into [0, 1], which is why the textbook formulation calibrates
    the score/log-odds scale."""
    p = np.clip(np.asarray(probabilities, dtype=float), PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


@dataclass
class _PlattCalibrated:
    """A binary presence/absence classifier whose positive-class probability
    has been Platt-scaled, exposing the same `classes_` / `predict_proba`
    surface as the bare sklearn estimator it wraps -- DimensionModel.predict
    reads it without knowing the difference.

    Why this exists: the per-category classifiers of a multi-value dimension
    (_fit_dimension_category_classifiers) are independent one-vs-all fits,
    each trained on its own category's presence/absence balance. Their raw
    predict_proba outputs are NOT comparable across categories -- a category
    with 9 positives and one with 120 produce scores on different effective
    scales, and `class_weight="balanced"` (see _build_classifier)
    deliberately distorts both away from the true base rate to stop the
    majority class dominating the fit. bucket_suggestion(), meanwhile,
    applies ABSOLUTE cutoffs (HIGH_THRESHOLD/MEDIUM_THRESHOLD/
    LOW_THRESHOLD), so "0.70" has to mean the same thing in every category or
    the buckets are meaningless. Calibration is what makes that true: after
    it, 0.70 means "roughly 70% of papers scoring here really are in this
    category", per category, independently.

    Note what this is NOT: a softmax or any other cross-category
    normalisation. These 5 dimensions are multi-label -- a paper can be in
    two domains at once -- so forcing the per-category probabilities to sum
    to 1 would turn co-occurring categories into competing ones (two genuine
    0.9s become ~0.5 each and both fall out of the "high" bucket), and would
    make each probability depend on how many categories the dimension
    happens to define. Calibration fixes cross-category comparability
    without coupling the categories to each other."""

    base: Classifier
    calibrator: LogisticRegression
    base_positive_index: int
    classes_: list[int]

    def predict_proba(self, x) -> np.ndarray:
        base_positive = [row[self.base_positive_index] for row in self.base.predict_proba(x)]
        positive = self.calibrator.predict_proba(_logit(base_positive))[
            :, list(self.calibrator.classes_).index(1)
        ]
        return np.column_stack([1.0 - positive, positive])


def _fit_calibrated_binary(
    classifier_kind: str, feature_kind: str, x, presence: list[int]
) -> tuple[Classifier | _PlattCalibrated, list]:
    """Same contract as _fit_without_search (fitted classifier + out-of-fold
    predictions) for a single binary presence/absence category, with Platt
    calibration layered on at essentially zero extra cost.

    The trick is that the out-of-fold pass this function has to run anyway
    already yields exactly the held-out scores a Platt calibrator needs to be
    fit on: one `cross_val_predict` with method="predict_proba" instead of
    the default method="predict", then a two-parameter logistic regression on
    that single column. No nested cross-validation, no separate calibration
    holdout -- which matters, because this runs once per category per model
    type per retrain (dozens of categories x 8 model types), where sklearn's
    CalibratedClassifierCV(cv=k) would multiply the fit count by k and turn a
    per-batch retrain into minutes of waiting.

    Two honest caveats, both standard for this construction (they are the
    same approximations sklearn's own CalibratedClassifierCV(ensemble=False)
    makes):

    - The base model returned for real suggestions is refit on ALL the data,
      while the calibrator was fitted against fold models that each saw only
      (k-1)/k of it. The full-data model is marginally sharper than the fold
      models the calibration curve describes, so calibration is very slightly
      conservative.
    - The returned out-of-fold predictions are thresholded from CALIBRATED
      probabilities, and the calibrator saw those same out-of-fold scores
      when it was fitted -- so unlike the base model's predictions they are
      not strictly out-of-sample. The leak is through a 2-parameter monotone
      sigmoid on one feature, so the optimism is negligible next to the base
      model's own; it is not zero, and that is why it is written down here
      rather than glossed.

    Expect the metrics in ml_model_metrics to MOVE when this lands, for a
    real reason: the old predictions came from `estimator.predict()`, whose
    argmax sits on top of `class_weight="balanced"` and therefore over-calls
    positives for rare categories. Thresholding calibrated probabilities at
    0.5 restores the true base rate, so rare categories will show fewer
    predicted positives -- lower recall, higher precision. That is the
    correction working, not a regression."""
    folds = _cv_folds(presence)
    estimator = _build_classifier(classifier_kind, feature_kind)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    # Column order from cross_val_predict(method="predict_proba") follows
    # sorted(set(presence)); both classes are guaranteed present by the
    # caller's MIN_POSITIVES_PER_CATEGORY / MIN_EXAMPLES_PER_CLASS guards.
    classes = sorted(set(presence))
    positive_index = classes.index(1)
    oof_proba = cross_val_predict(clone(estimator), x, presence, cv=skf, method="predict_proba", n_jobs=-1)
    oof_positive = [row[positive_index] for row in oof_proba]

    estimator.fit(x, presence)

    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(_logit(oof_positive), presence)
    calibrated = _PlattCalibrated(
        base=estimator, calibrator=calibrator, base_positive_index=positive_index, classes_=[0, 1]
    )

    calibrated_oof = calibrator.predict_proba(_logit(oof_positive))[:, list(calibrator.classes_).index(1)]
    oof_predictions = [1 if p >= PRESENCE_THRESHOLD else 0 for p in calibrated_oof]
    return calibrated, oof_predictions

# Used only to derive an override/agreement signal for `classify ml-stats` on
# multi-value dimensions (see the CLI's _classify_dimensions): whether the
# categories a category-classifier predicts present (probability >= this)
# match the categories the reviewer actually picked, at any level. Never used
# to filter what's shown or logged -- every trained category's raw
# probability is always both.
PRESENCE_THRESHOLD = 0.5

# Bucket boundaries for bucket_suggestion(): turns a multi-value dimension's
# raw per-category probabilities into one proposed high/medium/low
# classification the reviewer can accept as a whole. Deliberately separate
# from PRESENCE_THRESHOLD above (that one only measures override agreement,
# never bucket membership). A category below LOW_THRESHOLD is left out of the
# suggestion entirely rather than forced into "low".
HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.40
LOW_THRESHOLD = 0.15


def _record_text(record: dict) -> str:
    return f"{record.get('title') or ''} {record.get('abstract') or ''}".strip()


def record_text(record: dict) -> str:
    """Public alias of _record_text -- used by callers outside this module
    (the CLI's batch-metrics bookkeeping) that need the exact same text a
    suggestion was scored against, without duplicating the title+abstract
    join logic."""
    return _record_text(record)


@dataclass
class Suggestion:
    label: str
    probability: float
    training_size: int


@dataclass
class TrainedModel:
    vectorizer: TfidfVectorizer | embeddings.EmbeddingExtractor
    classifier: Classifier
    training_size: int
    class_counts: dict[str, int]

    def predict(self, text: str) -> Suggestion:
        x = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(x)[0]
        classes = self.classifier.classes_
        best_idx = int(proba.argmax())
        return Suggestion(
            label=str(classes[best_idx]), probability=float(proba[best_idx]), training_size=self.training_size
        )


def _labeled_examples_full(engine: Engine) -> list[tuple[str, str, str]]:
    """(text, label, timestamp) for every human topical_inclusion decision
    made so far, across all reviewers and sample scopes -- more training data
    only helps. A record decided more than once (e.g. re-reviewed)
    contributes only its most recent label. Ordered by timestamp ascending --
    the order decisions were actually made in, which src/validation/
    ml_model_metrics.py::backfill_batches relies on to reconstruct historical
    retrain batches."""
    classifications_human = reflect_table(engine, "classifications_human")
    records_table = reflect_table(engine, "records")

    with engine.connect() as conn:
        rows = conn.execute(
            select(
                classifications_human.c.record_id,
                classifications_human.c.label,
                classifications_human.c.timestamp,
            ).where(classifications_human.c.dimension == "topical_inclusion")
        ).all()
        record_ids = {r[0] for r in rows}
        if not record_ids:
            return []
        text_by_id = {
            r["record_id"]: _record_text(dict(r))
            for r in conn.execute(records_table.select().where(records_table.c.record_id.in_(record_ids))).mappings()
        }

    latest: dict[str, tuple[str, str]] = {}  # record_id -> (timestamp, label)
    for record_id, label, timestamp in rows:
        if record_id not in latest or timestamp > latest[record_id][0]:
            latest[record_id] = (timestamp, label)

    examples = [
        (text_by_id[rid], label, ts) for rid, (ts, label) in latest.items() if text_by_id.get(rid)
    ]
    examples.sort(key=lambda e: e[2])
    return examples


def _labeled_examples(engine: Engine) -> list[tuple[str, str]]:
    """(text, label) pairs -- see _labeled_examples_full for the full
    (text, label, timestamp) form this is derived from."""
    return [(text, label) for text, label, _ts in _labeled_examples_full(engine)]


def _fit_all_models(texts: list[str], labels: list[str]) -> tuple[dict[str, TrainedModel], dict[str, list]]:
    """Fits every entry in MODEL_SPECS on the same (texts, labels) set, each
    through _fit_with_cv's CV + hyperparameter-selection pipeline -- and
    returns its out-of-fold predictions alongside it (see _fit_with_cv),
    keyed the same way. This is the shared core of train_all() (live retrain
    in `classify assisted`, which only needs the models) and
    ml_model_metrics.cumulative_metrics()/backfill_batches() (which need the
    out-of-fold predictions to report quality), so a live retrain's
    production models and its reported metrics always come from the exact
    same single fit -- not two separate fits scoring slightly different
    models. Callers are responsible for the MIN_TOTAL_EXAMPLES /
    MIN_EXAMPLES_PER_CLASS guards before calling this -- it assumes the
    training set is already big enough and class-balanced enough to fit
    something meaningful.

    embedding_* model types are silently omitted (not raised) if the
    embedding backend fails -- e.g. no network to download SPECTER the first
    time it's needed -- so a reviewer without connectivity still gets the
    tfidf_* suggestion models rather than losing assistance entirely; the CLI
    surfaces that omission (src/cli/human_review_commands.py)."""
    class_counts: dict[str, int] = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, stop_words="english")
    tfidf_x = tfidf_vectorizer.fit_transform(texts)

    embedding_extractor: embeddings.EmbeddingExtractor | None = None
    embedding_x = None
    try:
        embedding_extractor = embeddings.EmbeddingExtractor()
        embedding_x = embedding_extractor.transform(texts)
    except Exception:  # noqa: BLE001 -- any failure (no network, OOM, corrupt cache, ...) degrades to tfidf-only
        embedding_extractor = None

    models: dict[str, TrainedModel] = {}
    oof_predictions_by_type: dict[str, list] = {}
    for model_type, (feature_kind, classifier_kind) in MODEL_SPECS.items():
        if feature_kind == "tfidf":
            vectorizer, x = tfidf_vectorizer, tfidf_x
        else:
            if embedding_extractor is None:
                continue
            vectorizer, x = embedding_extractor, embedding_x
        classifier, oof_predictions = _fit_with_cv(classifier_kind, feature_kind, x, labels)
        models[model_type] = TrainedModel(
            vectorizer=vectorizer, classifier=classifier, training_size=len(labels), class_counts=class_counts
        )
        oof_predictions_by_type[model_type] = oof_predictions
    return models, oof_predictions_by_type


def train_all(engine: Engine, *, max_examples: int | None = None) -> dict[str, TrainedModel] | None:
    """One TrainedModel per entry in MODEL_SPECS (see _fit_all_models),
    trained on the same labeled example set so they're all scored on
    directly comparable data -- only their features (tfidf/embedding) and
    classifier differ. Returns None under the same not-enough-data
    conditions as train().

    max_examples, if given, trains only on the first `max_examples`
    chronological examples instead of everything currently in the DB -- used
    by `classify assisted` (src/cli/human_review_commands.py) when resuming a
    batch that's already partway done (e.g. after an earlier session was
    aborted before that batch's retrain fired), so the decisions still
    pending toward that batch stay held out of training instead of leaking
    into the model that's about to be evaluated against them once the batch
    completes."""
    examples = _labeled_examples(engine)
    if max_examples is not None:
        examples = examples[:max_examples]
    if len(examples) < MIN_TOTAL_EXAMPLES:
        return None

    texts, labels = zip(*examples, strict=True)
    class_counts: dict[str, int] = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    if len(class_counts) < 2 or min(class_counts.values()) < MIN_EXAMPLES_PER_CLASS:
        return None

    models, _oof_predictions_by_type = _fit_all_models(list(texts), list(labels))
    return models


def train(engine: Engine) -> TrainedModel | None:
    """Returns None if there isn't yet enough labeled data to fit a
    meaningful model -- callers fall back to the unassisted review loop in
    that case (see MIN_TOTAL_EXAMPLES / MIN_EXAMPLES_PER_CLASS). Kept as the
    single-model entry point (DEFAULT_ACTIVE_MODEL, tfidf_logistic_regression)
    for callers that don't need the other model types -- see train_all() for
    all of MODEL_SPECS trained together."""
    models = train_all(engine)
    return models[DEFAULT_ACTIVE_MODEL] if models else None


def suggest(model: TrainedModel | None, record: dict) -> Suggestion | None:
    """None if there's no usable model yet, or the record has no text to
    score at all."""
    if model is None:
        return None
    text = _record_text(record)
    if not text:
        return None
    return model.predict(text)


def log_suggestion(
    engine: Engine,
    record_id: str,
    reviewer_id: str,
    suggested_label: str,
    suggested_probability: float,
    training_size: int,
    final_label: str,
    dimension: str = "topical_inclusion",
    was_override: bool | None = None,
) -> None:
    """Call once a decision has actually been saved for the record/dimension
    the suggestion was shown for -- never for a skipped/quit-without-deciding
    record, since there's no final_label to compare against yet.
    suggested_label/final_label are plain strings for topical_inclusion and
    single-value dimensions; callers serialize a multi-value dimension's raw
    category_probabilities dict to JSON before logging (see
    src/cli/human_review_commands.py). was_override defaults to a plain
    string comparison of suggested_label/final_label, which only makes sense
    when both represent the same kind of value (a single label) -- a
    multi-value dimension's suggested_label is a probability dict, not a
    label, so its caller must pass was_override explicitly."""
    ml_assist_suggestions = reflect_table(engine, "ml_assist_suggestions")
    now = datetime.now(UTC).isoformat()
    computed_override = (final_label != suggested_label) if was_override is None else was_override
    with engine.begin() as conn:
        conn.execute(
            ml_assist_suggestions.insert().values(
                suggestion_id=new_id(),
                record_id=record_id,
                reviewer_id=reviewer_id,
                dimension=dimension,
                suggested_label=suggested_label,
                suggested_probability=suggested_probability,
                training_set_size=training_size,
                final_label=final_label,
                was_override=1 if computed_override else 0,
                timestamp=now,
            )
        )


@dataclass
class AssistStats:
    total: int
    overrides: int
    recent_total: int
    recent_overrides: int


def assist_stats(engine: Engine, recent_window: int = 20, dimension: str | None = None) -> AssistStats:
    """Agreement/override counts over every logged suggestion, plus the same
    over just the most recent `recent_window` -- the trend line is what
    actually answers "is this working," not a single cumulative rate.
    dimension=None (default) reports across topical_inclusion and every
    classification dimension combined; pass a specific dimension name to
    scope to just that one (see logged_dimensions())."""
    ml_assist_suggestions = reflect_table(engine, "ml_assist_suggestions")
    stmt = select(ml_assist_suggestions.c.was_override, ml_assist_suggestions.c.timestamp).order_by(
        ml_assist_suggestions.c.timestamp
    )
    if dimension is not None:
        stmt = stmt.where(ml_assist_suggestions.c.dimension == dimension)
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()

    total = len(rows)
    overrides = sum(1 for r in rows if r[0])
    recent = rows[-recent_window:]
    return AssistStats(
        total=total, overrides=overrides, recent_total=len(recent), recent_overrides=sum(1 for r in recent if r[0])
    )


def logged_dimensions(engine: Engine) -> list[str]:
    """Distinct dimension values with at least one logged suggestion, in
    first-seen order -- lets `classify ml-stats` report a per-dimension
    breakdown without hardcoding dimension names."""
    ml_assist_suggestions = reflect_table(engine, "ml_assist_suggestions")
    with engine.connect() as conn:
        rows = conn.execute(
            select(ml_assist_suggestions.c.dimension, ml_assist_suggestions.c.timestamp).order_by(
                ml_assist_suggestions.c.timestamp
            )
        ).all()
    seen: list[str] = []
    for dimension, _timestamp in rows:
        if dimension not in seen:
            seen.append(dimension)
    return seen


@dataclass
class MultiLabelSuggestion:
    category_probabilities: dict[str, float]  # every trained category -> predicted-positive probability
    training_size: int


@dataclass
class SingleLabelSuggestion:
    label: str  # "" means "no category"
    probability: float
    training_size: int


def bucket_suggestion(suggestion: MultiLabelSuggestion) -> dict[str, list[str]]:
    """Turns a multi-value dimension's calibrated per-category probabilities
    (_fit_calibrated_binary) into a single proposed {"high": [...],
    "medium": [...], "low": [...]} classification.

    Each probability is P(the reviewer rates this category 'high'), so
    HIGH_THRESHOLD is a cutoff on the very quantity it names, and the
    medium/low bands read as decreasing confidence in that same judgement.
    (This only became true once the classifiers were retargeted off category
    presence, which the grouped H/M/L entry design makes constant -- see
    _parse_high_categories.)

    This is the shape `classify assisted` shows the reviewer as one
    suggestion to accept whole (Enter) or reject in favor of the manual
    prompt (c), same as every other item. Categories within a bucket are
    ordered by probability, highest first; a category below LOW_THRESHOLD is
    omitted entirely rather than forced into "low"."""
    ranked = sorted(suggestion.category_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    buckets: dict[str, list[str]] = {"high": [], "medium": [], "low": []}
    for category, probability in ranked:
        if probability >= HIGH_THRESHOLD:
            buckets["high"].append(category)
        elif probability >= MEDIUM_THRESHOLD:
            buckets["medium"].append(category)
        elif probability >= LOW_THRESHOLD:
            buckets["low"].append(category)
    return buckets


def _parse_categories(label_json: str) -> list[str]:
    """The list of categories selected for a multi-value dimension row --
    classifications_human.label is a JSON array (dimension_classification.
    save_dimension_classification). '' or malformed JSON both mean "none
    selected" rather than raising, since an explicit empty selection is a
    valid answer (see dimension_classification.py's module docstring).

    NOT the training target for multi-value dimensions -- see
    _parse_high_categories for why presence is unusable there. This stays
    for callers that genuinely want "which categories were selected at all"
    (agreement/adjudication read the same JSON-array shape)."""
    try:
        return json.loads(label_json) if label_json else []
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_high_categories(label_levels_json: str) -> list[str]:
    """The categories the reviewer put in the 'high' bucket, from
    classifications_human.label_levels_json ({category: level}, written by
    dimension_classification.save_dimension_classification).

    This is the multi-value training target, replacing category presence.
    Under the grouped H/M/L entry design every category of a dimension is
    assigned some level, so classifications_human.label contains all of them
    on every record: presence is constant-1, every category has zero
    negatives, and no classifier can be fit. The discriminating information
    the reviewer actually supplies is the LEVEL, and P(high) is the quantity
    bucket_suggestion's HIGH_THRESHOLD already tests against, so targeting it
    is what makes that threshold mean what it claims to.

    '' / None / malformed JSON all yield [] -- an explicit empty selection
    stores no levels map and legitimately means "no category is high".
    Callers must not confuse that with a pre-migration-005 row that lost its
    levels; _dimension_examples_full drops those instead of passing them
    through as all-zeros."""
    try:
        levels = json.loads(label_levels_json) if label_levels_json else {}
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(levels, dict):
        return []
    return [category for category, level in levels.items() if level == "high"]


@dataclass
class _CategoryClassifier:
    category: str
    classifier: Classifier


@dataclass
class DimensionModel:
    dimension: str
    multi_value: bool
    vectorizer: TfidfVectorizer | embeddings.EmbeddingExtractor
    training_size: int
    category_classifiers: list[_CategoryClassifier] | None = None  # multi-value dimensions
    classifier: Classifier | None = None  # single-value dimensions

    def predict(self, text: str) -> MultiLabelSuggestion | SingleLabelSuggestion:
        x = self.vectorizer.transform([text])
        if self.multi_value:
            probabilities: dict[str, float] = {}
            for cc in self.category_classifiers or []:
                proba = cc.classifier.predict_proba(x)[0]
                classes = list(cc.classifier.classes_)
                probabilities[cc.category] = float(proba[classes.index(1)])
            return MultiLabelSuggestion(category_probabilities=probabilities, training_size=self.training_size)
        assert self.classifier is not None  # guaranteed by _fit_dimension_single_value_classifier whenever multi_value=False
        proba = self.classifier.predict_proba(x)[0]
        classes = self.classifier.classes_
        best_idx = int(proba.argmax())
        return SingleLabelSuggestion(
            label=str(classes[best_idx]), probability=float(proba[best_idx]), training_size=self.training_size
        )


def _dimension_examples_full(engine: Engine, dimension: str, levels: bool = False) -> list[tuple[str, str, str]]:
    """(text, label, timestamp) for every human classification of this
    dimension so far, one row per record (latest wins on re-review), ordered
    by timestamp ascending -- mirrors _labeled_examples_full for
    topical_inclusion. ml_model_metrics.py's cumulative_dimension_metrics/
    backfill_dimension_batches rely on this chronological ordering.

    levels=True (what every multi-value caller passes) returns
    classifications_human.label_levels_json in the label slot instead of
    .label, so downstream code sees the {category: level} map that
    _parse_high_categories needs rather than the constant-1 presence array.
    The tuple shape is deliberately unchanged: single- and multi-value
    dimensions still flow through the same pipeline, they just disagree on
    what "the label" is, exactly as they already did (a bare category string
    for evidence_type, JSON for the rest).

    A row that selected categories but carries no levels map predates
    migration 005 and has no recoverable level information; it is DROPPED
    rather than passed through, because _parse_high_categories would
    otherwise read it as "no category is high" and train on a fabricated
    negative. A row with no categories at all keeps its empty map -- that is
    a real "nothing is high"."""
    classifications_human = reflect_table(engine, "classifications_human")
    records_table = reflect_table(engine, "records")

    label_column = classifications_human.c.label_levels_json if levels else classifications_human.c.label

    with engine.connect() as conn:
        rows = conn.execute(
            select(
                classifications_human.c.record_id,
                label_column,
                classifications_human.c.timestamp,
                classifications_human.c.label,
            ).where(classifications_human.c.dimension == dimension)
        ).all()
        if levels:
            rows = [
                (record_id, levels_json or "{}", timestamp)
                for record_id, levels_json, timestamp, raw_label in rows
                if levels_json or not _parse_categories(raw_label)
            ]
        else:
            rows = [(record_id, label, timestamp) for record_id, label, timestamp, _raw in rows]
        record_ids = {r[0] for r in rows}
        if not record_ids:
            return []
        text_by_id = {
            r["record_id"]: _record_text(dict(r))
            for r in conn.execute(records_table.select().where(records_table.c.record_id.in_(record_ids))).mappings()
        }

    latest: dict[str, tuple[str, str, str]] = {}  # record_id -> (timestamp, text, label)
    for record_id, label, timestamp in rows:
        if record_id not in latest or timestamp > latest[record_id][0]:
            latest[record_id] = (timestamp, text_by_id.get(record_id, ""), label)

    examples = [(text, label, ts) for ts, text, label in latest.values() if text]
    examples.sort(key=lambda e: e[2])
    return examples


def _dimension_examples(engine: Engine, dimension: str, levels: bool = False) -> list[tuple[str, str]]:
    """(text, label) pairs -- see _dimension_examples_full for the full
    (text, label, timestamp) form this is derived from, and for what
    levels=True puts in the label slot."""
    return [(text, label) for text, label, _ts in _dimension_examples_full(engine, dimension, levels=levels)]


def _dimension_feature_matrix(texts: list[str], feature_kind: str) -> tuple[TfidfVectorizer | embeddings.EmbeddingExtractor, Any]:
    """Shared by both dimension-fitting functions below: one feature matrix
    per (dimension, model_type) fit, reused across every category a
    multi-value dimension has. May raise if feature_kind == "embedding" and
    the embedding backend is unavailable -- callers catch this the same way
    _fit_all_models does."""
    if feature_kind == "tfidf":
        vectorizer: Any = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, stop_words="english")
        x = vectorizer.fit_transform(texts)
    else:
        vectorizer = embeddings.EmbeddingExtractor()
        x = vectorizer.transform(texts)
    return vectorizer, x


def _fit_dimension_category_classifiers(
    dim: DimensionSchema, texts: list[str], labels: list[str], model_type: str
) -> tuple[DimensionModel, dict[str, list]] | None:
    """Multi-value dimension: one binary high/not-high classifier per
    category, all using model_type's (feature_kind, classifier_kind) --
    mirrors _fit_all_models's per-model-type training, just for a
    dimension's categories instead of topical_inclusion's single label.

    `labels` must be the {category: level} maps, i.e. from
    _dimension_examples(_full)(..., levels=True) -- the target is "did the
    reviewer rate this category 'high'", not "was it selected". Category
    presence is constant-1 under grouped H/M/L entry and cannot be trained
    on; see _parse_high_categories.

    Caller is responsible for MIN_TOTAL_EXAMPLES on the overall example
    count; this guards each category's own MIN_POSITIVES_PER_CATEGORY
    'high' examples and MIN_EXAMPLES_PER_CLASS not-high ones. A category
    that is unanimously high (or never high) across the corpus so far is
    skipped for want of contrast, not because it is unimportant. Raises if
    feature_kind == "embedding" and the embedding backend is unavailable --
    caller catches, same pattern as _fit_all_models.

    Each category's classifier is Platt-calibrated (_fit_calibrated_binary),
    so its probability is comparable to every other category's and to
    bucket_suggestion's absolute HIGH/MEDIUM/LOW cutoffs -- see
    _PlattCalibrated for why one-vs-all probabilities are not comparable
    out of the box, and why calibration rather than a cross-category softmax
    is the right fix for a multi-label dimension.

    Uses _fit_calibrated_binary, which shares _fit_without_search's
    no-hyperparameter-search stance rather than _fit_with_cv's -- these are a
    fast, good-enough second opinion for the reviewer, not a tuned model, and
    this runs once per category per model type per retrain (potentially
    dozens of categories x 8 model types), so a hyperparameter search
    multiplied across all of them would be real, avoidable cost.

    Also returns each category's out-of-fold predictions, keyed by category
    -- what ml_model_metrics.evaluate_dimension_model_multi_value_from_oof
    scores against, instead of a separate holdout split. Every category gets
    its OWN StratifiedKFold (via _fit_without_search), stratified by that
    category's own high/not-high balance -- properly stratified per
    category, not one shared split for the whole dimension."""
    feature_kind, classifier_kind = MODEL_SPECS[model_type]
    vectorizer, x = _dimension_feature_matrix(texts, feature_kind)

    category_classifiers: list[_CategoryClassifier] = []
    oof_by_category: dict[str, list] = {}
    for category in dim.categories:
        is_high = [1 if category in _parse_high_categories(levels_json) else 0 for levels_json in labels]
        positives = sum(is_high)
        negatives = len(is_high) - positives
        if positives < MIN_POSITIVES_PER_CATEGORY or negatives < MIN_EXAMPLES_PER_CLASS:
            continue  # too few examples either way for a meaningful classifier -- no suggestion for this category

        classifier, oof_predictions = _fit_calibrated_binary(classifier_kind, feature_kind, x, is_high)
        category_classifiers.append(_CategoryClassifier(category=category, classifier=classifier))
        oof_by_category[category] = oof_predictions

    if not category_classifiers:
        return None
    model = DimensionModel(
        dimension=dim.name, multi_value=True, vectorizer=vectorizer, training_size=len(labels),
        category_classifiers=category_classifiers,
    )
    return model, oof_by_category


def _fit_dimension_single_value_classifier(
    dim: DimensionSchema, texts: list[str], labels: list[str], model_type: str
) -> tuple[DimensionModel, list, list] | None:
    """Single-value dimension (evidence_type): one multiclass classifier
    using model_type's (feature_kind, classifier_kind) -- same shape as
    topical_inclusion's own model. Rare categories are dropped from
    training (the reviewer can still pick them manually on override), same
    as before. Caller is responsible for MIN_TOTAL_EXAMPLES on the overall
    example count. Raises if feature_kind == "embedding" and the embedding
    backend is unavailable -- caller catches, same pattern as
    _fit_all_models.

    Uses _fit_without_search, not _fit_with_cv, same rationale as
    _fit_dimension_category_classifiers -- a fast second opinion across
    potentially dozens of (dimension, model_type) combinations, not a tuned
    model.

    Also returns (filtered_labels, out_of_fold_predictions) -- the true
    labels and out-of-fold predictions line up positionally, and
    filtered_labels can differ from the full `labels` (rare categories
    dropped), so callers need both, not just `labels`."""
    class_counts: dict[str, int] = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    eligible = {label for label, count in class_counts.items() if count >= MIN_EXAMPLES_PER_CLASS}
    filtered = [(text, label) for text, label in zip(texts, labels, strict=True) if label in eligible]
    if len(eligible) < 2 or len(filtered) < MIN_TOTAL_EXAMPLES:
        return None

    filtered_texts, filtered_labels = zip(*filtered, strict=True)
    feature_kind, classifier_kind = MODEL_SPECS[model_type]
    vectorizer, x = _dimension_feature_matrix(list(filtered_texts), feature_kind)
    classifier, oof_predictions = _fit_without_search(classifier_kind, feature_kind, x, list(filtered_labels))
    model = DimensionModel(
        dimension=dim.name, multi_value=False, vectorizer=vectorizer, training_size=len(filtered), classifier=classifier
    )
    return model, list(filtered_labels), oof_predictions


def train_dimensions_all(engine: Engine, schema: list[DimensionSchema]) -> dict[str, dict[str, DimensionModel]]:
    """dimension_name -> {model_type: DimensionModel}, one entry per
    MODEL_SPECS model type that has enough data to fit for that dimension
    (mirrors train_all's relationship to _fit_all_models, just per
    dimension). A dimension below MIN_TOTAL_EXAMPLES is simply absent.
    embedding_* entries are silently omitted, per dimension, if the
    embedding backend fails -- same graceful-degradation pattern as
    _fit_all_models, so a reviewer without network access still gets the
    tfidf_* dimension models."""
    models_by_dimension: dict[str, dict[str, DimensionModel]] = {}
    for dim in schema:
        examples = _dimension_examples(engine, dim.name, levels=dim.multi_value)
        if len(examples) < MIN_TOTAL_EXAMPLES:
            continue
        texts = [text for text, _label in examples]
        labels = [label for _text, label in examples]

        models: dict[str, DimensionModel] = {}
        for model_type in MODEL_TYPES:
            try:
                fitted = (
                    _fit_dimension_category_classifiers(dim, texts, labels, model_type)
                    if dim.multi_value
                    else _fit_dimension_single_value_classifier(dim, texts, labels, model_type)
                )
            except Exception:  # noqa: BLE001, S112 -- embedding backend unavailable, degrade to tfidf-only for this dimension
                continue
            if fitted is not None:
                models[model_type] = fitted[0]
        if models:
            models_by_dimension[dim.name] = models
    return models_by_dimension


def train_dimensions(engine: Engine, schema: list[DimensionSchema]) -> dict[str, DimensionModel]:
    """One model per dimension that currently has enough labeled data, using
    DEFAULT_ACTIVE_MODEL -- kept as the single-model entry point for callers
    that don't need the other model types (see train_dimensions_all for all
    of MODEL_SPECS trained together, and `classify assisted`'s
    active_dimension_model_type selection for how one is actually chosen to
    drive real suggestions across all 6 dimensions)."""
    return {
        name: models[DEFAULT_ACTIVE_MODEL]
        for name, models in train_dimensions_all(engine, schema).items()
        if DEFAULT_ACTIVE_MODEL in models
    }


def suggest_dimension(
    model: DimensionModel | None, record: dict
) -> MultiLabelSuggestion | SingleLabelSuggestion | None:
    """None if there's no usable model for this dimension yet, or the record
    has no text to score at all."""
    if model is None:
        return None
    text = _record_text(record)
    if not text:
        return None
    return model.predict(text)
