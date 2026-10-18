"""Per-batch, per-model-type suggestion-quality tracking for the
topical_inclusion suggestion model in `classify assisted`.

`classify assisted` retrains its suggestion models every --batch-size
decisions (src/cli/human_review_commands.py::assisted). Each such boundary is
one "batch" here. Right before retraining, every model type is fit ONCE
(cumulative_metrics -> ml_assist._fit_all_models) on the full cumulative set
of human topical_inclusion decisions made so far (everything up to and
including this batch), and scored by its own OUT-OF-FOLD cross-validation
predictions from that same fit (ml_assist._fit_with_cv: the same
StratifiedKFold already used for hyperparameter selection, refit with the
winning hyperparameters via sklearn.model_selection.cross_val_predict).
Every model in ml_assist.MODEL_TYPES is scored this way every batch, even
though only one of them (ml_assist.DEFAULT_ACTIVE_MODEL, or whichever the
reviewer picked at the previous retrain) actually drove the suggestions
shown that batch -- this is what lets the reviewer compare all of them (see
ml_assist.MODEL_SPECS: 4 classifier families x 2 feature representations,
tfidf and SPECTER embeddings) side by side and switch which one drives
suggestions going forward.

This out-of-fold approach replaced two earlier designs, in order:
1. Scoring the pre-batch model against just that batch's own ~--batch-size
   decisions -- a single batch is small, and human decisions skew heavily
   toward one label, so a ~20-record batch can easily be almost entirely one
   class, making the reported number noisy and potentially misleading.
2. A single stratified 80/20 train/holdout split of the full cumulative set
   (fit on 80%, score against the 20% holdout) -- larger and class-balanced
   by construction, but STILL noisy batch-to-batch even once the cumulative
   pool is large: a fresh random 20% is drawn every single retrain (not a
   nested/consistent sample), so successive batches' reported numbers are
   effectively independent random draws rather than a converging estimate,
   and a small absolute count of the minority class in any one draw can
   still swing macro F1 substantially.
Out-of-fold CV predictions fix both: every example in the cumulative set
contributes to the estimate (not just 20% of it), each scored by a model
that never trained on it within that fold, and the same fold assignment is
used consistently within a given retrain's fit -- so the metric reported
gets both larger-sample-size AND lower-variance as review continues, instead
of resetting to a fresh coin flip every batch.

Because the out-of-fold predictions come from the SAME fit used to build the
"production" models `classify assisted` actually suggests from (the
RandomizedSearchCV refit on the FULL cumulative set, sklearn's default
refit=True), there is only ONE fit per model type per retrain now -- no
separate "trained on a subset, just for scoring" pass. cumulative_metrics()
returns both the metrics and those same models so the CLI's retrain loop
doesn't need to fit everything a second time afterward.

backfill_batches() reconstructs the same per-batch evaluation retroactively
from every human topical_inclusion decision made so far, ordered by the
timestamp the decision was actually saved at (ml_assist.
_labeled_examples_full) -- useful the first time this module is introduced,
when past sessions never recorded per-batch metrics, and safe to re-run any
time (it always rebuilds the full history from scratch rather than
appending, so it can't double-count a batch also recorded live).

cumulative_dimension_metrics()/backfill_dimension_batches() are the
counterpart of the above for the 6 classification dimensions
(config/classification_schema.yaml), tracked in a SEPARATE file/plot
(DEFAULT_DIMENSION_METRICS_PATH/DEFAULT_DIMENSION_METRICS_PLOT_PATH). Unlike
topical_inclusion, where each model type gets one score, here each model
type gets one score PER DIMENSION, then those are averaged ACROSS the 6
dimensions into one aggregate score per model type -- because `classify
assisted` applies a single active model type uniformly to all 6 dimensions,
not one choice per dimension (ml_assist.py's module docstring). Each is
scored through the same fit that produces its production model:
evidence_type (single-value) via _cumulative_single_value_dimension_metrics
-> ml_assist._fit_dimension_single_value_classifier, and the 5 multi-value
dimensions via one out-of-fold-scored classifier per category (ml_assist.
_fit_dimension_category_classifiers), each with its OWN StratifiedKFold
stratified by that specific category's own high/not-high balance -- no
shared split needed, unlike the discarded holdout-split design, where a
multi-value dimension had no single scalar label to stratify a shared split
by in the first place.

Two corrections worth knowing about when reading historical batches in these
files, because both changed what the dimension numbers mean:
- The multi-value target was category PRESENCE until it was retargeted to
  P(category rated 'high'). Presence is constant-1 under the grouped H/M/L
  entry design, so every multi-value dimension silently stopped scoring once
  that design landed -- see ml_assist._parse_high_categories.
- evidence_type used cumulative_metrics(), which vetoes the whole dimension
  if any class has a single example, while the fit it mirrored dropped such
  classes and trained on the rest -- see
  _cumulative_single_value_dimension_metrics.
dimension_scoring_diagnostics() exists so a dimension that cannot be scored
says why, rather than vanishing from the report as both of those did.
Dimension-side batches are numbered independently of the topical file's
batch numbers (not forced to align 1:1) -- dimension data only starts
accumulating once records reach 'include' and then get classified, so it
inherently lags and thins out relative to topical_inclusion decisions; a
given topical retrain checkpoint may not have enough dimension data to
produce a scoreable batch yet even once topical_inclusion does. Both share
the same underlying cutoff timestamps (topical_inclusion's chronology)
though, so batches can still be compared by their `timestamp` field.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sqlalchemy import Engine

from review_pipeline.src.common.config import OUTPUTS_DIR
from review_pipeline.src.validation import ml_assist

DEFAULT_METRICS_PATH = OUTPUTS_DIR / "ml_model_metrics.json"
DEFAULT_METRICS_PLOT_PATH = OUTPUTS_DIR / "ml_model_metrics.png"
DEFAULT_DIMENSION_METRICS_PATH = OUTPUTS_DIR / "ml_dimension_model_metrics.json"
DEFAULT_DIMENSION_METRICS_PLOT_PATH = OUTPUTS_DIR / "ml_dimension_model_metrics.png"


@dataclass
class ModelMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    cohen_kappa: float  # chance-corrected agreement between model prediction and reviewer's actual decision
    n: int


def _metrics_from_predictions(true_labels: list, predicted_labels: list) -> ModelMetrics:
    """The actual metric computation shared by every evaluation path in this
    module -- given true labels and predictions already computed by
    whatever means (a model scored against raw text, or -- as of the
    out-of-fold-CV rework -- ml_assist._fit_with_cv's out-of-fold
    predictions), just crunches the numbers. cohen_kappa is stored as 0.0
    whenever sklearn considers it undefined -- true_labels and predicted
    only have one label in common (e.g. a class that happened to be
    entirely one label, or a model that predicted the same class
    throughout). 0.0 reads as "no measurable chance-corrected agreement
    yet" without putting a NaN into the metrics file, which json.dumps
    would otherwise emit as the non-standard literal `NaN`."""
    with warnings.catch_warnings():
        # Both warnings fire exactly on the single-shared-label case this
        # function already handles explicitly below (kappa -> 0.0) --
        # expected and handled, not something the reviewer needs to see.
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        precision, recall, f1, _support = precision_recall_fscore_support(
            true_labels, predicted_labels, average="macro", zero_division=0
        )
        kappa = cohen_kappa_score(true_labels, predicted_labels)
    return ModelMetrics(
        accuracy=float(accuracy_score(true_labels, predicted_labels)),
        precision_macro=float(precision),
        recall_macro=float(recall),
        f1_macro=float(f1),
        cohen_kappa=0.0 if math.isnan(kappa) else float(kappa),
        n=len(true_labels),
    )


def evaluate_model(model: ml_assist.TrainedModel, texts: list[str], true_labels: list[str]) -> ModelMetrics:
    """Scores one already-trained model against texts/labels it was NOT
    trained on -- the caller is responsible for that separation; this
    function just runs the model and hands the predictions to
    _metrics_from_predictions. Only used where an already-fitted model needs
    scoring against externally-supplied text (e.g. `classify ml-stats`-style
    ad hoc checks) -- the retrain-time reporting path
    (cumulative_metrics/cumulative_dimension_metrics) uses out-of-fold
    predictions from the fit itself instead, see module docstring."""
    predicted = [model.predict(text).label for text in texts]
    return _metrics_from_predictions(true_labels, predicted)


def evaluate_models(
    models: dict[str, ml_assist.TrainedModel], texts: list[str], true_labels: list[str]
) -> dict[str, ModelMetrics]:
    """evaluate_model() for every model in `models` (normally
    ml_assist.MODEL_TYPES), keyed the same way."""
    return {model_type: evaluate_model(model, texts, true_labels) for model_type, model in models.items()}


@dataclass
class CumulativeEvaluation:
    metrics: dict[str, ModelMetrics]
    models: dict[str, ml_assist.TrainedModel]  # the SAME fit used to compute `metrics` -- also the production models
    n: int  # cumulative example count fitted AND evaluated (every example both trains and, via CV rotation, tests)


def cumulative_metrics(texts: list[str], labels: list[str]) -> CumulativeEvaluation | None:
    """The metrics reported to the reviewer at every retrain and recorded in
    history (see module docstring): every entry in ml_assist.MODEL_SPECS is
    fit ONCE on the full cumulative (texts, labels) -- every human
    topical_inclusion decision made so far, not just the latest batch -- via
    ml_assist._fit_all_models, and scored by its own out-of-fold
    cross-validation predictions from that same fit (see ml_assist.
    _fit_with_cv). Also returns those same fitted `models` -- callers that
    need the day-to-day production models (e.g. `classify assisted`'s
    retrain) don't need a second, separate fit call; this single fit serves
    both. Returns None if the cumulative set is still too small/imbalanced
    for _fit_with_cv's CV to run at all (same MIN_TOTAL_EXAMPLES/
    MIN_EXAMPLES_PER_CLASS window every other guard in this pipeline uses)."""
    class_counts: dict[str, int] = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    if len(labels) < ml_assist.MIN_TOTAL_EXAMPLES or len(class_counts) < 2 or min(class_counts.values()) < ml_assist.MIN_EXAMPLES_PER_CLASS:
        return None

    models, oof_predictions_by_type = ml_assist._fit_all_models(texts, labels)
    metrics = {mt: _metrics_from_predictions(labels, oof) for mt, oof in oof_predictions_by_type.items()}
    return CumulativeEvaluation(metrics=metrics, models=models, n=len(texts))


def _average_model_metrics(metrics: list[ModelMetrics]) -> ModelMetrics:
    """Mean of every numeric field across `metrics` (n is SUMMED, not
    averaged, so it still reflects the total number of examples actually
    scored somewhere in the average) -- shared by
    evaluate_dimension_model_multi_value (averaging across one dimension's
    categories) and cumulative_dimension_metrics (averaging across the 6
    dimensions)."""
    count = len(metrics)
    return ModelMetrics(
        accuracy=sum(m.accuracy for m in metrics) / count,
        precision_macro=sum(m.precision_macro for m in metrics) / count,
        recall_macro=sum(m.recall_macro for m in metrics) / count,
        f1_macro=sum(m.f1_macro for m in metrics) / count,
        cohen_kappa=sum(m.cohen_kappa for m in metrics) / count,
        n=sum(m.n for m in metrics),
    )


def _dimension_examples_by_name(
    engine: Engine, schema: list[ml_assist.DimensionSchema]
) -> dict[str, list[tuple[str, str, str]]]:
    """Every dimension's (text, label, timestamp) examples, keyed by name --
    with levels=True for the multi-value dimensions, so their label slot
    holds the {category: level} map their high/not-high target is read from
    (ml_assist._parse_high_categories) rather than the constant-1 presence
    array. Shared by cumulative_dimension_metrics, backfill_dimension_batches
    and dimension_scoring_diagnostics so all three see identical data; the
    diagnostics would otherwise be able to disagree with the evaluation they
    are supposed to explain."""
    return {
        dim.name: ml_assist._dimension_examples_full(engine, dim.name, levels=dim.multi_value) for dim in schema
    }


def dimension_scoring_diagnostics(
    engine: Engine, schema: list[ml_assist.DimensionSchema]
) -> list[tuple[str, str]]:
    """(dimension, reason) for every dimension that cannot be scored yet, in
    schema order -- so `classify assisted` can say WHY no dimension metrics
    table appeared instead of silently skipping the whole block, which is
    indistinguishable from the feature being broken.

    Mirrors the guards in _cumulative_dimension_metrics_from_examples,
    _fit_dimension_category_classifiers and
    _fit_dimension_single_value_classifier rather than re-deriving them, and
    reports the closest category/class to clearing its bar so the reviewer
    can see how far off it is. A dimension that scores fine is absent from
    the result."""
    examples_by_dimension = _dimension_examples_by_name(engine, schema)
    reasons: list[tuple[str, str]] = []

    for dim in schema:
        examples = examples_by_dimension.get(dim.name, [])
        labels = [label for _text, label, _ts in examples]
        if len(labels) < ml_assist.MIN_TOTAL_EXAMPLES:
            reasons.append(
                (dim.name, f"only {len(labels)} classified record(s); needs {ml_assist.MIN_TOTAL_EXAMPLES}")
            )
            continue

        if dim.multi_value:
            scoreable = []
            best: tuple[int, int, str] | None = None  # (positives, negatives, category)
            for category in dim.categories:
                positives = sum(1 for m in labels if category in ml_assist._parse_high_categories(m))
                negatives = len(labels) - positives
                if positives >= ml_assist.MIN_POSITIVES_PER_CATEGORY and negatives >= ml_assist.MIN_EXAMPLES_PER_CLASS:
                    scoreable.append(category)
                elif best is None or (min(positives, negatives), positives) > (min(best[0], best[1]), best[0]):
                    best = (positives, negatives, category)
            if scoreable:
                continue
            closest = (
                f"; closest is {best[2]!r} at {best[0]} high / {best[1]} not-high" if best else ""
            )
            reason = (
                f"no category yet has {ml_assist.MIN_POSITIVES_PER_CATEGORY}+ 'high' and "
                f"{ml_assist.MIN_EXAMPLES_PER_CLASS}+ not-high example(s) across {len(labels)} "
                f"classified record(s){closest}"
            )
            reasons.append((dim.name, reason))
        else:
            # Mirrors _fit_dimension_single_value_classifier's guard, which
            # DROPS classes below MIN_EXAMPLES_PER_CLASS and fits on the
            # rest -- so a rare class is not on its own a reason the
            # dimension cannot be scored.
            class_counts: dict[str, int] = {}
            for label in labels:
                class_counts[label] = class_counts.get(label, 0) + 1
            eligible = {c for c, n in class_counts.items() if n >= ml_assist.MIN_EXAMPLES_PER_CLASS}
            kept = sum(n for c, n in class_counts.items() if c in eligible)
            if len(eligible) >= 2 and kept >= ml_assist.MIN_TOTAL_EXAMPLES:
                continue
            detail = ", ".join(f"{c!r} ({n})" for c, n in sorted(class_counts.items(), key=lambda kv: -kv[1]))
            if len(eligible) < 2:
                reason = f"fewer than 2 classes reach {ml_assist.MIN_EXAMPLES_PER_CLASS} example(s) -- have {detail}"
            else:
                reason = (
                    f"only {kept} record(s) left after dropping classes below "
                    f"{ml_assist.MIN_EXAMPLES_PER_CLASS} example(s); needs "
                    f"{ml_assist.MIN_TOTAL_EXAMPLES} -- have {detail}"
                )
            reasons.append((dim.name, reason))

    return reasons


def evaluate_dimension_model_multi_value_from_oof(
    labels: list[str], oof_by_category: dict[str, list]
) -> ModelMetrics | None:
    """A multi-value dimension model's per-category out-of-fold predictions
    (ml_assist._fit_dimension_category_classifiers) scored one category at a
    time against the true high/not-high target parsed from `labels` (via
    ml_assist._parse_high_categories), then averaged into one ModelMetrics
    (_average_model_metrics) -- the per-dimension score
    cumulative_dimension_metrics feeds into its cross-dimension average.

    `labels` must therefore be the {category: level} maps the fit itself was
    given, i.e. from ml_assist._dimension_examples_full(..., levels=True) --
    scoring against category presence instead would compare the predictions
    to a constant-1 vector (see ml_assist._parse_high_categories).

    A category is skipped if it happens to be single-class across every
    `labels` example (can't score a classifier with no positive/negative
    contrast to check it against) -- normally a category that is either
    always or never rated high so far. None if no category could be scored
    at all."""
    if not oof_by_category:
        return None
    per_category: list[ModelMetrics] = []
    for category, oof_predictions in oof_by_category.items():
        true_high = [1 if category in ml_assist._parse_high_categories(levels_json) else 0 for levels_json in labels]
        if len(set(true_high)) < 2:
            continue
        per_category.append(_metrics_from_predictions(true_high, oof_predictions))
    if not per_category:
        return None
    return _average_model_metrics(per_category)


def _cumulative_multi_value_dimension_metrics(
    dim: ml_assist.DimensionSchema, texts: list[str], labels: list[str]
) -> dict[str, ModelMetrics] | None:
    """cumulative_dimension_metrics's per-multi-value-dimension step: every
    MODEL_SPECS model type is fit ONCE on the full (texts, labels)
    (ml_assist._fit_dimension_category_classifiers, one classifier per
    category), each category scored by its own out-of-fold cross-validation
    predictions from that same fit (evaluate_dimension_model_multi_value_from_oof)
    -- no separate holdout split needed; every category gets its own
    properly-stratified CV, stratified by that specific category's own
    presence/absence balance."""
    metrics_by_type: dict[str, ModelMetrics] = {}
    for model_type in ml_assist.MODEL_TYPES:
        try:
            fitted = ml_assist._fit_dimension_category_classifiers(dim, texts, labels, model_type)
        except Exception:  # noqa: BLE001, S112 -- embedding backend unavailable, same degrade-gracefully pattern as elsewhere
            continue
        if fitted is None:
            continue
        _model, oof_by_category = fitted
        m = evaluate_dimension_model_multi_value_from_oof(labels, oof_by_category)
        if m is not None:
            metrics_by_type[model_type] = m
    return metrics_by_type or None


def _cumulative_single_value_dimension_metrics(
    dim: ml_assist.DimensionSchema, texts: list[str], labels: list[str]
) -> dict[str, ModelMetrics] | None:
    """cumulative_dimension_metrics's per-single-value-dimension step
    (evidence_type). Scores the SAME fit that produces the production model
    -- ml_assist._fit_dimension_single_value_classifier, once per model type
    -- against its own out-of-fold predictions.

    This deliberately does NOT go through cumulative_metrics(), which was
    the earlier route. cumulative_metrics refuses outright if ANY class sits
    below MIN_EXAMPLES_PER_CLASS, whereas the fit it was supposed to be
    mirroring drops those rare classes and trains on the rest. The two
    disagreed, and the stricter one won: a single record carrying a
    one-of-a-kind evidence_type (e.g. one 'prototype' among 19) made the
    whole dimension permanently unscoreable while its suggestions kept being
    shown -- so the reviewer could never compare model types for it. Scoring
    the filtered set the model actually trained on reports on the model that
    actually exists, and rare-class singletons stop being a silent veto.

    The reported metrics are therefore over the filtered labels, not all of
    them; ModelMetrics.n makes that visible."""
    metrics_by_type: dict[str, ModelMetrics] = {}
    for model_type in ml_assist.MODEL_TYPES:
        try:
            fitted = ml_assist._fit_dimension_single_value_classifier(dim, texts, labels, model_type)
        except Exception:  # noqa: BLE001, S112 -- embedding backend unavailable, same degrade-gracefully pattern as elsewhere
            continue
        if fitted is None:
            continue
        _model, filtered_labels, oof_predictions = fitted
        metrics_by_type[model_type] = _metrics_from_predictions(filtered_labels, oof_predictions)
    return metrics_by_type or None


@dataclass
class DimensionEvaluation:
    per_dimension_metrics: dict[str, dict[str, ModelMetrics]]  # dimension -> model_type -> ModelMetrics
    aggregate_metrics: dict[str, ModelMetrics]  # model_type -> averaged across every dimension that scored it


def _cumulative_dimension_metrics_from_examples(
    examples_by_dimension: dict[str, list[tuple[str, str, str]]], schema: list[ml_assist.DimensionSchema]
) -> DimensionEvaluation | None:
    """The actual evaluation logic behind cumulative_dimension_metrics(),
    factored out to take already-fetched (text, label, timestamp) example
    lists rather than querying the engine directly -- backfill_dimension_
    batches() reuses this against a timestamp-filtered view of the same data
    to replay historical checkpoints without re-querying per batch."""
    per_dimension_metrics: dict[str, dict[str, ModelMetrics]] = {}
    for dim in schema:
        examples = examples_by_dimension.get(dim.name, [])
        if len(examples) < ml_assist.MIN_TOTAL_EXAMPLES:
            continue
        texts = [text for text, _label, _ts in examples]
        labels = [label for _text, label, _ts in examples]

        if dim.multi_value:
            metrics = _cumulative_multi_value_dimension_metrics(dim, texts, labels)
        else:
            metrics = _cumulative_single_value_dimension_metrics(dim, texts, labels)
        if metrics:
            per_dimension_metrics[dim.name] = metrics

    if not per_dimension_metrics:
        return None

    scores_by_model_type: dict[str, list[ModelMetrics]] = {}
    for dim_metrics in per_dimension_metrics.values():
        for model_type, m in dim_metrics.items():
            scores_by_model_type.setdefault(model_type, []).append(m)
    aggregate_metrics = {mt: _average_model_metrics(ms) for mt, ms in scores_by_model_type.items()}
    return DimensionEvaluation(per_dimension_metrics=per_dimension_metrics, aggregate_metrics=aggregate_metrics)


def cumulative_dimension_metrics(engine: Engine, schema: list[ml_assist.DimensionSchema]) -> DimensionEvaluation | None:
    """The dimension-side counterpart of cumulative_metrics() (see module
    docstring): every MODEL_SPECS model type is scored per-dimension via
    out-of-fold cross-validation over the full cumulative example set for
    that dimension, then averaged ACROSS the 6 dimensions into one aggregate
    score per model type -- what `classify assisted` shows the reviewer when
    picking which model type should drive suggestions for all 6 dimensions
    at once. None if no dimension had enough data to evaluate at all."""
    examples_by_dimension = _dimension_examples_by_name(engine, schema)
    return _cumulative_dimension_metrics_from_examples(examples_by_dimension, schema)


@dataclass
class BatchRecord:
    batch: int
    timestamp: str
    n_evaluated: int  # cumulative example count as of this batch (== n_trained_on -- every example both trains and,
    n_trained_on: int  # via CV rotation, gets an out-of-fold prediction; kept as two fields for JSON schema stability)
    metrics: dict[str, ModelMetrics]
    active_model: str  # which model type actually drove suggestions during this batch


def _read_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(path: Path = DEFAULT_METRICS_PATH) -> list[BatchRecord]:
    """[] if the file doesn't exist yet -- nothing has ever been recorded."""
    raw = _read_payload(path)
    return [
        BatchRecord(
            batch=b["batch"],
            timestamp=b["timestamp"],
            n_evaluated=b["n_evaluated"],
            n_trained_on=b["n_trained_on"],
            metrics={model_type: ModelMetrics(**m) for model_type, m in b["metrics"].items()},
            active_model=b["active_model"],
        )
        for b in raw.get("batches", [])
    ]


def load_current_active_model(path: Path = DEFAULT_METRICS_PATH) -> str | None:
    """The model type the reviewer most recently chose to drive suggestions
    going forward -- distinct from any single BatchRecord.active_model, which
    only ever records which model was active DURING that already-completed
    batch. None if nothing's been recorded yet (callers fall back to
    ml_assist.DEFAULT_ACTIVE_MODEL). This is what makes a reviewer's pick at
    one retrain still be the starting model the NEXT `classify assisted`
    session opens with, even if that next session is a different process."""
    return _read_payload(path).get("current_active_model")


def save_current_active_model(model_type: str, path: Path = DEFAULT_METRICS_PATH) -> None:
    """Persists the reviewer's retrain-time model choice for the next
    session to pick up (see load_current_active_model). Preserves whatever
    batch history already exists in the file -- this only ever touches the
    current_active_model key."""
    payload = _read_payload(path)
    payload["current_active_model"] = model_type
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_pending_batch(path: Path = DEFAULT_METRICS_PATH) -> list[tuple[str, str]]:
    """(text, final_label) pairs decided so far toward the batch that hasn't
    retrained yet. [] if nothing's pending -- a fresh corpus, or right after
    a retrain clears it (see save_pending_batch). This is what lets
    `classify assisted` resume an aborted session's batch progress instead of
    losing count of how close the next retrain is: every decision made in
    that command is persisted here immediately (not just batched in memory),
    so a session that quits or crashes mid-batch doesn't reset the countdown
    for the next `classify assisted` run."""
    raw = _read_payload(path).get("pending_batch", [])
    return [(pair[0], pair[1]) for pair in raw]


def save_pending_batch(pending: list[tuple[str, str]], path: Path = DEFAULT_METRICS_PATH) -> None:
    """Persists the in-progress batch's (text, label) pairs -- call after
    every decision made in `classify assisted`, and with [] right after a
    retrain fires. Preserves whatever batch history and current_active_model
    already exist in the file, same as save_current_active_model."""
    payload = _read_payload(path)
    payload["pending_batch"] = [[text, label] for text, label in pending]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_history(history: list[BatchRecord], path: Path) -> None:
    """Rewrites the batches list, preserving any current_active_model
    already in the file (see save_current_active_model) -- this function
    only owns the batch history, not the live model selection."""
    payload = _read_payload(path)
    payload["batches"] = [
        {
            "batch": b.batch,
            "timestamp": b.timestamp,
            "n_evaluated": b.n_evaluated,
            "n_trained_on": b.n_trained_on,
            "metrics": {model_type: asdict(m) for model_type, m in b.metrics.items()},
            "active_model": b.active_model,
        }
        for b in history
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_batch(
    record: BatchRecord,
    path: Path = DEFAULT_METRICS_PATH,
    plot_path: Path = DEFAULT_METRICS_PLOT_PATH,
) -> list[BatchRecord]:
    """Appends/replaces `record` in the on-disk history (keyed by batch
    number, so re-running a batch overwrites rather than duplicates it),
    rewrites the JSON file, and regenerates the plot -- the single entry
    point both the live retrain loop and backfill_batches() go through, so
    the file and plot can never drift out of sync with each other."""
    history = [b for b in load_history(path) if b.batch != record.batch]
    history.append(record)
    history.sort(key=lambda b: b.batch)
    _write_history(history, path)
    plot_history(history, plot_path)
    return history


def plot_history(history: list[BatchRecord], out_path: Path = DEFAULT_METRICS_PLOT_PATH) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    # One color per classifier family, shared across both feature
    # representations -- solid line for tfidf_*, dashed for embedding_* --
    # so with 8 series the plot still reads as "4 classifiers x 2 feature
    # kinds" rather than 8 arbitrarily-colored lines.
    classifier_colors: dict[str, str] = {
        "logistic_regression": "tab:blue",
        "decision_tree": "tab:orange",
        "random_forest": "tab:green",
        "gradient_boosting": "tab:red",
    }
    feature_linestyles: dict[str, str] = {"tfidf": "-", "embedding": "--"}

    model_types = sorted({model_type for b in history for model_type in b.metrics})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_type in model_types:
        xs = [b.batch for b in history if model_type in b.metrics]
        ys = [b.metrics[model_type].f1_macro for b in history if model_type in b.metrics]
        spec = ml_assist.MODEL_SPECS.get(model_type)
        color = classifier_colors.get(spec[1]) if spec else None
        linestyle = feature_linestyles.get(spec[0], "-") if spec else "-"
        ax.plot(
            xs, ys, marker="o", label=model_type.replace("_", " "),
            color=color, linestyle=linestyle,
        )

    # Ring around whichever model's point was the one actually driving
    # suggestions that batch, so the plot also shows the human's picks over time.
    active_xs = [b.batch for b in history if b.active_model in b.metrics]
    active_ys = [b.metrics[b.active_model].f1_macro for b in history if b.active_model in b.metrics]
    if active_xs:
        ax.scatter(
            active_xs, active_ys, s=160, facecolors="none", edgecolors="black", linewidths=1.5,
            zorder=5, label="active model",
        )

    ax.set_xlabel("Retrain batch")
    ax.set_ylabel("Macro F1 (topical_inclusion, out-of-fold CV over all decisions to date)")
    ax.set_title("ML-assist suggestion model quality by batch")
    ax.set_xticks([b.batch for b in history])
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def backfill_batches(
    engine: Engine,
    batch_size: int = 20,
    path: Path = DEFAULT_METRICS_PATH,
    plot_path: Path = DEFAULT_METRICS_PLOT_PATH,
) -> list[BatchRecord]:
    """Reconstructs the full per-batch metrics history from every human
    topical_inclusion decision made so far, in the order they were actually
    decided (ml_assist._labeled_examples_full). Batch k's metrics come from
    cumulative_metrics() over every example up to and including batch k
    (out-of-fold cross-validation over the full cumulative set, see module
    docstring) -- the same evaluation the live retrain loop uses, just
    replayed after the fact. Batches whose cumulative example count is
    still below ml_assist.MIN_TOTAL_EXAMPLES, or too class-imbalanced for
    _fit_with_cv's CV to run at all (cumulative_metrics returns None), are
    skipped -- that's normally just the first batch or two. Overwrites (not
    appends to) the file, so it's safe to re-run any time the batch_size
    used historically is uncertain or corrected. active_model is always
    recorded as ml_assist.DEFAULT_ACTIVE_MODEL for backfilled batches --
    there was no live reviewer choice to reconstruct for sessions that
    predate this feature."""
    examples = ml_assist._labeled_examples_full(engine)  # (text, label, timestamp), chronological
    history: list[BatchRecord] = []
    batch_number = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        cumulative = examples[: start + len(batch)]
        if len(cumulative) < ml_assist.MIN_TOTAL_EXAMPLES:
            continue

        cumulative_texts = [text for text, _label, _ts in cumulative]
        cumulative_labels = [label for _text, label, _ts in cumulative]

        evaluation = cumulative_metrics(cumulative_texts, cumulative_labels)
        if evaluation is None:
            continue  # cumulative set too small/imbalanced for CV to run yet

        batch_number += 1
        history.append(
            BatchRecord(
                batch=batch_number,
                timestamp=batch[-1][2],
                n_evaluated=evaluation.n,
                n_trained_on=evaluation.n,
                metrics=evaluation.metrics,
                active_model=ml_assist.DEFAULT_ACTIVE_MODEL,
            )
        )

    _write_history(history, path)
    plot_history(history, plot_path)
    return history


@dataclass
class DimensionBatchRecord:
    batch: int
    timestamp: str
    per_dimension_metrics: dict[str, dict[str, ModelMetrics]]  # dimension -> model_type -> ModelMetrics
    aggregate_metrics: dict[str, ModelMetrics]  # model_type -> averaged across every dimension scored this batch
    active_dimension_model: str  # which model type actually drove dimension suggestions during this batch


def load_dimension_history(path: Path = DEFAULT_DIMENSION_METRICS_PATH) -> list[DimensionBatchRecord]:
    """[] if the file doesn't exist yet -- nothing has ever been recorded."""
    raw = _read_payload(path)
    return [
        DimensionBatchRecord(
            batch=b["batch"],
            timestamp=b["timestamp"],
            per_dimension_metrics={
                dim: {model_type: ModelMetrics(**m) for model_type, m in dim_metrics.items()}
                for dim, dim_metrics in b["per_dimension_metrics"].items()
            },
            aggregate_metrics={model_type: ModelMetrics(**m) for model_type, m in b["aggregate_metrics"].items()},
            active_dimension_model=b["active_dimension_model"],
        )
        for b in raw.get("batches", [])
    ]


def load_current_active_dimension_model(path: Path = DEFAULT_DIMENSION_METRICS_PATH) -> str | None:
    """The model type the reviewer most recently chose to drive suggestions
    for all 6 classification dimensions -- see load_current_active_model,
    same shape, separate file/key since the topical_inclusion and dimension
    choices are independent."""
    return _read_payload(path).get("current_active_dimension_model")


def save_current_active_dimension_model(model_type: str, path: Path = DEFAULT_DIMENSION_METRICS_PATH) -> None:
    """Persists the reviewer's retrain-time dimension-model choice for the
    next session to pick up (see load_current_active_dimension_model)."""
    payload = _read_payload(path)
    payload["current_active_dimension_model"] = model_type
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_dimension_history(history: list[DimensionBatchRecord], path: Path) -> None:
    """Rewrites the batches list, preserving any current_active_dimension_model
    already in the file -- mirrors _write_history."""
    payload = _read_payload(path)
    payload["batches"] = [
        {
            "batch": b.batch,
            "timestamp": b.timestamp,
            "per_dimension_metrics": {
                dim: {model_type: asdict(m) for model_type, m in dim_metrics.items()}
                for dim, dim_metrics in b.per_dimension_metrics.items()
            },
            "aggregate_metrics": {model_type: asdict(m) for model_type, m in b.aggregate_metrics.items()},
            "active_dimension_model": b.active_dimension_model,
        }
        for b in history
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_dimension_batch(
    record: DimensionBatchRecord,
    path: Path = DEFAULT_DIMENSION_METRICS_PATH,
    plot_path: Path = DEFAULT_DIMENSION_METRICS_PLOT_PATH,
) -> list[DimensionBatchRecord]:
    """Mirrors record_batch() for the dimension file/plot -- the single
    entry point both the live retrain loop and backfill_dimension_batches()
    go through."""
    history = [b for b in load_dimension_history(path) if b.batch != record.batch]
    history.append(record)
    history.sort(key=lambda b: b.batch)
    _write_dimension_history(history, path)
    plot_dimension_history(history, plot_path)
    return history


def plot_dimension_history(history: list[DimensionBatchRecord], out_path: Path = DEFAULT_DIMENSION_METRICS_PLOT_PATH) -> None:
    """Mirrors plot_history(): same color-by-classifier-family/
    linestyle-by-feature-kind encoding, but plots each model type's
    aggregate_metrics[model_type].f1_macro (the cross-dimension average) by
    batch instead of a single dimension's own score."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    classifier_colors: dict[str, str] = {
        "logistic_regression": "tab:blue",
        "decision_tree": "tab:orange",
        "random_forest": "tab:green",
        "gradient_boosting": "tab:red",
    }
    feature_linestyles: dict[str, str] = {"tfidf": "-", "embedding": "--"}

    model_types = sorted({model_type for b in history for model_type in b.aggregate_metrics})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_type in model_types:
        xs = [b.batch for b in history if model_type in b.aggregate_metrics]
        ys = [b.aggregate_metrics[model_type].f1_macro for b in history if model_type in b.aggregate_metrics]
        spec = ml_assist.MODEL_SPECS.get(model_type)
        color = classifier_colors.get(spec[1]) if spec else None
        linestyle = feature_linestyles.get(spec[0], "-") if spec else "-"
        ax.plot(
            xs, ys, marker="o", label=model_type.replace("_", " "),
            color=color, linestyle=linestyle,
        )

    active_xs = [b.batch for b in history if b.active_dimension_model in b.aggregate_metrics]
    active_ys = [
        b.aggregate_metrics[b.active_dimension_model].f1_macro
        for b in history if b.active_dimension_model in b.aggregate_metrics
    ]
    if active_xs:
        ax.scatter(
            active_xs, active_ys, s=160, facecolors="none", edgecolors="black", linewidths=1.5,
            zorder=5, label="active model",
        )

    ax.set_xlabel("Retrain batch")
    ax.set_ylabel("Macro F1 (6-dimension average, out-of-fold CV over all decisions to date)")
    ax.set_title("ML-assist dimension-suggestion model quality by batch")
    ax.set_xticks([b.batch for b in history])
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def backfill_dimension_batches(
    engine: Engine,
    schema: list[ml_assist.DimensionSchema],
    batch_size: int = 20,
    path: Path = DEFAULT_DIMENSION_METRICS_PATH,
    plot_path: Path = DEFAULT_DIMENSION_METRICS_PLOT_PATH,
) -> list[DimensionBatchRecord]:
    """The dimension-side counterpart of backfill_batches(): replays the
    SAME topical_inclusion chronology/--batch-size boundaries (so a given
    checkpoint's cutoff timestamp means the same thing in both files), but
    numbers its own batches independently and densely -- dimension data
    only starts accumulating once records reach 'include' and then get
    fully classified, so a topical checkpoint may have no scoreable
    dimension data yet even once the topical file does (see module
    docstring). Overwrites (not appends to) the file, same as
    backfill_batches()."""
    topical_examples = ml_assist._labeled_examples_full(engine)  # defines the shared cutoff timestamps
    full_examples_by_dimension = _dimension_examples_by_name(engine, schema)

    history: list[DimensionBatchRecord] = []
    batch_number = 0
    for start in range(0, len(topical_examples), batch_size):
        batch = topical_examples[start : start + batch_size]
        cutoff_timestamp = batch[-1][2]
        filtered_by_dimension = {
            name: [e for e in examples if e[2] <= cutoff_timestamp]
            for name, examples in full_examples_by_dimension.items()
        }
        evaluation = _cumulative_dimension_metrics_from_examples(filtered_by_dimension, schema)
        if evaluation is None:
            continue

        batch_number += 1
        history.append(
            DimensionBatchRecord(
                batch=batch_number,
                timestamp=cutoff_timestamp,
                per_dimension_metrics=evaluation.per_dimension_metrics,
                aggregate_metrics=evaluation.aggregate_metrics,
                active_dimension_model=ml_assist.DEFAULT_ACTIVE_MODEL,
            )
        )

    _write_dimension_history(history, path)
    plot_dimension_history(history, plot_path)
    return history
