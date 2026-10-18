"""Phase 12: AI-human agreement metrics.

Deliberately never called "inter-rater reliability" unqualified anywhere in
this module's output -- one "rater" is a language model, not a second human
coder, and that distinction matters for how the numbers should be read. All
report text says "AI-human agreement" explicitly (docs/TECHNICAL_IMPLEMENTATION_
PLAN.md section 18).

Handles both single-label dimensions (a plain string label, e.g.
topical_inclusion's include/exclude/uncertain) and multi-label dimensions (a
JSON-array-encoded label, per the schema's own `label TEXT -- JSON array if
multi-label` convention) -- which kind a dimension is is detected from the
stored label's shape, not hardcoded, so this works for the 6-dimensional
content classification once that exists, not just screening.

⚠ Weighting. The Phase 11 validation sample is drawn with UNEQUAL probability
per stratum (D-29: 200 include / 130 exclude / 70 uncertain, against population
strata of 1,067 / 6,545 / 238). Every function here computes both an unweighted
statistic -- valid only as a description of the sample itself -- and, when a
sampling manifest with weights is available, a design-weighted estimate, which
is the only one that describes the corpus. Reporting the unweighted include rate
of this sample as if it were the corpus would overstate it roughly threefold.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from sqlalchemy import Engine, select

from review_pipeline.src.db.tables import reflect_table


@dataclass
class PairedDecision:
    record_id: str
    ai_label: str
    human_label: str
    stratum: str | None = None
    # Design weight N_h/n_h: how many corpus records this one stands for.
    # 1.0 means "unweighted", which is correct only for an equal-probability
    # sample or a census.
    weight: float = 1.0
    # The AI's own confidence for its label, when recorded. Combined with the
    # label it forms the ordinal score that _weighted_auc ranks on -- which is
    # how a screener that orders well but cuts badly can be told apart from one
    # that is simply wrong (D-36).
    ai_confidence: str | None = None


def _latest_labels_by_record(rows: list[dict]) -> dict[str, str]:
    """Most recent label per record_id (by timestamp), if a record was somehow
    coded more than once by the same coder-type for this dimension."""
    latest: dict[str, tuple[str, str]] = {}  # record_id -> (timestamp, label)
    for row in rows:
        record_id, label, timestamp = row["record_id"], row["label"], row["timestamp"]
        if record_id not in latest or timestamp > latest[record_id][0]:
            latest[record_id] = (timestamp, label)
    return {record_id: label for record_id, (_ts, label) in latest.items()}


def _latest_confidence_by_record(rows: list[dict]) -> dict[str, str | None]:
    """Same most-recent-wins rule as _latest_labels_by_record, for confidence."""
    latest: dict[str, tuple[str, str | None]] = {}
    for row in rows:
        record_id, timestamp = row["record_id"], row["timestamp"]
        if record_id not in latest or timestamp > latest[record_id][0]:
            latest[record_id] = (timestamp, row.get("confidence"))
    return {record_id: confidence for record_id, (_ts, confidence) in latest.items()}


def load_sampling_weights(manifest_path: Path | None = None) -> tuple[dict[str, str], dict[str, float]]:
    """(record_id -> stratum, stratum -> design weight) for the built validation
    sample, or ({}, {}) when no weighted sample exists.

    Returning empty dicts is the honest default: it makes every pair weight 1.0,
    which is correct for a census or an equal-probability sample and is exactly
    what the caller should fall back to when there is no manifest to read.
    """
    from review_pipeline.src.validation.sampling import (
        DEFAULT_MANIFEST_PATH,
        load_validation_sample,
    )

    path = manifest_path or DEFAULT_MANIFEST_PATH
    if not path.exists():
        return {}, {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    weights = {k: float(v) for k, v in (manifest.get("sampling_weights") or {}).items()}
    if not weights:
        return {}, {}
    membership = load_validation_sample() or {}
    return membership, weights


def paired_decisions(
    engine: Engine, dimension: str, manifest_path: Path | None = None
) -> list[PairedDecision]:
    """Records that have BOTH an AI and a human label for this dimension --
    agreement can only be computed where both exist.

    Each pair carries its stratum and design weight when the validation sample
    was drawn with unequal probability (D-29), so downstream estimates can
    describe the corpus rather than the sample."""
    classifications_ai = reflect_table(engine, "classifications_ai")
    classifications_human = reflect_table(engine, "classifications_human")

    with engine.connect() as conn:
        ai_rows = [
            dict(r)
            for r in conn.execute(
                select(classifications_ai).where(classifications_ai.c.dimension == dimension)
            ).mappings().all()
        ]
        human_rows = [
            dict(r)
            for r in conn.execute(
                select(classifications_human).where(classifications_human.c.dimension == dimension)
            ).mappings().all()
        ]

    ai_by_record = _latest_labels_by_record(ai_rows)
    ai_confidence_by_record = _latest_confidence_by_record(ai_rows)
    human_by_record = _latest_labels_by_record(human_rows)

    membership, weights = load_sampling_weights(manifest_path)

    return [
        PairedDecision(
            record_id=rid,
            ai_label=ai_by_record[rid],
            human_label=human_by_record[rid],
            stratum=membership.get(rid),
            weight=weights.get(membership.get(rid, ""), 1.0),
            ai_confidence=ai_confidence_by_record.get(rid),
        )
        for rid in sorted(set(ai_by_record) & set(human_by_record))
    ]


def _is_multi_label(pairs: list[PairedDecision]) -> bool:
    for p in pairs:
        for label in (p.ai_label, p.human_label):
            try:
                parsed = json.loads(label)
            except (json.JSONDecodeError, TypeError):
                return False
            if not isinstance(parsed, list):
                return False
    return bool(pairs)


@dataclass
class SingleLabelAgreementResult:
    n: int
    percent_agreement: float
    cohen_kappa: float | None  # None if undefined (e.g. only one class present)
    confusion_matrix: dict[str, dict[str, int]]  # confusion_matrix[human_label][ai_label] = count
    false_negative_rate: float | None  # human='include', ai!='include', as a fraction of human='include' cases
    false_negatives: list[str] = field(default_factory=list)  # record_ids of the FN case above
    # --- design-weighted estimates (None when the sample is unweighted) ------
    # These describe the CORPUS; the fields above describe the SAMPLE. When the
    # sample is drawn with unequal probability they differ, sometimes greatly.
    weighted_percent_agreement: float | None = None
    weighted_cohen_kappa: float | None = None
    weighted_false_negative_rate: float | None = None
    weighted_precision: float | None = None  # P(human=positive | ai=positive)
    is_weighted: bool = False
    effective_n: float | None = None  # Kish effective sample size
    # --- kappa in context (D-36) --------------------------------------------
    # kappa_max is the ceiling the observed marginals impose; PABAK and AC1 are
    # the two standard prevalence-robust alternatives; AUC scores the ordering
    # rather than the cut-point. See the block comment above _marginal_profile.
    kappa_max: float | None = None
    pabak: float | None = None
    gwet_ac1: float | None = None
    auc: float | None = None
    weighted_kappa_max: float | None = None
    weighted_pabak: float | None = None
    weighted_gwet_ac1: float | None = None
    weighted_auc: float | None = None
    # stratum -> (adjudicated, sampled). A stratum with sampled > adjudicated
    # contributes NOTHING to the weighted estimates above, which are therefore
    # conditional on the strata that are actually complete -- the failure mode
    # that produced the withdrawn 51.9% (D-35), made visible in the artifact.
    stratum_coverage: dict[str, tuple[int, int]] = field(default_factory=dict)


def _weighted_kappa(pairs: list[PairedDecision]) -> float | None:
    """Cohen's kappa computed on design-weighted cell counts.

    Same formula as the unweighted case -- (Po - Pe) / (1 - Pe) -- but every
    count is a sum of weights rather than a count of rows, so each stratum
    contributes in proportion to the corpus, not to the sample.
    """
    labels = sorted({p.ai_label for p in pairs} | {p.human_label for p in pairs})
    if len(labels) < 2:
        return None
    total = sum(p.weight for p in pairs)
    if total <= 0:
        return None

    observed = sum(p.weight for p in pairs if p.ai_label == p.human_label) / total
    ai_marginal = {lab: sum(p.weight for p in pairs if p.ai_label == lab) / total for lab in labels}
    human_marginal = {lab: sum(p.weight for p in pairs if p.human_label == lab) / total for lab in labels}
    expected = sum(ai_marginal[lab] * human_marginal[lab] for lab in labels)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


# --- Beyond kappa (D-36) ----------------------------------------------------
#
# Cohen's kappa is not a clean measure of how well two coders agree when their
# marginals are far apart, and on this sample they are: the AI's marginal IS the
# sampling design (200/130/70 by construction), and the human never used
# `uncertain` at all. Two consequences, both well documented in the methods
# literature and both reported here so a reader does not have to reconstruct
# them:
#
#   * kappa has a CEILING below 1 whenever the marginals differ. With the
#     marginals held fixed, the largest attainable observed agreement is
#     sum_l min(p_ai(l), p_human(l)), and kappa_max is that value put through
#     the same formula. Reporting kappa without kappa_max invites a reader to
#     compare 0.11 against 1.0 when the reachable maximum is 0.16, and to blame
#     the coder for what is arithmetic. (Cohen 1960's own bound; Umesh et al.)
#   * kappa is depressed by skewed prevalence even when agreement is high --
#     the first Feinstein & Cicchetti (1990) paradox. PABAK (Byrt, Bishop &
#     Carlin 1993) removes both the prevalence and the bias adjustment, and
#     Gwet's AC1 (2008) replaces Cohen's chance term with one that does not
#     explode under skew. AC1 is prevalence-robust but NOT bias-robust, so the
#     gap between AC1 and kappa is itself the diagnostic: a large gap says the
#     disagreement lives in the marginals rather than in the individual calls.
#
# The Landis & Koch (1977) verbal bands are deliberately not printed. Their own
# authors called them arbitrary, they were never validated, and against a
# ceiling of 0.16 a label like "slight" describes the sampling design rather
# than the screener.


def _marginal_profile(
    triples: list[tuple[str, str, float]],
) -> tuple[list[str], float, dict[str, float], dict[str, float], float] | None:
    """(labels, Po, AI marginal, human marginal, Pe) over weighted cell mass."""
    labels = sorted({a for a, _h, _w in triples} | {h for _a, h, _w in triples})
    total = sum(w for _a, _h, w in triples)
    if len(labels) < 2 or total <= 0:
        return None
    observed = sum(w for a, h, w in triples if a == h) / total
    ai_marginal = {lab: sum(w for a, _h, w in triples if a == lab) / total for lab in labels}
    human_marginal = {lab: sum(w for _a, h, w in triples if h == lab) / total for lab in labels}
    expected = sum(ai_marginal[lab] * human_marginal[lab] for lab in labels)
    return labels, observed, ai_marginal, human_marginal, expected


def _triples(pairs: list[PairedDecision], weighted: bool) -> list[tuple[str, str, float]]:
    return [(p.ai_label, p.human_label, p.weight if weighted else 1.0) for p in pairs]


def _kappa_max(pairs: list[PairedDecision], weighted: bool) -> float | None:
    """The largest kappa attainable without changing either coder's marginals.

    Any per-record improvement is bounded by this; the distance from the
    observed kappa to it is the only part of the gap to 1.0 that a better
    screener could close."""
    profile = _marginal_profile(_triples(pairs, weighted))
    if profile is None:
        return None
    labels, _po, ai_marginal, human_marginal, pe = profile
    if pe >= 1.0:
        return None
    po_max = sum(min(ai_marginal[lab], human_marginal[lab]) for lab in labels)
    return (po_max - pe) / (1 - pe)


def _pabak(pairs: list[PairedDecision], weighted: bool) -> float | None:
    """Prevalence-and-bias-adjusted kappa: (q*Po - 1) / (q - 1)."""
    profile = _marginal_profile(_triples(pairs, weighted))
    if profile is None:
        return None
    labels, po, _ai, _human, _pe = profile
    q = len(labels)
    return (q * po - 1) / (q - 1)


def _gwet_ac1(pairs: list[PairedDecision], weighted: bool) -> float | None:
    """Gwet's AC1: kappa's structure with a chance term that does not inflate
    under skewed prevalence."""
    profile = _marginal_profile(_triples(pairs, weighted))
    if profile is None:
        return None
    labels, po, ai_marginal, human_marginal, _pe = profile
    q = len(labels)
    if q < 2:
        return None
    mean_prevalence = {lab: (ai_marginal[lab] + human_marginal[lab]) / 2 for lab in labels}
    pe_gamma = sum(pi * (1 - pi) for pi in mean_prevalence.values()) / (q - 1)
    if pe_gamma >= 1.0:
        return None
    return (po - pe_gamma) / (1 - pe_gamma)


# Higher = the AI leans harder towards `include`. Confidence breaks the label's
# ties in the direction the label points: a high-confidence exclude is the
# AI's strongest negative, a high-confidence include its strongest positive.
_ORDINAL_SCORE: dict[tuple[str, str | None], int] = {
    ("exclude", "high"): 0, ("exclude", "medium"): 1, ("exclude", "low"): 2, ("exclude", None): 1,
    ("uncertain", "high"): 3, ("uncertain", "medium"): 3, ("uncertain", "low"): 3, ("uncertain", None): 3,
    ("include", "low"): 4, ("include", "medium"): 5, ("include", "high"): 6, ("include", None): 5,
}


def _ordinal_score(pair: PairedDecision) -> int | None:
    return _ORDINAL_SCORE.get((pair.ai_label, pair.ai_confidence if pair.ai_confidence in ("high", "medium", "low") else None))


def _weighted_auc(pairs: list[PairedDecision], positive_label: str, weighted: bool) -> float | None:
    """Area under the ROC of the AI's ordinal (label, confidence) score against
    the human decision -- the probability that a randomly chosen human-positive
    record is scored above a randomly chosen human-negative one, ties at 0.5.

    This is the one statistic that separates RANKING from THRESHOLD. A screener
    can rank well (high AUC) and still decide badly (low kappa, low precision)
    if its cut-point sits in the wrong place -- and a cut-point is far cheaper
    to move than a model is to replace.
    """
    scored = [(s, p) for p in pairs if (s := _ordinal_score(p)) is not None]
    positives = [(s, p.weight if weighted else 1.0) for s, p in scored if p.human_label == positive_label]
    negatives = [(s, p.weight if weighted else 1.0) for s, p in scored if p.human_label != positive_label]
    if not positives or not negatives:
        return None

    negative_mass: dict[int, float] = {}
    for score, weight in negatives:
        negative_mass[score] = negative_mass.get(score, 0.0) + weight
    total_negative = sum(negative_mass.values())
    total_positive = sum(w for _s, w in positives)

    concordant = 0.0
    for score, weight in positives:
        for negative_score, mass in negative_mass.items():
            if score > negative_score:
                concordant += weight * mass
            elif score == negative_score:
                concordant += 0.5 * weight * mass
    return concordant / (total_positive * total_negative)


def _effective_n(pairs: list[PairedDecision]) -> float:
    """Kish's effective sample size: (sum w)^2 / sum(w^2).

    With very unequal weights the 400 adjudicated records carry the statistical
    information of rather fewer, and confidence intervals must be widened
    accordingly. Reporting n=400 next to a weighted estimate overstates its
    precision.
    """
    numerator = sum(p.weight for p in pairs) ** 2
    denominator = sum(p.weight * p.weight for p in pairs)
    return numerator / denominator if denominator else 0.0


def single_label_agreement(pairs: list[PairedDecision], positive_label: str = "include") -> SingleLabelAgreementResult:
    n = len(pairs)
    if n == 0:
        return SingleLabelAgreementResult(n=0, percent_agreement=0.0, cohen_kappa=None, confusion_matrix={}, false_negative_rate=None)

    agree = sum(1 for p in pairs if p.ai_label == p.human_label)
    percent_agreement = agree / n

    ai_labels = [p.ai_label for p in pairs]
    human_labels = [p.human_label for p in pairs]
    kappa = cohen_kappa_score(human_labels, ai_labels) if len(set(human_labels) | set(ai_labels)) > 1 else None

    confusion: dict[str, dict[str, int]] = {}
    for p in pairs:
        confusion.setdefault(p.human_label, {})
        confusion[p.human_label][p.ai_label] = confusion[p.human_label].get(p.ai_label, 0) + 1

    human_positive = [p for p in pairs if p.human_label == positive_label]
    false_negatives = [p.record_id for p in human_positive if p.ai_label != positive_label]
    fn_rate = len(false_negatives) / len(human_positive) if human_positive else None

    is_weighted = any(p.weight != 1.0 for p in pairs)
    total_weight = sum(p.weight for p in pairs)
    weighted_agreement = (
        sum(p.weight for p in pairs if p.ai_label == p.human_label) / total_weight
        if total_weight
        else None
    )

    human_positive_weight = sum(p.weight for p in human_positive)
    weighted_fn_rate = (
        sum(p.weight for p in human_positive if p.ai_label != positive_label) / human_positive_weight
        if human_positive_weight
        else None
    )
    ai_positive = [p for p in pairs if p.ai_label == positive_label]
    ai_positive_weight = sum(p.weight for p in ai_positive)
    weighted_precision = (
        sum(p.weight for p in ai_positive if p.human_label == positive_label) / ai_positive_weight
        if ai_positive_weight
        else None
    )

    return SingleLabelAgreementResult(
        n=n,
        percent_agreement=percent_agreement,
        cohen_kappa=kappa,
        confusion_matrix=confusion,
        false_negative_rate=fn_rate,
        false_negatives=false_negatives,
        weighted_percent_agreement=weighted_agreement,
        weighted_cohen_kappa=_weighted_kappa(pairs),
        weighted_false_negative_rate=weighted_fn_rate,
        weighted_precision=weighted_precision,
        is_weighted=is_weighted,
        effective_n=_effective_n(pairs),
        kappa_max=_kappa_max(pairs, weighted=False),
        pabak=_pabak(pairs, weighted=False),
        gwet_ac1=_gwet_ac1(pairs, weighted=False),
        auc=_weighted_auc(pairs, positive_label, weighted=False),
        weighted_kappa_max=_kappa_max(pairs, weighted=True) if is_weighted else None,
        weighted_pabak=_pabak(pairs, weighted=True) if is_weighted else None,
        weighted_gwet_ac1=_gwet_ac1(pairs, weighted=True) if is_weighted else None,
        weighted_auc=_weighted_auc(pairs, positive_label, weighted=True) if is_weighted else None,
    )


@dataclass
class MultiLabelAgreementResult:
    n: int
    mean_jaccard: float
    micro_f1: float
    macro_f1: float
    per_label: dict[str, dict[str, float]]  # label -> {precision, recall, f1}


def multi_label_agreement(pairs: list[PairedDecision]) -> MultiLabelAgreementResult:
    n = len(pairs)
    if n == 0:
        return MultiLabelAgreementResult(n=0, mean_jaccard=0.0, micro_f1=0.0, macro_f1=0.0, per_label={})

    ai_sets = [set(json.loads(p.ai_label)) for p in pairs]
    human_sets = [set(json.loads(p.human_label)) for p in pairs]
    all_labels = sorted(set().union(*ai_sets, *human_sets)) if (ai_sets or human_sets) else []

    def to_binary(sets: list[set]) -> list[list[int]]:
        return [[1 if label in s else 0 for label in all_labels] for s in sets]

    ai_binary = to_binary(ai_sets)
    human_binary = to_binary(human_sets)

    jaccard_scores = [
        len(a & h) / len(a | h) if (a | h) else 1.0 for a, h in zip(ai_sets, human_sets, strict=True)
    ]
    mean_jaccard = sum(jaccard_scores) / n

    if not all_labels:
        return MultiLabelAgreementResult(n=n, mean_jaccard=mean_jaccard, micro_f1=0.0, macro_f1=0.0, per_label={})

    precision, recall, f1, _support = precision_recall_fscore_support(
        human_binary, ai_binary, average=None, zero_division=0
    )
    per_label = {
        label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        for i, label in enumerate(all_labels)
    }
    micro = precision_recall_fscore_support(human_binary, ai_binary, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(human_binary, ai_binary, average="macro", zero_division=0)

    return MultiLabelAgreementResult(
        n=n, mean_jaccard=mean_jaccard, micro_f1=float(micro[2]), macro_f1=float(macro[2]), per_label=per_label
    )


@dataclass
class DimensionAgreement:
    dimension: str
    kind: str  # 'single_label' or 'multi_label'
    single: SingleLabelAgreementResult | None = None
    multi: MultiLabelAgreementResult | None = None


def _stratum_coverage(pairs: list[PairedDecision], manifest_path: Path | None = None) -> dict[str, tuple[int, int]]:
    """stratum -> (adjudicated in this pairing, sampled per the manifest)."""
    from review_pipeline.src.validation.sampling import DEFAULT_MANIFEST_PATH

    path = manifest_path or DEFAULT_MANIFEST_PATH
    if not path.exists():
        return {}
    sampled = {k: int(v) for k, v in (json.loads(path.read_text(encoding="utf-8")).get("stratum_counts") or {}).items()}
    adjudicated: dict[str, int] = {}
    for p in pairs:
        if p.stratum:
            adjudicated[p.stratum] = adjudicated.get(p.stratum, 0) + 1
    return {stratum: (adjudicated.get(stratum, 0), n) for stratum, n in sampled.items()}


def compute_dimension_agreement(engine: Engine, dimension: str) -> DimensionAgreement:
    pairs = paired_decisions(engine, dimension)
    if _is_multi_label(pairs):
        return DimensionAgreement(dimension=dimension, kind="multi_label", multi=multi_label_agreement(pairs))
    single = single_label_agreement(pairs)
    single.stratum_coverage = _stratum_coverage(pairs)
    return DimensionAgreement(dimension=dimension, kind="single_label", single=single)


def write_agreement_csv(results: list[DimensionAgreement], out_path: Path) -> None:
    """Long format: dimension, metric, value, n -- one row per metric."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dimension", "metric", "value", "n"])
        for r in results:
            if r.kind == "single_label" and r.single:
                s = r.single
                writer.writerow([r.dimension, "percent_agreement", s.percent_agreement, s.n])
                writer.writerow([r.dimension, "cohen_kappa", s.cohen_kappa if s.cohen_kappa is not None else "", s.n])
                for metric_name, value in (
                    ("cohen_kappa_max", s.kappa_max),
                    ("pabak", s.pabak),
                    ("gwet_ac1", s.gwet_ac1),
                    ("auc_ordinal_score", s.auc),
                ):
                    writer.writerow([r.dimension, metric_name, value if value is not None else "", s.n])
                if s.is_weighted:
                    writer.writerow([r.dimension, "weighted_percent_agreement", s.weighted_percent_agreement if s.weighted_percent_agreement is not None else "", s.n])
                    writer.writerow([r.dimension, "weighted_cohen_kappa", s.weighted_cohen_kappa if s.weighted_cohen_kappa is not None else "", s.n])
                    writer.writerow([r.dimension, "weighted_false_negative_rate", s.weighted_false_negative_rate if s.weighted_false_negative_rate is not None else "", s.n])
                    writer.writerow([r.dimension, "weighted_precision_include", s.weighted_precision if s.weighted_precision is not None else "", s.n])
                    writer.writerow([r.dimension, "kish_effective_n", s.effective_n if s.effective_n is not None else "", s.n])
                    for metric_name, value in (
                        ("weighted_cohen_kappa_max", s.weighted_kappa_max),
                        ("weighted_pabak", s.weighted_pabak),
                        ("weighted_gwet_ac1", s.weighted_gwet_ac1),
                        ("weighted_auc_ordinal_score", s.weighted_auc),
                    ):
                        writer.writerow([r.dimension, metric_name, value if value is not None else "", s.n])
                writer.writerow(
                    [r.dimension, "false_negative_rate", s.false_negative_rate if s.false_negative_rate is not None else "", s.n]
                )
            elif r.kind == "multi_label" and r.multi:
                m = r.multi
                writer.writerow([r.dimension, "mean_jaccard", m.mean_jaccard, m.n])
                writer.writerow([r.dimension, "micro_f1", m.micro_f1, m.n])
                writer.writerow([r.dimension, "macro_f1", m.macro_f1, m.n])
                for label, metrics in m.per_label.items():
                    for metric_name, value in metrics.items():
                        writer.writerow([r.dimension, f"{label}.{metric_name}", value, m.n])


def _coverage_lines(s: SingleLabelAgreementResult) -> list[str]:
    """Name any stratum that was drawn but not yet adjudicated.

    A weighted estimate silently omits such a stratum -- the records are in the
    sample, contribute no rows here, and their share of the corpus is
    redistributed onto the strata that ARE complete. That is precisely how the
    withdrawn 51.9% false-negative rate arose (D-35), so it is stated up front
    rather than left for a reader to notice."""
    incomplete = {
        stratum: (done, drawn) for stratum, (done, drawn) in s.stratum_coverage.items() if done < drawn
    }
    if not incomplete:
        return []
    detail = ", ".join(f"`{stratum}` {done}/{drawn}" for stratum, (done, drawn) in sorted(incomplete.items()))
    return [
        (
            "> ⚠ **Incomplete strata.** " + detail + " adjudicated. Every weighted figure below "
            "is conditional on the strata that are complete; the corpus share of an unfinished "
            "stratum is not estimated, it is simply absent. Finish these before quoting a "
            "corpus-level recall or false-negative rate."
        ),
        "",
    ]


def _kappa_context_lines(s: SingleLabelAgreementResult) -> list[str]:
    """The panel that stops a bare kappa from being read against 1.0 (D-36)."""

    def fmt(value: float | None, as_percent: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1%}" if as_percent else f"{value:.3f}"

    lines = [
        (
            "**Kappa in context.** A bare kappa is not interpretable when the two coders' "
            "marginals differ, and here they differ by construction -- the AI's marginal *is* "
            "the sampling design. Read kappa against kappa_max, not against 1.0."
        ),
        "",
        "| statistic | unweighted | design-weighted |",
        "|---|---|---|",
        f"| Cohen's kappa | {fmt(s.cohen_kappa)} | {fmt(s.weighted_cohen_kappa)} |",
        f"| kappa_max (ceiling at these marginals) | {fmt(s.kappa_max)} | {fmt(s.weighted_kappa_max)} |",
    ]
    for label, observed, ceiling in (
        ("unweighted", s.cohen_kappa, s.kappa_max),
        ("weighted", s.weighted_cohen_kappa, s.weighted_kappa_max),
    ):
        if observed is not None and ceiling:
            lines.append(f"| kappa as % of its ceiling ({label}) | {observed / ceiling:.0%} | - |")
    lines += [
        f"| PABAK (prevalence- and bias-adjusted) | {fmt(s.pabak)} | {fmt(s.weighted_pabak)} |",
        f"| Gwet's AC1 (prevalence-robust) | {fmt(s.gwet_ac1)} | {fmt(s.weighted_gwet_ac1)} |",
        f"| AUC of the AI's ordinal (label, confidence) score | {fmt(s.auc)} | {fmt(s.weighted_auc)} |",
        "",
        (
            "- **kappa_max** is the largest kappa reachable without either coder changing its "
            "marginal totals. The distance from kappa to kappa_max is the only part of the gap "
            "to 1.0 that better per-record judgement could close."
        ),
        (
            "- **PABAK / AC1** strip out the prevalence effect that depresses kappa under skew "
            "(Feinstein & Cicchetti 1990; Byrt et al. 1993; Gwet 2008). AC1 is prevalence-robust "
            "but not bias-robust, so a large AC1-minus-kappa gap says the disagreement sits in "
            "the marginals rather than in the individual calls."
        ),
        (
            "- **AUC** scores the *ordering* rather than the cut-point: a high AUC beside a low "
            "kappa means the screener ranks well and thresholds badly, which is a far cheaper "
            "defect to fix."
        ),
        (
            "- Landis & Koch's verbal bands are deliberately omitted; they were never validated "
            "and are misleading against a ceiling well below 1."
        ),
    ]
    return lines


def write_agreement_report(results: list[DimensionAgreement], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AI-Human Agreement Report",
        "",
        (
            "**Note:** this measures agreement between an AI screening/classification pass and a "
            "single human reviewer's decisions on the same records -- not inter-rater reliability "
            "among multiple independent human coders. Read percentages accordingly."
        ),
        "",
    ]
    for r in results:
        lines.append(f"## {r.dimension}")
        lines.append("")
        if r.kind == "single_label" and r.single:
            s = r.single
            if s.n == 0:
                lines.append("No paired AI/human decisions yet for this dimension.")
                lines.append("")
                continue
            lines.append(
                f"**False negative rate (human said include, AI did not): "
                f"{s.false_negative_rate:.1%}**" if s.false_negative_rate is not None else "**False negative rate: n/a (no human-include cases)**"
            )
            if s.false_negatives:
                lines.append(f"Affected record_ids: {', '.join(s.false_negatives)}")
            lines.append("")
            if s.is_weighted:
                lines.append(
                    "> **Read the weighted column.** This sample is stratified on the AI label with "
                    f"unequal probability (D-29), so the unweighted figures describe the {s.n} "
                    "adjudicated records only. The weighted figures are the estimates for the corpus."
                )
                lines.append("")
                lines.extend(_coverage_lines(s))
                lines.append("| statistic | unweighted (this sample) | design-weighted (corpus estimate) |")
                lines.append("|---|---|---|")
                lines.append(
                    f"| Percent agreement | {s.percent_agreement:.1%} | "
                    + (f"{s.weighted_percent_agreement:.1%} |" if s.weighted_percent_agreement is not None else "n/a |")
                )
                lines.append(
                    "| Cohen's kappa | "
                    + (f"{s.cohen_kappa:.3f}" if s.cohen_kappa is not None else "n/a")
                    + " | "
                    + (f"{s.weighted_cohen_kappa:.3f} |" if s.weighted_cohen_kappa is not None else "n/a |")
                )
                lines.append(
                    "| False negative rate | "
                    + (f"{s.false_negative_rate:.1%}" if s.false_negative_rate is not None else "n/a")
                    + " | "
                    + (f"{s.weighted_false_negative_rate:.1%} |" if s.weighted_false_negative_rate is not None else "n/a |")
                )
                lines.append(
                    "| Precision of AI include | - | "
                    + (f"{s.weighted_precision:.1%} |" if s.weighted_precision is not None else "n/a |")
                )
                lines.append("")
                lines.extend(_kappa_context_lines(s))
                lines.append("")
                lines.append(f"- n adjudicated = {s.n}")
                if s.effective_n is not None:
                    lines.append(
                        f"- Kish effective n = {s.effective_n:.0f} -- **use this, not n, when quoting a "
                        "confidence interval on any weighted figure above.**"
                    )
                lines.append(
                    "- Weights come from `outputs/sampling_manifest.json`; the confusion matrix below "
                    "is raw counts and is NOT weighted."
                )
            else:
                lines.append(f"- n = {s.n}")
                lines.append(f"- Percent agreement (AI vs human): {s.percent_agreement:.1%}")
                lines.append(f"- Cohen's kappa: {s.cohen_kappa:.3f}" if s.cohen_kappa is not None else "- Cohen's kappa: n/a (only one class present)")
                lines.append("")
                lines.extend(_kappa_context_lines(s))
            lines.append("- Confusion matrix (rows=human, cols=AI):")
            all_labels = sorted({lbl for row in s.confusion_matrix.values() for lbl in row} | set(s.confusion_matrix))
            header = "  | " + " | ".join(["human\\ai", *all_labels]) + " |"
            lines.append(header)
            lines.append("  |" + "---|" * (len(all_labels) + 1))
            for human_label in all_labels:
                row = s.confusion_matrix.get(human_label, {})
                lines.append("  | " + " | ".join([human_label, *(str(row.get(al, 0)) for al in all_labels)]) + " |")
            lines.append("")
        elif r.kind == "multi_label" and r.multi:
            m = r.multi
            if m.n == 0:
                lines.append("No paired AI/human decisions yet for this dimension.")
                lines.append("")
                continue
            lines.append(f"- n = {m.n}")
            lines.append(f"- Mean Jaccard similarity: {m.mean_jaccard:.3f}")
            lines.append(f"- Micro F1: {m.micro_f1:.3f}")
            lines.append(f"- Macro F1: {m.macro_f1:.3f}")
            lines.append("")
            lines.append("| label | precision | recall | f1 |")
            lines.append("|---|---|---|---|")
            for label, metrics in sorted(m.per_label.items()):
                lines.append(f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
