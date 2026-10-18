"""Phase 11 (plan section 17): builds the stratified validation sample that
`review-cli classify human` actually walks, instead of the full D1 pool
(1800+ records topical-decision by topical-decision -- unworkable for a
single reviewer).

Stratification uses only signal that exists at this point in the pipeline:
Phase 10 AI `topical_inclusion` decisions (classifications_ai) and record
metadata (title/abstract keywords for CE/PSE and the brief's named technical
themes). It deliberately does NOT use domain/technology labels, because
those don't exist yet -- assigning them is exactly what this human pass
validates.

Reading classifications_ai here is fine even though the human-classify
*session* itself must never see AI labels (src/validation/blinded_queue.py's
structural guarantee, checked by tests/test_blinded_queue.py): sample
construction runs once, offline, before any reviewer opens a record, and
this module is never imported by the live review screen.

Deterministic: build_validation_sample(seed, target_size) always returns the
same rows for the same (seed, database state), so the sample is regenerable
byte-for-byte. write_sample_files() records the exact stratum definitions
and per-stratum counts used, per the brief's reproducibility requirement.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, select

from review_pipeline.src.common.config import OUTPUTS_DIR
from review_pipeline.src.db.tables import reflect_table

DEFAULT_SAMPLE_CSV_PATH = OUTPUTS_DIR / "validation_sample.csv"
DEFAULT_MANIFEST_PATH = OUTPUTS_DIR / "sampling_manifest.json"

DEFAULT_SEED = 42
DEFAULT_TARGET_SIZE = 90

CHEMICAL_PSE_KEYWORDS: tuple[str, ...] = (
    "chemical engineering", "process systems engineering", "process system",
    "chemical process", "distillation", "reactor", "process control",
    "plant operation", "process industry", "petrochemical",
)
THEMATIC_KEYWORDS: tuple[str, ...] = (
    "safety", "simulation", "multi-agent", "multi agent", "digital twin",
    "maintenance", "reliability",
)
BENCHMARK_KEYWORDS: tuple[str, ...] = (
    "benchmark", "benchmarking", "benchmarked",
    "evaluat",  # stem: evaluate/evaluating/evaluation/evaluated -- safe once
    # ANDed with genai_block+domain_block below, which already does the
    # heavy lifting of cutting the off-topic snowballed-in ML papers.
    "synthetic dataset", "synthetic data", "reference dataset",
    "standardized dataset", "standardized benchmark", "public dataset",
)

# Stratum name -> fraction of target_size to fill from that stratum's pool,
# in this priority order (a record already claimed by an earlier stratum is
# never reassigned). Whatever's left after these is filled by random_common.
STRATUM_QUOTA_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("low_confidence", 0.25),
    ("excluded_borderline", 0.15),
    ("chemical_pse", 0.20),
    ("thematic", 0.20),
)
STRATA_ORDER: tuple[str, ...] = tuple(name for name, _ in STRATUM_QUOTA_FRACTIONS) + ("random_common",)

# --- Allocation modes -------------------------------------------------------
#
# "thematic" (the original, and still the default) fills quotas by
# disagreement-likelihood and topic: low-confidence records, borderline
# excludes, and chemical/PSE or thematic keyword matches. It is good at
# *finding* screening errors and poor at *estimating* how often they happen,
# because the AI's `include` label is not a stratum, so whatever number of
# includes lands in the sample is incidental. At target_size=400 it yielded 71.
#
# "by_ai_label" stratifies on the Phase 10 AI decision itself and deliberately
# over-samples `include` and `uncertain`. That is what makes the precision of
# the include label -- the quantity the synthesis actually rests on -- estimable
# to a stated half-width, and it keeps a large enough exclude stratum to bound
# the false-negative rate. Strata are unequal-probability by design, so every
# population-level statistic computed from this sample MUST be reweighted by
# N_h/n_h; the manifest records both counts for exactly that reason.
#
# Within the exclude stratum the thematic mode's instinct is preserved: the
# lower-confidence excludes are drawn first, so the exclude cell still points
# at the records most likely to be wrong.
AI_LABEL_QUOTA_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("ai_include", 0.50),
    ("ai_exclude", 0.325),
    ("ai_uncertain", 0.175),
)
AI_LABEL_STRATA_ORDER: tuple[str, ...] = tuple(name for name, _ in AI_LABEL_QUOTA_FRACTIONS)

# --- The exclude stratum is two frames, not one (D-35) ----------------------
#
# `build_label_stratified_sample` sorts the exclude candidates by confidence
# before truncating to the quota. That sort is deterministic, so at
# target_size=400 the 130 exclude slots went to the 130 lowest-confidence
# excludes and the remaining 6,415 -- including all 5,773 high-confidence ones
# -- had selection probability EXACTLY ZERO. A weight of N/n = 6545/130 = 50.3
# says the opposite: that each sampled record stands for ~50 drawn at random.
# Extrapolating anything measured in that cell across the whole exclude
# population is therefore unsupported, and the 51.9% false-negative rate that
# came out of it has been withdrawn.
#
# The fix is to name the two frames separately rather than to pretend one draw
# covered both. `ai_exclude_borderline` is the sub-population the original draw
# actually reached (confidence != high, N = 774 at Phase 10); `ai_exclude_high`
# is the untouched remainder, sampled by `extend_sample_with_random_excludes`
# with equal probability so that it carries a defensible weight of its own.
# Splitting them changes no record's membership -- the 400 completed
# adjudications stand -- only the population each cell is allowed to speak for.
AI_EXCLUDE_BORDERLINE = "ai_exclude_borderline"
AI_EXCLUDE_HIGH = "ai_exclude_high"
EXTENDED_STRATA_ORDER: tuple[str, ...] = (
    "ai_include",
    AI_EXCLUDE_BORDERLINE,
    AI_EXCLUDE_HIGH,
    "ai_uncertain",
)


@dataclass
class SampleRow:
    record_id: str
    stratum: str


@dataclass
class SamplingResult:
    seed: int
    target_size: int
    pool_size: int
    rows: list[SampleRow] = field(default_factory=list)
    stratum_counts: dict[str, int] = field(default_factory=dict)
    # Population size of each stratum, so N_h/n_h weights are recoverable
    # from the manifest alone. Empty for the thematic allocation, whose
    # strata overlap and therefore have no clean N_h.
    population_counts: dict[str, int] = field(default_factory=dict)


def _d1_pool(engine: Engine) -> list[dict]:
    records_table = reflect_table(engine, "records")
    with engine.connect() as conn:
        rows = conn.execute(
            records_table.select().where(
                records_table.c.d1 == 1, records_table.c.canonical_record_id.is_(None)
            )
        ).mappings().all()
    return [dict(r) for r in rows]


def _ai_topical_decisions(engine: Engine) -> dict[str, tuple[str | None, str | None]]:
    """record_id -> (decision, confidence) from the AI topical_inclusion pass.

    Most recent decision wins. A handful of records were screened twice (a batch
    overlap, and the D-21 re-screen under the amended criteria), and two of them
    changed label between the passes -- so which row is read decides which
    stratum the record belongs to. Ordering by timestamp makes that the current
    decision rather than whichever row the database happened to return last.
    """
    classifications_ai = reflect_table(engine, "classifications_ai")
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                classifications_ai.c.record_id,
                classifications_ai.c.label,
                classifications_ai.c.confidence,
                classifications_ai.c.timestamp,
            )
            .where(classifications_ai.c.dimension == "topical_inclusion")
            .order_by(classifications_ai.c.timestamp)
        ).all()
    return {r[0]: (r[1], r[2]) for r in rows}


def _matches_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def _ai_dimension_labels(engine: Engine, dimension: str) -> dict[str, str]:
    """record_id -> label for one classifications_ai dimension. Multi-value
    dimensions keep their raw JSON-array-as-string label -- callers that only
    need a substring match (e.g. does evidence_type mention 'benchmark') can
    use it as-is without parsing."""
    classifications_ai = reflect_table(engine, "classifications_ai")
    with engine.connect() as conn:
        rows = conn.execute(
            select(classifications_ai.c.record_id, classifications_ai.c.label).where(
                classifications_ai.c.dimension == dimension
            )
        ).all()
    return {r[0]: r[1] for r in rows}


def build_validation_sample(
    engine: Engine, seed: int = DEFAULT_SEED, target_size: int = DEFAULT_TARGET_SIZE
) -> SamplingResult:
    pool = _d1_pool(engine)
    ai_decisions = _ai_topical_decisions(engine)
    rng = random.Random(seed)

    assigned: dict[str, str] = {}

    def decision_confidence(record: dict) -> tuple[str | None, str | None]:
        return ai_decisions.get(record["record_id"], (None, None))

    def take(predicate, stratum: str, quota: int) -> None:
        candidates = [r["record_id"] for r in pool if r["record_id"] not in assigned and predicate(r)]
        rng.shuffle(candidates)
        for record_id in candidates[:quota]:
            assigned[record_id] = stratum

    for stratum, fraction in STRATUM_QUOTA_FRACTIONS:
        quota = max(1, round(target_size * fraction)) if pool else 0
        if stratum == "low_confidence":
            take(lambda r: decision_confidence(r)[1] in ("low", "medium"), stratum, quota)
        elif stratum == "excluded_borderline":
            take(
                lambda r: decision_confidence(r)[0] == "exclude" and decision_confidence(r)[1] != "high",
                stratum,
                quota,
            )
        elif stratum == "chemical_pse":
            take(
                lambda r: _matches_any(f"{r.get('title') or ''} {r.get('abstract') or ''}", CHEMICAL_PSE_KEYWORDS),
                stratum,
                quota,
            )
        elif stratum == "thematic":
            take(
                lambda r: _matches_any(f"{r.get('title') or ''} {r.get('abstract') or ''}", THEMATIC_KEYWORDS),
                stratum,
                quota,
            )

    remaining_quota = max(0, target_size - len(assigned))
    if remaining_quota:
        take(lambda r: True, "random_common", remaining_quota)

    stratum_counts: dict[str, int] = {}
    for record_id, stratum in assigned.items():
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1

    rows = sorted(
        (SampleRow(record_id, stratum) for record_id, stratum in assigned.items()),
        key=lambda row: (STRATA_ORDER.index(row.stratum), row.record_id),
    )

    return SamplingResult(seed=seed, target_size=target_size, pool_size=len(pool), rows=rows, stratum_counts=stratum_counts)


def build_label_stratified_sample(
    engine: Engine, seed: int = DEFAULT_SEED, target_size: int = DEFAULT_TARGET_SIZE
) -> SamplingResult:
    """Stratify on the Phase 10 AI decision, over-sampling `include` and
    `uncertain` so that the precision of the include label is estimable.

    Deterministic for a given (seed, database state), like its sibling. Quotas
    are capped by the size of each stratum's pool; any shortfall spills to the
    exclude stratum, which is always the largest pool.
    """
    pool = _d1_pool(engine)
    ai_decisions = _ai_topical_decisions(engine)
    rng = random.Random(seed)

    by_label: dict[str, list[dict]] = {"include": [], "exclude": [], "uncertain": []}
    for record in pool:
        decision, _confidence = ai_decisions.get(record["record_id"], (None, None))
        if decision in by_label:
            by_label[decision].append(record)

    def confidence_rank(record: dict) -> int:
        """Lower-confidence records first, so the exclude cell keeps pointing at
        the records most likely to be wrong."""
        order = {"low": 0, "medium": 1, "high": 2}
        return order.get(ai_decisions.get(record["record_id"], (None, None))[1], 2)

    assigned: dict[str, str] = {}
    quotas = {
        stratum: min(round(target_size * fraction), len(by_label[stratum.removeprefix("ai_")]))
        for stratum, fraction in AI_LABEL_QUOTA_FRACTIONS
    }
    shortfall = target_size - sum(quotas.values())
    if shortfall > 0:
        headroom = len(by_label["exclude"]) - quotas["ai_exclude"]
        quotas["ai_exclude"] += min(shortfall, headroom)

    for stratum in AI_LABEL_STRATA_ORDER:
        candidates = list(by_label[stratum.removeprefix("ai_")])
        rng.shuffle(candidates)
        if stratum == "ai_exclude":
            candidates.sort(key=confidence_rank)  # stable: ties keep the shuffle
        for record in candidates[: quotas[stratum]]:
            assigned[record["record_id"]] = stratum

    stratum_counts: dict[str, int] = {}
    for record_id, stratum in assigned.items():
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1

    rows = sorted(
        (SampleRow(record_id, stratum) for record_id, stratum in assigned.items()),
        key=lambda row: (AI_LABEL_STRATA_ORDER.index(row.stratum), row.record_id),
    )
    result = SamplingResult(
        seed=seed, target_size=target_size, pool_size=len(pool), rows=rows, stratum_counts=stratum_counts
    )
    result.population_counts = {
        f"ai_{label}": len(records) for label, records in by_label.items()
    }
    return result


def _exclude_frames(pool: list[dict], ai_decisions: dict[str, tuple[str | None, str | None]]) -> tuple[list[dict], list[dict]]:
    """(borderline, high) -- the AI-`exclude` pool split on whether the AI was
    confident. See the AI_EXCLUDE_* note above for why the split exists."""
    borderline, high = [], []
    for record in pool:
        decision, confidence = ai_decisions.get(record["record_id"], (None, None))
        if decision != "exclude":
            continue
        (high if confidence == "high" else borderline).append(record)
    return borderline, high


def extend_sample_with_random_excludes(
    engine: Engine,
    existing: dict[str, str],
    size: int,
    seed: int = DEFAULT_SEED + 1,
) -> SamplingResult:
    """Close the unvalidated high-confidence exclude stratum with a genuine
    equal-probability draw, and re-label the existing exclude rows with the
    frame they were actually drawn from (D-35).

    `existing` is a record_id -> stratum map, i.e. exactly what
    `load_validation_sample()` returns. Its rows are carried through verbatim
    apart from `ai_exclude`, which is resolved to `ai_exclude_borderline` or
    `ai_exclude_high` by each record's own AI confidence -- so no already-
    adjudicated record leaves the sample or changes which decisions it belongs
    to. `size` new records are then drawn uniformly at random, without
    replacement, from the high-confidence excludes not already present.

    The returned `population_counts` are the four real frames, so the manifest's
    N_h/n_h weights finally describe draws that were actually taken at random.
    """
    pool = _d1_pool(engine)
    ai_decisions = _ai_topical_decisions(engine)
    _borderline, high = _exclude_frames(pool, ai_decisions)
    rng = random.Random(seed)

    assigned: dict[str, str] = {}
    for record_id, stratum in existing.items():
        if stratum == "ai_exclude":
            _decision, confidence = ai_decisions.get(record_id, (None, None))
            stratum = AI_EXCLUDE_HIGH if confidence == "high" else AI_EXCLUDE_BORDERLINE
        assigned[record_id] = stratum

    candidates = sorted(r["record_id"] for r in high if r["record_id"] not in assigned)
    rng.shuffle(candidates)
    for record_id in candidates[:size]:
        assigned[record_id] = AI_EXCLUDE_HIGH

    stratum_counts: dict[str, int] = {}
    for stratum in assigned.values():
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1

    by_label: dict[str, int] = {"include": 0, "uncertain": 0}
    for record in pool:
        decision, _confidence = ai_decisions.get(record["record_id"], (None, None))
        if decision in by_label:
            by_label[decision] += 1

    order = {name: i for i, name in enumerate(EXTENDED_STRATA_ORDER)}
    rows = sorted(
        (SampleRow(record_id, stratum) for record_id, stratum in assigned.items()),
        key=lambda row: (order.get(row.stratum, len(order)), row.stratum, row.record_id),
    )
    result = SamplingResult(
        seed=seed, target_size=len(rows), pool_size=len(pool), rows=rows, stratum_counts=stratum_counts
    )
    result.population_counts = {
        "ai_include": by_label["include"],
        "ai_uncertain": by_label["uncertain"],
        AI_EXCLUDE_BORDERLINE: len(_borderline),
        AI_EXCLUDE_HIGH: len(high),
    }
    return result


def write_sample_files(
    result: SamplingResult,
    csv_path: Path = DEFAULT_SAMPLE_CSV_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["record_id", "stratum"])
        for row in result.rows:
            writer.writerow([row.record_id, row.stratum])

    manifest = {
        "seed": result.seed,
        "target_size": result.target_size,
        "pool_size": result.pool_size,
        "sample_size": len(result.rows),
        "stratum_counts": result.stratum_counts,
        "population_counts": result.population_counts,
        "sampling_weights": {
            stratum: result.population_counts[stratum] / count
            for stratum, count in result.stratum_counts.items()
            if result.population_counts.get(stratum)
        },
        "stratum_definitions": {
            "ai_include": "Phase 10 AI decision = include (over-sampled)",
            "ai_exclude": "Phase 10 AI decision = exclude, lower-confidence first",
            AI_EXCLUDE_BORDERLINE: (
                "Phase 10 AI decision = exclude with confidence != high -- the frame the "
                "original lower-confidence-first draw actually reached (D-35)"
            ),
            AI_EXCLUDE_HIGH: (
                "Phase 10 AI decision = exclude with confidence = high -- drawn with equal "
                "probability by `classify extend-sample` (D-35)"
            ),
            "ai_uncertain": "Phase 10 AI decision = uncertain (over-sampled)",
            "low_confidence": "AI topical_inclusion confidence in {low, medium}",
            "excluded_borderline": "AI decision=exclude with confidence != high",
            "chemical_pse": f"title/abstract matches any of {CHEMICAL_PSE_KEYWORDS}",
            "thematic": f"title/abstract matches any of {THEMATIC_KEYWORDS}",
            "random_common": "random fill from the remaining D1 pool",
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_validation_sample(csv_path: Path = DEFAULT_SAMPLE_CSV_PATH) -> dict[str, str] | None:
    """record_id -> stratum for the currently-built sample, or None if
    build_validation_sample()/write_sample_files() haven't been run yet."""
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["record_id"]: row["stratum"] for row in reader}


def mandatory_review_ids(engine: Engine) -> dict[str, str]:
    """D1 records whose AI topical_inclusion decision cannot stand alone as
    the corpus's final decision: the decision is 'uncertain', its confidence
    is not 'high', or no AI decision exists at all. These receive a full
    independent human decision regardless of validation-sample membership --
    the manuscript's methodology (draft_paper/cas-sc-template-v2.tex section
    2.6) accepts AI's high-confidence decisions elsewhere without individual
    re-review, a tradeoff whose soundness is checked by the validation
    sample's AI-human agreement metrics rather than by re-reviewing every
    high-confidence record too. Returns record_id -> reason, so callers can
    label sample membership informatively for records that aren't also in
    the stratified validation sample."""
    pool_ids = {r["record_id"] for r in _d1_pool(engine)}
    ai_decisions = _ai_topical_decisions(engine)
    mandatory: dict[str, str] = {}
    for record_id in pool_ids:
        decision, confidence = ai_decisions.get(record_id, (None, None))
        if decision is None:
            mandatory[record_id] = "no_ai_decision"
        elif decision == "uncertain":
            mandatory[record_id] = "ai_uncertain"
        elif confidence != "high":
            mandatory[record_id] = "ai_non_high_confidence"
    return mandatory


def benchmark_related_ids(engine: Engine) -> dict[str, str]:
    """D1 records plausibly relevant to the benchmark-dataset gap-analysis
    thread (config/gap_claims.yaml's gap_pse_benchmarks): AI already tagged
    evidence_type as benchmark-like, or the title/abstract uses benchmark/
    dataset language AND the genai+domain terms that gate normal D0
    discovery (query_builder.matches_genai_domain) -- most of the D1 pool
    came in via Phase 19 citation snowballing with no topical filter at all
    (see citation_trace.py), so "benchmark" alone matches thousands of
    off-topic ML papers (MedicalAgentsBench, RefactorBench, clinical-record
    benchmarks, ...). Requiring genai+domain too keeps this to records a
    domain reader would actually recognize as GenAI-copilot-adjacent.
    evidence_type is only assigned once a record's topical_inclusion is
    include, so a still-pending or AI-excluded record needs the keyword
    fallback to be caught here at all. Returns record_id -> reason, the
    same shape as mandatory_review_ids, so it composes identically in the
    CLI's --sample handling."""
    from review_pipeline.src.search.query_builder import load_search_queries_config

    config = load_search_queries_config()
    pool = _d1_pool(engine)
    evidence_type_labels = _ai_dimension_labels(engine, "evidence_type")
    result: dict[str, str] = {}
    for record in pool:
        record_id = record["record_id"]
        label = evidence_type_labels.get(record_id, "")
        if "benchmark" in label.lower():
            result[record_id] = "ai_evidence_type_benchmark"
            continue
        haystack = f"{record.get('title') or ''} {record.get('abstract') or ''}"
        if (
            _matches_any(haystack, BENCHMARK_KEYWORDS)
            and _matches_any(haystack, tuple(config.genai_block))
            and _matches_any(haystack, tuple(config.domain_block))
        ):
            result[record_id] = "benchmark_keyword_match"
    return result
