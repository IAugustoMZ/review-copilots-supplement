"""Phase 12b: an *unbiased* validation of the v2 screening criterion (D-47).

WHY THIS EXISTS
---------------
Phase 10b reported v2's precision against the human reviewer as 56.9%, up from
v1's 19.0%. That number is a resubstitution estimate and is optimistically
biased: v2 was written *after* inspecting 45 of v1's false positives and all 10
of its false negatives, drawn from the very 500 records it is then scored
against. Those records are training data in the statistical sense. Phase 12b
replaces that number with one measured on records v2's author never saw
adjudicated.

THE BUDGET IS ONE SAMPLE, AND IT IS SMALL ON PURPOSE
----------------------------------------------------
There is a single reviewer and no appetite for a third pass, so the design
question is not "what is the best sample" but "what is the smallest sample that
still answers the one question the manuscript cannot go to press without".

That question is the precision of a v2 `include`, because that is the quantity
the corpus rests on: 348 records carry a v2 include, and how many of them are
real decides what synthesis is built from. Everything else is secondary.

Two further facts shrink the bill:

  - Only the *unadjudicated* v2 includes are usable. 58 of the 348 already
    carry a Phase 11 human decision and are excluded from the estimand for the
    contamination reason above, leaving N = 290. A small population makes the
    finite-population correction do real work: at n = 100 and p ~= 0.57 the
    half-width is +/-7.9pp, where an infinite-population calculation would
    demand ~150 for the same precision.
  - None of that work is throwaway. All 290 must be human-adjudicated before
    they can enter synthesis regardless. Reviewing 100 of them is 34% of a task
    already on the critical path, not new overhead.

STRATA
------
    stratum          pool                                       N     n   weight
    v2_include       v2 include, no human decision yet        290   100     2.90
    v2_loss_probe    v1 include & v2 exclude, no human yet    598    30    19.93

`v2_include` estimates precision. `v2_loss_probe` is deliberately NOT a recall
estimate -- 30 draws at an expected ~5% include density cannot estimate a rate,
and pretending otherwise is how the withdrawn 51.9% false-negative figure
happened (D-35). What 30 draws buy is a one-sided BOUND on how much real corpus
the narrowing threw away, over the only sub-population where v2's false
negatives can live: records the wider criterion admitted and the narrower one
rejected. v2 is strictly narrower than v1, so a record both passes rejected
cannot be a v2-specific loss. Report this cell as an upper bound with its exact
binomial interval, never as a point estimate of recall.

Total: 130 records, against Phase 11's 500.

BLINDING AND INDEPENDENCE
-------------------------
The reviewer walks these through the same `classify assisted` loop as Phase 11,
which cannot read AI labels at all (src/validation/blinded_queue.py's checked
invariant). Seed 46 is distinct from Phase 11's 42/43 and Phase 10b's 45, so no
draw is a re-draw of another.

WHAT THIS STILL CANNOT DO
-------------------------
Records outside both pools -- v1 excludes that v2 also excludes -- are not
sampled here. Phase 11b's equal-probability draw of 100 high-confidence excludes
(1 include) remains the only evidence about that population, and Phase 12b does
not improve it. Stated plainly rather than papered over: the corpus's residual
is bounded by a single observed record, and no further sampling is planned.
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
from review_pipeline.src.screening.rescreen import (
    V2_PROMPT_VERSION,
    latest_v1_decisions,
)

DEFAULT_V2_SAMPLE_CSV_PATH = OUTPUTS_DIR / "validation_v2_sample.csv"
DEFAULT_V2_MANIFEST_PATH = OUTPUTS_DIR / "validation_v2_manifest.json"

DEFAULT_SEED = 46  # distinct from Phase 11's 42/43 and Phase 10b's 45

V2_INCLUDE = "v2_include"
V2_LOSS_PROBE = "v2_loss_probe"
V2_STRATA_ORDER: tuple[str, ...] = (V2_INCLUDE, V2_LOSS_PROBE)

DEFAULT_INCLUDE_N = 100
DEFAULT_LOSS_PROBE_N = 30


def proportion_half_width(n: int, population_n: int, p: float = 0.5, z: float = 1.96) -> float:
    """95% half-width for a proportion under simple random sampling WITHOUT
    replacement.

    The finite-population correction is not a rounding detail here, it is the
    reason the sample is affordable: at n=100 out of N=290 it multiplies the
    half-width by sqrt((290-100)/289) = 0.81, so 100 draws buy what ~150 would
    buy from an unbounded population. `p=0.5` is the conservative default --
    every other p gives a narrower interval.

    src/validation/sample_size.py's `achieved_half_width` is not this function:
    it sizes for Cohen's kappa, whose variance depends on prevalence and on the
    kappa being estimated. This one sizes for a plain proportion.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if population_n <= 1:
        return 0.0
    fpc = (population_n - n) / (population_n - 1)
    return z * ((p * (1 - p) / n) ** 0.5) * (max(fpc, 0.0) ** 0.5)


def latest_v2_decisions(engine: Engine) -> dict[str, str]:
    """record_id -> label from the Phase 10b v2 pass only."""
    classifications_ai = reflect_table(engine, "classifications_ai")
    query = (
        select(classifications_ai.c.record_id, classifications_ai.c.label)
        .where(
            classifications_ai.c.dimension == "topical_inclusion",
            classifications_ai.c.prompt_version == V2_PROMPT_VERSION,
        )
        .order_by(classifications_ai.c.timestamp)
    )
    with engine.connect() as conn:
        return {record_id: label for record_id, label in conn.execute(query)}


def already_adjudicated(engine: Engine) -> set[str]:
    """Records carrying any human topical decision. Excluded from both pools:
    for `v2_include` because a Phase 11 decision is contaminated evidence about
    a criterion written with Phase 11 in hand, and for `v2_loss_probe` for the
    same reason plus the plain one that re-reviewing them buys nothing."""
    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        return {
            row[0]
            for row in conn.execute(
                select(classifications_human.c.record_id).where(
                    classifications_human.c.dimension == "topical_inclusion"
                )
            )
        }


@dataclass
class V2SampleResult:
    seed: int
    members: dict[str, str] = field(default_factory=dict)  # record_id -> stratum
    population_counts: dict[str, int] = field(default_factory=dict)  # N_h
    sample_counts: dict[str, int] = field(default_factory=dict)  # n_h

    @property
    def weights(self) -> dict[str, float]:
        return {
            stratum: self.population_counts[stratum] / count
            for stratum, count in self.sample_counts.items()
            if count
        }


def build_v2_validation_sample(
    engine: Engine,
    include_n: int = DEFAULT_INCLUDE_N,
    loss_probe_n: int = DEFAULT_LOSS_PROBE_N,
    seed: int = DEFAULT_SEED,
) -> V2SampleResult:
    """Equal-probability draws from the two pools. Deterministic given `seed`.

    Pools are sorted before `rng.sample` for the same reason Phase 10b sorts:
    unsorted, the draw would depend on screening chronology rather than on the
    seed alone, and would not be reproducible from the seed after any re-ingest.
    """
    v1 = latest_v1_decisions(engine)
    v2 = latest_v2_decisions(engine)
    adjudicated = already_adjudicated(engine)

    pools: dict[str, list[str]] = {
        V2_INCLUDE: sorted(
            rid for rid, label in v2.items() if label == "include" and rid not in adjudicated
        ),
        # v2 is strictly narrower than v1, so a v2-specific loss can only be a
        # record v1 admitted and v2 rejected. Records both passes excluded are
        # out of scope for this probe by construction, not by omission.
        V2_LOSS_PROBE: sorted(
            rid
            for rid, label in v2.items()
            if label == "exclude"
            and v1.get(rid, ("", None))[0] == "include"
            and rid not in adjudicated
        ),
    }
    quotas = {V2_INCLUDE: include_n, V2_LOSS_PROBE: loss_probe_n}

    result = V2SampleResult(seed=seed)
    for stratum in V2_STRATA_ORDER:
        pool = pools[stratum]
        result.population_counts[stratum] = len(pool)
        rng = random.Random(f"{seed}:{stratum}")
        taken = rng.sample(pool, min(quotas[stratum], len(pool)))
        result.sample_counts[stratum] = len(taken)
        for record_id in sorted(taken):
            result.members[record_id] = stratum
    return result


def write_v2_sample_files(
    result: V2SampleResult,
    csv_path: Path = DEFAULT_V2_SAMPLE_CSV_PATH,
    manifest_path: Path = DEFAULT_V2_MANIFEST_PATH,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    order = {name: i for i, name in enumerate(V2_STRATA_ORDER)}
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["record_id", "stratum"])
        for record_id, stratum in sorted(
            result.members.items(), key=lambda kv: (order[kv[1]], kv[0])
        ):
            writer.writerow([record_id, stratum])

    manifest = {
        "phase": "12b",
        "measures": V2_PROMPT_VERSION,
        "seed": result.seed,
        "sample_size": len(result.members),
        "population_counts": result.population_counts,
        "sample_counts": result.sample_counts,
        "sampling_weights": result.weights,
        "stratum_definitions": {
            V2_INCLUDE: (
                "v2 topical_inclusion = include, no human decision yet. Estimates the "
                "precision of a v2 include, unbiased: every record in the pool was "
                "unseen when v2 was written."
            ),
            V2_LOSS_PROBE: (
                "v1 include AND v2 exclude, no human decision yet. A one-sided BOUND on "
                "corpus lost to the narrowing -- not a recall estimate; n is far too "
                "small for a rate (see D-35)."
            ),
        },
        "excluded_from_pools": (
            "Records already carrying a Phase 11 human decision. v2 was authored after "
            "inspecting Phase 11 outcomes, so those records cannot test it."
        ),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_v2_validation_sample(
    csv_path: Path = DEFAULT_V2_SAMPLE_CSV_PATH,
) -> dict[str, str] | None:
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as fh:
        return {row["record_id"]: row["stratum"] for row in csv.DictReader(fh)}


# --------------------------------------------------------------------------
# Analysis (D-50). Every Phase 12b number the manuscript quotes is produced
# here and written to outputs/, so no figure reaches the .tex by hand.
# --------------------------------------------------------------------------

DEFAULT_V2_RESULTS_PATH = OUTPUTS_DIR / "validation" / "phase12b_results.md"
DEFAULT_V2_RESULTS_JSON = OUTPUTS_DIR / "validation" / "phase12b_results.json"


def clopper_pearson(x: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided interval for a binomial proportion.

    Exact rather than normal-approximate because the loss probe is 1/30: a Wald
    interval there runs below zero and a Wilson one still leans on asymptotics
    that 30 draws do not earn.
    """
    from scipy import stats

    alpha = 1.0 - conf
    lower = 0.0 if x == 0 else float(stats.beta.ppf(alpha / 2, x, n - x + 1))
    upper = 1.0 if x == n else float(stats.beta.ppf(1 - alpha / 2, x + 1, n - x))
    return lower, upper


def one_sided_upper(x: int, n: int, conf: float = 0.95) -> float:
    """Exact one-sided 95% upper bound. This is what the loss probe reports."""
    from scipy import stats

    return 1.0 if x == n else float(stats.beta.ppf(conf, x + 1, n - x))


@dataclass
class StratumResult:
    stratum: str
    n: int
    includes: int
    population_n: int

    @property
    def rate(self) -> float:
        return self.includes / self.n if self.n else 0.0

    @property
    def exact_ci(self) -> tuple[float, float]:
        return clopper_pearson(self.includes, self.n)

    @property
    def upper_bound(self) -> float:
        return one_sided_upper(self.includes, self.n)

    @property
    def half_width(self) -> float:
        """FPC-corrected, at the *observed* p rather than the p=0.5 used to size."""
        return proportion_half_width(self.n, self.population_n, p=self.rate)


@dataclass
class V2ValidationResult:
    strata: dict[str, StratumResult] = field(default_factory=dict)
    unadjudicated: dict[str, int] = field(default_factory=dict)
    resubstitution_n: int = 0
    resubstitution_includes: int = 0

    @property
    def resubstitution_rate(self) -> float:
        return self.resubstitution_includes / self.resubstitution_n if self.resubstitution_n else 0.0

    @property
    def bias(self) -> float:
        """Resubstitution precision minus honest precision, in proportion points."""
        return self.resubstitution_rate - self.strata[V2_INCLUDE].rate


def analyse_v2_validation(
    engine: Engine,
    csv_path: Path = DEFAULT_V2_SAMPLE_CSV_PATH,
) -> V2ValidationResult:
    """Score the drawn sample against the reviewer's decisions.

    The `resubstitution_*` fields recompute Phase 10b §7's precision over the v2
    includes that already carried a Phase 11 decision when the sample was drawn.
    Those are exactly the records v2 was written with in hand, so the pair
    (resubstitution, honest) measures the bias directly on the same criterion
    rather than inferring it.
    """
    membership = load_v2_validation_sample(csv_path)
    if membership is None:
        raise FileNotFoundError(f"No Phase 12b sample at {csv_path}; run `classify build-sample-v2`.")

    human = _latest_human_topical(engine)
    v2 = latest_v2_decisions(engine)

    result = V2ValidationResult()
    for stratum in V2_STRATA_ORDER:
        ids = [rid for rid, s in membership.items() if s == stratum]
        adjudicated = [rid for rid in ids if rid in human]
        result.unadjudicated[stratum] = len(ids) - len(adjudicated)
        result.strata[stratum] = StratumResult(
            stratum=stratum,
            n=len(adjudicated),
            includes=sum(1 for rid in adjudicated if human[rid] == "include"),
            population_n=_population_n(stratum, csv_path),
        )

    contaminated = [
        rid
        for rid, label in v2.items()
        if label == "include" and rid in human and rid not in membership
    ]
    result.resubstitution_n = len(contaminated)
    result.resubstitution_includes = sum(1 for rid in contaminated if human[rid] == "include")
    return result


def _population_n(stratum: str, csv_path: Path) -> int:
    manifest_path = csv_path.parent / DEFAULT_V2_MANIFEST_PATH.name
    counts = json.loads(manifest_path.read_text(encoding="utf-8"))["population_counts"]
    return int(counts[stratum])


def _latest_human_topical(engine: Engine) -> dict[str, str]:
    classifications_human = reflect_table(engine, "classifications_human")
    query = (
        select(classifications_human.c.record_id, classifications_human.c.label)
        .where(classifications_human.c.dimension == "topical_inclusion")
        .order_by(classifications_human.c.timestamp)
    )
    with engine.connect() as conn:
        return {record_id: label for record_id, label in conn.execute(query)}


def write_v2_results(
    result: V2ValidationResult,
    report_path: Path = DEFAULT_V2_RESULTS_PATH,
    json_path: Path = DEFAULT_V2_RESULTS_JSON,
) -> None:
    inc = result.strata[V2_INCLUDE]
    probe = result.strata[V2_LOSS_PROBE]
    lo, hi = inc.exact_ci

    lines = [
        "# Phase 12b results — v2 screening criterion, measured without resubstitution",
        "",
        "*Generated by `review-cli classify report-v2`. Do not hand-edit; regenerate.*",
        "",
        "## Precision of a v2 `include` (the estimand)",
        "",
        f"- **{inc.includes}/{inc.n} = {inc.rate:.1%}** of v2 includes are real includes.",
        f"- exact 95% CI {lo:.1%}–{hi:.1%}; FPC half-width ±{inc.half_width * 100:.1f}pp "
        f"(N = {inc.population_n}).",
        f"- Phase 10b's resubstitution estimate on the contaminated records: "
        f"{result.resubstitution_includes}/{result.resubstitution_n} = "
        f"{result.resubstitution_rate:.1%}.",
        f"- **Optimistic bias: {result.bias * 100:.1f}pp.**",
        "",
        "## Loss probe — a bound, not a rate",
        "",
        f"- {probe.includes}/{probe.n} = {probe.rate:.1%} of the records v1 admitted and v2 "
        f"rejected were real includes.",
        f"- exact one-sided 95% upper bound **{probe.upper_bound:.1%}**, i.e. at most "
        f"{probe.upper_bound * probe.population_n:.0f} of {probe.population_n} records.",
        "- Reported as an upper bound only. No recall figure is derived from it (D-35, D-48).",
        "",
    ]
    if any(result.unadjudicated.values()):
        lines += [
            "⚠ **Incomplete.** Unadjudicated records remain: "
            + ", ".join(f"{k} {v}" for k, v in result.unadjudicated.items() if v)
            + ". Figures above are provisional.",
            "",
        ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "phase": "12b",
        "generated_at": datetime.now(UTC).isoformat(),
        "precision": {
            "includes": inc.includes,
            "n": inc.n,
            "population_n": inc.population_n,
            "rate": inc.rate,
            "exact_ci": list(inc.exact_ci),
            "fpc_half_width": inc.half_width,
        },
        "loss_probe": {
            "includes": probe.includes,
            "n": probe.n,
            "population_n": probe.population_n,
            "rate": probe.rate,
            "one_sided_upper": probe.upper_bound,
            "bounded_records": probe.upper_bound * probe.population_n,
        },
        "resubstitution": {
            "includes": result.resubstitution_includes,
            "n": result.resubstitution_n,
            "rate": result.resubstitution_rate,
            "bias": result.bias,
        },
        "unadjudicated": result.unadjudicated,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
