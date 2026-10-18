"""Phase 10b: re-screening the corpus under the v2 criterion (D-41..D-44).

WHY A FRAME AND NOT A CENSUS
----------------------------
v1 screened all 7,850 D1 records. Re-screening all 7,850 under v2 is not
affordable, so Phase 10b re-screens a *frame*: a census of the strata where the
includes actually live, plus a probability sample of the one stratum that is
too large to census and too sparse to be worth it.

The 500-record Phase 11 validation sample measured the include density of each
AI-label stratum directly, which is what makes this frame defensible rather
than convenient:

    stratum                  N       n    human includes    density   est. includes
    ai_include            1,067    200        38            19.0%        203
    ai_uncertain            238     70         5             7.1%         17
    ai_exclude_borderline   772    130         4             3.1%         24
    ai_exclude_high       5,773    100         1             1.0%         58
                                                                        ----
                                                                         301

Read the last two columns together. The first three strata are 2,077 records --
26% of D1 -- and hold an estimated 244 of the 301 true includes, 81% of the
corpus, at an 11.7% density. The fourth is 74% of D1 and holds an estimated
19% of the corpus at a 1.0% density, and that 1.0% rests on a single observed
record, so its own estimate is worth 58 +/- 58.

Hence:

  - CENSUS the first three strata (2,077 records). No sampling error is
    introduced anywhere the corpus is actually being built, and no weighting is
    needed to state what the corpus contains.
  - SAMPLE the fourth (600 of 5,773, equal probability, seed 45). 600 is chosen
    for a +/-0.8pp half-width on a 1% density with the finite-population
    correction -- enough to tell "a few dozen missed records" from "several
    hundred", which is the only distinction the manuscript needs to make. It is
    NOT enough to pin the residual precisely, and the report says so.

Total: 2,677 records, 34% of D1, for an estimated 89% of the true includes.

WHAT THE SAMPLED STRATUM CAN AND CANNOT SUPPORT
-----------------------------------------------
Records the v2 pass includes from the sampled stratum join the corpus directly
-- they are real records, found by a real screening decision. What is *not*
licensed is treating the sample's include rate as a corpus census: the 5,173
unsampled high-confidence excludes remain unexamined under v2, and the residual
estimate carries the weight 5,773/600 = 9.62 and its confidence interval. Any
corpus count must state whether it is the enumerated set or the weighted
estimate. `rescreen_manifest.json` carries the weights so the distinction can
be computed rather than asserted.

A second limit, and the sharper one: the sampled stratum was excluded by v1
with high confidence, and v2 is *narrower* than v1. A narrower criterion cannot
re-admit what a wider one already rejected on topic. So a near-zero yield here
is evidence that v1 and v2 agree, not evidence that the stratum is empty. The
recall question is answered by the Phase 11 human sample, which is independent
of both criteria -- not by this pass.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Engine, select

from review_pipeline.src.db.tables import reflect_table

# Stratum names. These match the Phase 11 sampling strata (D-38) exactly, so
# the density table above and the frame below are indexed the same way.
AI_INCLUDE = "ai_include"
AI_UNCERTAIN = "ai_uncertain"
AI_EXCLUDE_BORDERLINE = "ai_exclude_borderline"
AI_EXCLUDE_HIGH = "ai_exclude_high"

CENSUS_STRATA: tuple[str, ...] = (AI_INCLUDE, AI_UNCERTAIN, AI_EXCLUDE_BORDERLINE)
SAMPLED_STRATA: tuple[str, ...] = (AI_EXCLUDE_HIGH,)
FRAME_STRATA_ORDER: tuple[str, ...] = CENSUS_STRATA + SAMPLED_STRATA

DEFAULT_HIGH_EXCLUDE_SAMPLE = 600
DEFAULT_SEED = 45  # distinct from Phase 11's 42/43 so the draws are independent

V2_PROMPT_VERSION = "screen-v2"


def stratum_of(label: str, confidence: str | None) -> str:
    """The Phase 11 stratum a v1 decision belongs to. `exclude` splits on
    confidence because that is where the two exclude frames differ (D-38)."""
    if label == "exclude":
        return AI_EXCLUDE_HIGH if confidence == "high" else AI_EXCLUDE_BORDERLINE
    if label == "uncertain":
        return AI_UNCERTAIN
    return AI_INCLUDE


def latest_v1_decisions(engine: Engine) -> dict[str, tuple[str, str | None]]:
    """record_id -> (label, confidence) from the v1 AI screening pass.

    Ordered by timestamp and overwritten as it goes, so a record screened more
    than once (batch overlap, the D-21 re-screen) resolves to its *last* v1
    decision rather than an arbitrary one -- the same rule agreement.py uses.
    """
    classifications_ai = reflect_table(engine, "classifications_ai")
    query = (
        select(
            classifications_ai.c.record_id,
            classifications_ai.c.label,
            classifications_ai.c.confidence,
            classifications_ai.c.prompt_version,
        )
        .where(classifications_ai.c.dimension == "topical_inclusion")
        .order_by(classifications_ai.c.timestamp)
    )
    decisions: dict[str, tuple[str, str | None]] = {}
    with engine.connect() as conn:
        for record_id, label, confidence, prompt_version in conn.execute(query):
            if prompt_version == V2_PROMPT_VERSION:
                continue  # v2's own rows must not redefine the frame they came from
            decisions[record_id] = (label, confidence)
    return decisions


@dataclass
class FrameResult:
    """The Phase 10b re-screen frame plus everything needed to report it."""

    members: dict[str, str] = field(default_factory=dict)  # record_id -> stratum
    population_counts: dict[str, int] = field(default_factory=dict)  # N_h over all of D1
    frame_counts: dict[str, int] = field(default_factory=dict)  # n_h actually taken
    seed: int = DEFAULT_SEED

    @property
    def weights(self) -> dict[str, float]:
        """N_h / n_h. Exactly 1.0 for every censused stratum, by construction."""
        return {
            stratum: self.population_counts[stratum] / self.frame_counts[stratum]
            for stratum in self.frame_counts
            if self.frame_counts[stratum]
        }


def build_frame(
    engine: Engine,
    high_exclude_sample: int = DEFAULT_HIGH_EXCLUDE_SAMPLE,
    seed: int = DEFAULT_SEED,
) -> FrameResult:
    """Census the three dense strata; draw `high_exclude_sample` from the sparse
    one at equal probability. Deterministic given `seed`."""
    v1 = latest_v1_decisions(engine)

    by_stratum: dict[str, list[str]] = {name: [] for name in FRAME_STRATA_ORDER}
    for record_id, (label, confidence) in v1.items():
        by_stratum[stratum_of(label, confidence)].append(record_id)

    result = FrameResult(seed=seed)
    for stratum in FRAME_STRATA_ORDER:
        # Sort before shuffling: dict iteration order follows insertion, which
        # follows a timestamp ordering, so an unsorted list would make the draw
        # depend on screening history rather than on the seed alone.
        pool = sorted(by_stratum[stratum])
        result.population_counts[stratum] = len(pool)

        if stratum in CENSUS_STRATA:
            taken = pool
        else:
            rng = random.Random(f"{seed}:{stratum}")
            taken = rng.sample(pool, min(high_exclude_sample, len(pool)))

        result.frame_counts[stratum] = len(taken)
        for record_id in taken:
            result.members[record_id] = stratum

    return result


def write_manifest(frame: FrameResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "10b",
        "prompt_version": V2_PROMPT_VERSION,
        "seed": frame.seed,
        "frame_size": len(frame.members),
        "strata": [
            {
                "stratum": stratum,
                "mode": "census" if stratum in CENSUS_STRATA else "probability_sample",
                "population_N": frame.population_counts[stratum],
                "frame_n": frame.frame_counts[stratum],
                "weight": frame.weights.get(stratum),
            }
            for stratum in FRAME_STRATA_ORDER
        ],
        "members": frame.members,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_manifest(path: Path) -> dict[str, str]:
    """record_id -> stratum, for callers that only need frame membership."""
    return json.loads(path.read_text(encoding="utf-8"))["members"]


def already_rescreened(engine: Engine) -> set[str]:
    classifications_ai = reflect_table(engine, "classifications_ai")
    query = select(classifications_ai.c.record_id).where(
        classifications_ai.c.dimension == "topical_inclusion",
        classifications_ai.c.prompt_version == V2_PROMPT_VERSION,
    )
    with engine.connect() as conn:
        return {row[0] for row in conn.execute(query)}


ABSTRACT_CHAR_CAP = 1400
"""Abstracts are truncated in the export. The two v2 gates -- is there a
generative system doing a work task, and is that task in a process/
manufacturing setting -- are stated in an abstract's opening move, not its
results. The D1 abstract-length distribution is median 1,283 / p90 1,940 /
max 27,683 characters, so a 1,400 cap leaves over half of them untouched and
trims only the tail. Truncated records carry `abstract_truncated: true` so a
decision made on a clipped abstract is identifiable after the fact."""


def export_rescreen_batch(
    engine: Engine,
    frame: dict[str, str],
    limit: int | None = None,
    abstract_chars: int = ABSTRACT_CHAR_CAP,
) -> list[dict]:
    """Next `limit` frame members with no v2 decision yet.

    Deliberately does NOT export the v1 label. The v2 pass must judge the
    record, not ratify or contradict a decision it can see -- otherwise the
    precision it reports is anchoring, not screening. Stratum is withheld for
    the same reason: it is derived from the v1 label.
    """
    records_table = reflect_table(engine, "records")
    done = already_rescreened(engine)
    pending_ids = sorted(rid for rid in frame if rid not in done)
    if limit is not None:
        pending_ids = pending_ids[:limit]
    if not pending_ids:
        return []

    with engine.connect() as conn:
        rows = conn.execute(
            records_table.select().where(records_table.c.record_id.in_(pending_ids))
        ).mappings().all()

    by_id = {r["record_id"]: r for r in rows}
    batch = []
    for rid in pending_ids:
        if rid not in by_id:
            continue
        abstract = by_id[rid]["abstract"] or ""
        entry = {
            "record_id": rid,
            "title": by_id[rid]["title"],
            "abstract": abstract[:abstract_chars],
            "venue": by_id[rid]["venue"],
            "year": by_id[rid]["year"],
            "publication_status": by_id[rid]["publication_status"],
        }
        if len(abstract) > abstract_chars:
            entry["abstract_truncated"] = True
        if not abstract:
            # 1,588 of 7,850 D1 records have no abstract at all. Flagging it
            # keeps "decided on the title alone" a countable category rather
            # than an invisible one.
            entry["abstract_missing"] = True
        batch.append(entry)
    return batch


@dataclass
class FrameProgress:
    per_stratum: dict[str, tuple[int, int]] = field(default_factory=dict)  # done, total

    @property
    def done(self) -> int:
        return sum(d for d, _ in self.per_stratum.values())

    @property
    def total(self) -> int:
        return sum(t for _, t in self.per_stratum.values())


def frame_progress(engine: Engine, frame: dict[str, str]) -> FrameProgress:
    done_ids = already_rescreened(engine)
    totals = Counter(frame.values())
    done = Counter(stratum for rid, stratum in frame.items() if rid in done_ids)
    return FrameProgress(
        per_stratum={s: (done.get(s, 0), totals[s]) for s in FRAME_STRATA_ORDER if totals[s]}
    )
