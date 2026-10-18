# Phase 10b — re-screening D1 under a revised criterion, on a census-plus-sample frame

*Written 2026-09-01, at completion of the frame. Companion to
`SAMPLING_AND_AGREEMENT.md` (Phase 11 design) and
`PHASE11_AGREEMENT_ANALYSIS.md` (Phase 11 results).*

---

## 1. Why there is a Phase 10b at all

Phase 10 screened all 7,850 D1 records under `screening_criteria.yaml` (v1).
Phase 11 validated a stratified sample of 500 of those decisions against a human
reviewer. The result, design-weighted:

| quantity | v1 |
|---|---|
| precision of an AI `include` | **19.0%** |
| false-negative rate | 32.7% |
| κ | 0.219 (κ_max 0.338 — 65% of ceiling) |
| Gwet's AC1 | 0.834 |
| weighted AUC over the 7-level ordinal score | **0.828** |

An AUC of 0.828 with a precision of 19.0% is a specific, diagnosable failure:
**the screener ranks records well and cuts them in the wrong place.** That is a
threshold problem, not a judgement problem, so the response is to rewrite the
criterion rather than to replace the screener.

## 2. What was wrong with v1, recovered from the reviewer's own rationales

Rather than guess at a rewrite, the reviewer's operating criterion was recovered
from the free-text rationales they typed on all 500 adjudicated records. Two
clauses in v1 were doing nearly all of the damage.

**Leak 1 — the scope clause.** v1's `scope` admitted "general-industrial-copilot
/ general-agentic-AI literature usable as RQ2/RQ3 comparison context". Read
literally, that admits the entire LLM-methods literature: LLaMA, Gorilla, HELM,
GraphRAG, MCP security, GUI agents, LLM-as-judge.

**Leak 2 — the domain clause.** v1's first inclusion criterion says "an
industrial or engineering task" while its own scope says "chemical, process, or
manufacturing". The screener followed the wider phrase, admitting geomatics,
geotechnics, photonics, structural and construction work.

The reviewer's applied criterion is a **conjunction**, and their rationales say
so in almost those words: 247 records were rejected as "not related to chemical
engineering", 204 as "not related to gen ai based industrial copilots". v2
(`config/screening_criteria_v2.yaml`) encodes the conjunction explicitly as two
gates that must both pass:

- `gate_artifact` — a generative / foundation-model system doing a work task;
- `gate_domain` — chemical, process or manufacturing **production**.

Two new exclusion reasons make the narrowing countable rather than silent:

- `context_only` — artifact passes, domain fails. This keeps the RQ2/RQ3
  comparison material *retrievable* instead of destroying it.
- `science_not_process` — chemistry/materials as discovery science, not as
  production.

`decision_rules.uncertain` was also tightened: `uncertain` is now reserved for
records whose abstract cannot settle a gate. Under v1, 240 records were marked
uncertain and the reviewer included only 7% of them.

## 3. The frame: census where the corpus is, probability sample where it is not

The user's instruction was to sample rather than re-screen all 7,850. Sampling
uniformly would have been the wrong answer, because Phase 11 had already
measured how unevenly the true includes are distributed. Using the Phase 11
per-stratum human include densities:

| stratum | N | n (Phase 11) | human includes | density | est. includes in N |
|---|---|---|---|---|---|
| ai_include | 1,067 | 200 | 38 | 19.0% | 203 |
| ai_uncertain | 238 | 70 | 5 | 7.1% | 17 |
| ai_exclude_borderline | 772 | 130 | 4 | 3.1% | 24 |
| ai_exclude_high | 5,773 | 100 | 1 | 1.0% | 58 |
| | | | | | **301** |

The first three strata are 2,077 records — 26% of D1 — and hold an estimated 244
of the ~301 true includes (81% of the corpus) at an 11.7% density. The fourth is
74% of D1 and holds an estimated 19% at a 1.0% density, and that 1.0% rests on a
**single** observed record, so its own estimate is worth 58 ± 58.

Hence the design: **census the first three strata, probability-sample the
fourth.**

| stratum | mode | N | frame n | design weight |
|---|---|---|---|---|
| ai_include | census | 1,067 | 1,067 | 1.00 |
| ai_uncertain | census | 238 | 238 | 1.00 |
| ai_exclude_borderline | census | 772 | 772 | 1.00 |
| ai_exclude_high | probability sample | 5,773 | 600 | 9.62 |
| **total** | | **7,850** | **2,677** | |

2,677 records — 34% of D1 — for an estimated 89% of the true includes, with
**zero sampling error in the strata where the corpus is actually built**.

Implementation: `src/screening/rescreen.py`, seed 45 (deliberately distinct from
Phase 11's 42/43 so the draws are independent), manifest written to
`outputs/rescreen_manifest.json`. `build_frame` sorts each pool before
`rng.sample` so the draw depends on the seed alone; `latest_v1_decisions` skips
rows whose `prompt_version == "screen-v2"`, so v2's own rows cannot redefine the
frame they came from.

**The sharpest limit on this design, stated up front:** the sampled stratum was
excluded by v1 *with high confidence*, and v2 is **narrower** than v1. A
narrower criterion cannot re-admit what a wider one already rejected on topic.
So a near-zero yield in `ai_exclude_high` is evidence that v1 and v2 agree, not
evidence that the stratum is empty. Only human adjudication can test that
stratum, and Phase 11b's 100 draws (1 include) are the only such evidence.

## 4. Blinding

`export_rescreen_batch` deliberately does **not** export the v1 label. The v2
pass judges the record, not the v1 decision — otherwise the precision it reports
is anchoring, not screening.

## 5. Execution

Nine batches of 300 (the last, 277), 2026-08-31 → 2026-09-01, coder
`claude-opus-5`, `prompt_version = screen-v2`. Per batch:

```
review-cli screen rescreen-export --limit 300
# render, adjudicate, expand to ScreeningDecision JSON
review-cli screen ai-ingest <batch>.json --coder claude-opus-5 --prompt-version screen-v2
```

## 6. Results

**2,677 / 2,677 re-screened. 348 includes, 2,326 excludes, 3 uncertain.**

| stratum | n | v2 includes | rate | weighted contribution |
|---|---|---|---|---|
| ai_include | 1,067 | 323 | 30.3% | 323 |
| ai_uncertain | 238 | 20 | 8.4% | 20 |
| ai_exclude_borderline | 772 | 4 | 0.5% | 4 |
| ai_exclude_high | 600 | 1 | 0.2% | 10 |
| **total** | **2,677** | **348** | **13.0%** | **≈357** |

Design-weighted estimate of the D1 corpus under v2: **≈357 records**, against
Phase 11's human-anchored estimate of ~301. The two are the same order and the
gap is expected: v2 is an AI pass, not an adjudicated one, and its own precision
against the reviewer is 32.0% (§7, as corrected by Phase 12b). Applying that
honest precision brings the estimate down to **≈162**
(`PHASE12B_V2_VALIDATION.md` §6.6), and that is the figure the manuscript uses.

Exclusion reasons, from the recorded rationales:

| reason | n | share of excludes |
|---|---|---|
| `context_only` (gate_domain fails; generative, wrong domain) | 1,194 | 51.3% |
| `science_not_process` (chemistry/materials as discovery science) | 422 | 18.1% |
| `wrong_technology` (gate_artifact fails; AI is not generative) | 391 | 16.8% |
| `off_topic` (no GenAI system doing a work task) | 300 | 12.9% |
| `benchmark_other_domain` | 10 | 0.4% |
| `not_original_research` | 9 | 0.4% |

**Reading:** 69.4% of exclusions are *domain* failures on records whose artifact
gate passes. The corpus is not being trimmed of AI papers; it is being trimmed
of AI papers about something else. That is exactly what Leak 1 predicted, and it
is the reason `context_only` was made a countable reason rather than folded into
`off_topic` — 1,194 records of RQ2/RQ3 comparison material remain addressable.

## 7. v2 measured against the 500 human decisions — **apparent, not honest**

414 of the 500 human-adjudicated records fall inside the Phase 10b frame
(ai_include 200, ai_exclude_borderline 130, ai_uncertain 70, ai_exclude_high 14
— the last is small because Phase 11b's 100 and Phase 10b's 600 were drawn under
independent seeds).

|  | v2 include | v2 exclude | v2 uncertain |
|---|---|---|---|
| **human include** (47) | 33 | 13 | 1 |
| **human exclude** (367) | 25 | 341 | 1 |

- precision **56.9%** (v1: 19.0%) — a 3.0× improvement
- recall **70.2%**
- design-weighted figures are identical to unweighted here, because all but 14
  of the overlapping records sit in weight-1.00 census strata.

⚠ **SUPERSEDED — this is a resubstitution estimate and was optimistically biased
by 24.9pp.** v2 was written *after* inspecting 45 of v1's false positives and all
10 of its false negatives from this same 500. Those records are, in the
statistical sense, training data.

Phase 12b measured the same quantity on 100 records adjudicated in neither phase
(drawn 2026-09-01 under seed 46, adjudicated 2026-09-02; design pre-registered in
`PHASE12B_V2_VALIDATION.md`, D-47…D-52):

| | resubstitution (here) | **honest (Phase 12b)** |
|---|---|---|
| precision of a v2 `include` | 56.9% | **32.0%** (95% CI 23.0–42.1) |
| improvement over v1's 19.0% | 3.0× | **1.7×** |

**56.9% and 3.0× are withdrawn.** They may appear in the manuscript only where
the point being made is the bias itself. Every substantive claim about v2's
precision cites 32.0%, and every corpus-size figure downstream of §6 is
re-estimated in `PHASE12B_V2_VALIDATION.md` §6.6.

The 70.2% recall in this section is withdrawn on the same grounds and is **not**
replaced: Phase 12b bounds loss rather than estimating recall (§6.2 there), and no
recall figure for v2 will exist.

## 8. What v2 gets wrong, by name

v2 does **not** recover v1's false negatives — exactly as a narrower criterion
must not. The specific human-INCLUDE records that v2 excludes are the real
boundary disagreement, and they are all `context_only` or `science_not_process`:

- the FDD-teaching-model paper → `context_only`
- ReLM chemical reaction prediction → `science_not_process`
- the maritime/diving accident NLP paper → `context_only`
- the LLM construction accident report paper → `context_only`

These are not screener errors under v2's stated criterion; they are the cost of
the criterion, and they should be reported as such rather than repaired
silently. The reviewer's own boundary is not fully self-consistent here either
(they included an LLM construction accident report paper and excluded a
construction safety RAG paper), which `known_boundary_noise` in
`screening_criteria_v2.yaml` records.

## 9. Limits

1. §7's precision was resubstitution-biased by 24.9pp. **Phase 12b has replaced
   it: 32.0%, not 56.9%.**
2. 5,173 of the 5,773 high-confidence excludes were never re-screened. The
   design accepts this and prices it: weight 9.62, one observed include, so the
   stratum's contribution to the corpus estimate (≈10 records) is the least
   reliable number in §6. Phase 12b quantified what this costs and did not
   improve it: at most ≈269 eligible records may be hiding there, point estimate
   ≈58 against an estimated corpus of ≈162. This is the review's single largest
   limitation (`PHASE12B_V2_VALIDATION.md` §6.7).
3. v2 is an AI criterion applied by an AI screener. The corpus that enters
   synthesis must still be human-adjudicated; Phase 10b narrows *what has to be
   adjudicated* from 7,850 to 348.
4. `wrong_technology` (391 records, 16.8%) is a v1 leak that the rewrite did
   **not** target: right-domain PSE papers with no generative component. v2
   catches them via `gate_artifact`, but the leak was not diagnosed in advance
   from the reviewer's rationales — it surfaced during execution.
