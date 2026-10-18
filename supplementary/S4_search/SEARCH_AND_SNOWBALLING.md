# Search, Snowballing, and Saturation: Decisions Log for the 2026-08-29 Corpus Rebuild

Status: written **during** the rebuild, decision by decision, so the manuscript's
search-strategy subsection can be assembled from a contemporaneous record rather
than reconstructed after the fact. Every count below was read from
`review_pipeline/data/normalized/review.db` at the time the decision was taken;
where a count later changed, both values are given.

Companion documents: `SAMPLING_AND_AGREEMENT.md` (validation sample and κ),
`TECHNICAL_IMPLEMENTATION_PLAN.md` (phase definitions).

---

## 0. Why this corpus was rebuilt

On 2026-08-28 the entire `review_pipeline/data/` working tree was deleted. The
DB was gitignored at the time and had no backup, so the loss included the
primary corpus (~1,839 D1 records), the AI screening decisions, and — the only
part not regenerable by re-running the pipeline — roughly 698 human topical
decisions and 1,378 human dimension classifications.

Forensic recovery (`ext4magic` journal and magic-scan modes, `debugfs lsdel`)
returned 8,406 files, including the complete full-text PDF cache (141 PDFs), 173
pipeline logs, and a JSON export listing 535 records of the lost corpus by ULID
and title. `review.db` itself was not recoverable: it had been rewritten
continuously over four days (31 ML retrains), so its page-1 block had been
reused and the file could not be carved.

**Consequence for the manuscript.** The search was re-executed from the
configuration, which is deterministic and reproducible; the human classifications
were not reconstructed, because fabricating labels that feed a published
agreement statistic would make that statistic false. The 535 salvaged records are
used only as an *external coverage check* on the rebuild (§5), never as data.

The rebuilt corpus is therefore a genuine re-run, not a restoration, and the
manuscript should describe it as such.

---

## 1. Seed search (iteration 1)

Fifteen query families (CE1–CE15) from `config/search_queries.yaml` were run
against Crossref, OpenAlex, and Semantic Scholar.

| | |
|---|---|
| Raw records ingested | 1,119 |
| Unique records after dedup | 71 |
| D1-eligible | 53 (all `core`) |

**Decision 1.1 — CE1–CE15 is a seed set, not the corpus.** The fifteen families
share a `genai + domain` query prefix and differ in their disjunctive tails.
Crossref and OpenAlex rank by relevance rather than evaluating Boolean grammar,
so the fifteen families return near-identical result sets: 1,119 raw records
collapse to 71 unique. This is a property of the APIs, not a defect in the query
design, and it means recall in this review comes from citation-graph traversal
rather than from query breadth. The manuscript must state this plainly rather
than implying fifteen independent searches.

**Decision 1.2 — a repeated deterministic iteration was abandoned.** A second
pass of the same fifteen families (iteration 2) added 1,053 raw records and
**zero** unique titles, as it must: the queries are deterministic and the
databases had not changed. Repeating a query is not a saturation observation.
Saturation is assessed across *snowballing rounds*, where each round has a
genuinely different seed set (§4).

**Decision 1.3 — Semantic Scholar is inoperative for query search, and is
reported as such.** Across CE search runs, Semantic Scholar returned 24 errors
and 10 zero-result responses and **zero** successful runs — the same behaviour
the pre-loss corpus showed (71 errors, 19 zero-results, 0 ok), which means the
original 1,839-record corpus also came entirely from Crossref and OpenAlex. The
query was deliberately *not* reshaped mid-rebuild: changing a connector's query
form partway through would mean the corpus was assembled under two different
search protocols, which is worse for the methods section than a documented
limitation. Semantic Scholar is reported as contributing no records to the
query-based search.

Note the asymmetry established later: the same connector works well for
*citation* lookups (§3), so the limitation is specific to its search endpoint as
this pipeline calls it.

---

## 2. Snowballing round 1 — all D1 seeds (iteration 2)

Backward (references) and forward (citations) traversal from all 53 D1 seeds,
via OpenAlex and Semantic Scholar, tagged `corpus_type='snowball'` and passed
through the same dedup and eligibility phases as primary records
(`src/search/citation_trace.py`).

| | |
|---|---|
| Seeds traced | 53 |
| Candidates returned | 5,605 |
| Records considered at dedup (D0) | 7,848 |
| Duplicate groups / records merged away | 1,800 / 2,339 |
| **D1-eligible after round 1** | **1,713** (53 core, 1,660 contextual) |
| excluded: before temporal scope | 1,043 |
| excluded: not peer reviewed | 581 |

Round 1 multiplied the corpus by 32×. It confirms Decision 1.1: the citation
graph, not the query set, is what builds this corpus.

**Decision 2.1 — an anonymous-pool retry policy was added.** Semantic Scholar's
keyless pool is shared and returns 429 under contention. A dedicated
`ANONYMOUS_POOL_RETRY` policy (6 attempts, 4 s base backoff, 60 s cap) is applied
only when no API key is present; keyed requests keep the original policy, since
waiting out a rate limit on a private budget is pointless
(`src/search/connectors/semantic_scholar.py`). This changed S2 citation lookups
from unusable to mostly working (§3).

---

## 3. Snowballing round 2 — restricted to citation-graph core seeds (iteration 3)

Round 2 could not simply repeat round 1 with the enlarged seed set. Two
constraints:

- Re-tracing the 53 round-1 seeds is a **guaranteed no-op**: the citation graph
  is deterministic, so those API calls would re-ingest records already held.
- Tracing all 1,663 untraced D1 seeds costs roughly **8 hours** of API time.

**Decision 3.1 — round 2 traces the citation-graph core, defined by
co-discovery degree ≥ 3.** A record's *co-discovery degree* is the number of
distinct search runs that returned it, counted through the canonical record so
that hits landing on duplicates merged away in Phase 8 still credit the survivor.
Degree *k* means *k* independent retrieval paths — CE query strings, plus the
reference or citation lists of *k* distinct seeds — converged on the same work.
That is the citation-graph sense of "core": a hub the field keeps pointing at,
as opposed to a leaf reached exactly once.

The degree distribution over the 1,713 D1 records is strongly skewed, which is
what makes the cut meaningful rather than arbitrary:

| Co-discovery degree | D1 records (cumulative) | Untraced seeds | Est. runtime |
|---|---|---|---|
| ≥ 1 | 1,713 | 1,663 | ~8 h |
| ≥ 2 | 1,038 | 988 | ~4.7 h |
| **≥ 3** | **149** | **103** | **~30 min** |
| ≥ 4 | 110 | 64 | ~18 min |
| ≥ 5 | 72 | 28 | ~8 min |

Degree ≥ 3 was chosen: it sits just past the elbow of the degree distribution
(889 of the 1,713 records have degree exactly 2, and only 149 have 3 or more),
so it separates hubs from the bulk without a hand-picked threshold, and it
returns a round in ~30 minutes instead of ~8 hours.

**This is a deliberate restriction of recall, and the manuscript must say so.**
Round 2 is *not* a full second-order traversal. It tests whether tracing the
most-converged-upon works surfaces materially new literature; if it does, further
rounds are warranted, and if it does not, that is evidence for saturation of the
*core* region of the graph specifically — not of the whole D1 pool. A round over
the degree-≥2 tier remains available if the manuscript needs the stronger claim.

**Decision 3.2 — already-traced seeds are skipped explicitly.** The
`--skip-already-traced` flag reads the seed record id back out of the
`SNOWBALL.<direction>.<connector>.<seed_id>` query ids in `search_runs` and drops
those seeds. This makes rounds composable — round *n* covers what rounds 1…*n*−1
did not — and keeps the saturation curve honest, because re-ingested known
records would otherwise inflate a round's raw yield while adding no unique
titles.

Both filters default to off (`--min-codiscovery-degree 1`, no skip), so an
unrestricted full-corpus round remains the default behaviour of the command.

Round 2 result: **18,976 candidates from 103 seeds** (1 skipped, no DOI), against
5,605 from 53 seeds in round 1. Semantic Scholar citation lookups succeeded
throughout (74 ok / 29 zero-results / 8 errors), confirming that Decision 1.3's
limitation is confined to its search endpoint.

After dedup and eligibility over the whole corpus:

| | Round 1 | Round 2 |
|---|---|---|
| Records considered (D0) | 7,848 | 26,848 |
| Duplicate groups | 1,800 | 4,822 |
| Records merged away | 2,339 | 8,651 |
| Preprint/published pairs logged | 102 | 1,486 (1,244 auto-linked, 242 pending human review) |
| **D1 eligible** | **1,713** | **7,794** (56 core, 7,738 contextual) |

**Decision 3.3 — the hub criterion is topically uncontrolled, and the round's
yield reflects that.** Co-discovery degree measures convergence in the citation
graph, not relevance to the review. Landmark AI papers are cited by everything,
including by this corpus's own domain papers, so they accumulate degree without
being about industrial copilots. Attributing the 9,210 newly-D1 records to the
seeds that found them shows both effects: the highest-yielding seeds are mostly
on-topic (LLM agents in chemistry, ChatMOF, autonomous chemical research,
predictive chemistry), but the same list contains *PubChem 2025 update*,
*Performance of ChatGPT on USMLE*, *Chatting and cheating: academic integrity*,
and *Evolutionary-scale prediction of atomic-level protein structure* (ESMFold).
The top 15 seeds account for 42% of the yield and 88 seeds contributed at all.

This does not invalidate the round -- topical filtering is Phase 10's job, not
D1's -- but it does constrain what the saturation curve means (§4).

---

## 4. Saturation

Saturation is tracked by `earliest_seen_iteration` on each record and read out by
`review-cli iterate stats` / `iterate report`: per iteration, the count of
records seen for the first time, the cumulative unique total, and the marginal
growth rate, with an elbow test on the marginal series.

| Iteration | Kind | New unique records | Cumulative |
|---|---|---|---|
| 1 | CE1–CE15 seed search | 71 | 71 |
| 2 | Snowball, all 53 D1 seeds | 3,266 | 3,337 |
| 3 | Snowball, core seeds (degree ≥ 3) | 10,349 | 13,686 |

D1 grew 1,713 → 7,794 and the elbow test still reports "no elbow yet -- corpus
is still growing substantially."

**Decision 4.3 — D1 growth is the wrong saturation signal, and the reported
curve must be the post-screening one.** D1 applies only temporal scope and
peer-review status; it carries no topical test, so a round that traverses an
off-topic region of the citation graph (Decision 3.3) inflates D1 without adding
any literature the review would cite. The curve above is therefore an upper
bound on real growth. The saturation claim in the manuscript will be made on
records that pass Phase 10 topical screening, with this D1 curve reported
alongside it as the retrieval-level counterpart. Phase 10 has not yet been run
on the enlarged corpus.

**Decision 4.1 — the deterministic repeat is excluded from the curve.** The
abandoned iteration-2 query repeat (Decision 1.2) contributes no point. A
saturation curve is only interpretable if each point represents a genuine new
search effort; a repeated deterministic query would plant a false zero-growth
point and make the corpus look saturated when nothing had been searched.

**Decision 4.2 — stopping is decided on the curve, not on a fixed round count.**
Rounds continue until marginal new-unique growth flattens. With two points, no
elbow is estimable; the tool correctly reports "no elbow yet — corpus is still
growing substantially." Round 3 supplies the third point, which is the minimum
for any elbow claim, and the decision to stop or to run a degree-≥2 round will be
made from it.

---

## 5. External coverage check against the lost corpus

The 535 salvaged records of the pre-loss corpus (ULID + title only) provide an
independent check on whether the rebuild recovers the same literature. Matching
is by normalized title with prefix matching, because the salvaged titles are
truncated at ~90 characters.

| | |
|---|---|
| Coverage of salvaged records, D0 | 71% |
| Coverage of salvaged records, D1 | 69% |

An earlier figure of 49% was a measurement error on my side: exact-match
normalization failed on every truncated title. The corrected figures are above.

The ~155 unmatched records skew toward LAPSE preprints and materials-science
work. This is expected under Decision 1.1 — those records entered the original
corpus by citation paths not yet retraced — and is one of the things further
snowballing rounds are meant to close. Coverage is a diagnostic, not a target:
the rebuilt corpus is defined by the protocol, not by resemblance to the lost one.

---

## 6. Persistence and backup policy

Adopted 2026-08-29, in direct response to §0.

**Revised the same day, once the corpus grew.** The snapshot gzipped to 26 MB at
7.8k records but **90 MB at 26.8k**, against a 100 MB per-file limit — tracking
it in git was a backup that would have stopped working mid-project, silently.
The split now is: git carries `human_classifications.json.gz` (`review-cli db
export-human`), the human decisions alone, at KB scale and diffable per commit;
the full snapshot goes to the same Azure Blob container as the PDF cache
(`review-cli db blob-push`, Cool tier). Everything git no longer holds is
regenerable by re-running the pipeline; what git does hold is exactly what is
not. The 7,872-record snapshot committed earlier remains in git history.

The live `review.db` is 139 MB at 7.8k records — of which ~108 MB is the
`raw_metadata_json` provenance column — against a 100 MB per-file platform limit,
so it cannot be tracked directly. It gzips to ~27 MB, so the tracked artefact is a
compressed snapshot: `review-cli db snapshot` writes `review.db.gz`,
`review-cli db restore` reads it back, and the snapshot is committed at every
pipeline milestone.

Two failure modes are handled explicitly, both covered by
`tests/test_db_snapshot.py`: the snapshot is staged to a `.partial` file and
renamed only on success, so an interrupted run cannot leave a truncated snapshot
that a later restore would trust; and restore refuses to overwrite an existing
DB without `--force`, since the live file may hold work newer than the snapshot.

Dropping `raw_metadata_json` would fit the DB under the limit uncompressed, but
it was rejected: re-querying an API today returns a different response than it
did on the collection date, so those payloads are provenance evidence and are not
re-obtainable.

---

## 7. Scope amendment: manufacturing, and benchmark generation as an object of study

Taken 2026-08-30, after round 2 and after 250 records had been screened. The
trigger was the thesis this review supports: it proposes constructing a
**benchmark generator for training industrial copilots**, and intends the review
to establish whether such a thing already exists.

**Decision 7.1 -- the domain block was widened to include manufacturing.** The
review's declared scope is chemical, process *and* manufacturing engineering,
but `domain_block` carried no general manufacturing term (only "pharmaceutical
manufacturing"). Since the client-side `genai AND domain` test is what actually
decides D0 membership -- not the connectors' relevance ranking -- the retrieval
filter was narrower than the review it serves. Added: `manufacturing`,
`industrial automation`, `smart manufacturing`, `process manufacturing`,
`industrial process`, `shop floor`, `plant operations`.

Every family is re-run under the widened block as **iteration 4**, rather than
appending the new terms only to new families. A corpus assembled half under one
domain filter and half under another cannot be described by a single protocol,
and the methods section has to describe a single protocol.

**Decision 7.2 -- three families added for benchmark construction.** CE15
already retrieved benchmark *datasets*; CE16-CE18 retrieve the machinery that
produces them:

| Family | Target |
|---|---|
| CE16 | benchmark generation and generators, automatic benchmark construction, task/scenario/test-case generation |
| CE17 | evaluation environments, testbeds, sandboxes, agent benchmarks, evaluation harnesses, digital twins, virtual commissioning |
| CE18 | synthetic training data, instruction tuning and instruction datasets, fine-tuning corpora, self-instruct, curriculum learning |

**Decision 7.3 -- benchmark and evaluation work is now an inclusion criterion,
and cross-domain benchmarks get their own exclusion reason.** A claim that
industrial-copilot benchmark generators do not exist is only credible if the
search actively looked for what would refute it; until this amendment, the
criteria would have *excluded* exactly that literature as off-topic.

The corpus scope stays narrow by explicit decision: benchmarks for coding, web
browsing and general reasoning agents (SWE-bench and its variants, BrowseComp,
MLGym, BEARCUBS, ProjectEval, DSGBench) remain **out**. They are excluded under
a dedicated reason, `benchmark_other_domain`, so the decision log can still
report how many were retrieved and set aside.

⚠ **The limitation this creates, which the manuscript must state.** The review
does not synthesise the cross-domain agent-benchmark literature, so its absence
claim is scoped to the industrial/process/engineering domain and cannot be
phrased as an absence in the agent-benchmark field at large. The dedicated
exclusion reason is what keeps the contrast quantifiable despite that.

Early signal, before iteration 4 completes: of the 7,794 D1 records, 123 carry
"benchmark" in the title and 292 "evaluat*", but only 3 mention "synthetic data"
and none "training data" -- consistent with a field that benchmarks copilots
occasionally and generates benchmarks for them almost never. This is an
observation about titles, not yet a screened result, and is recorded here only
as the reason the amendment was thought worth its cost.

**Decision 7.4 -- the 250 already-screened records are re-screened.** Criteria
changed after screening began, so those records were judged against a different
rule set than the rest of the corpus would be. Re-screening 250 records is cheap;
an agreement statistic computed over a corpus screened under two rule sets is not
defensible at any price.

**Iteration 4 result.** CE1-CE18 under the widened block returned 1,790 raw
records that collapsed to **56 unique** and **41 new D1** -- marginal growth of
0.4%, against 310% for the preceding snowball round.

The 41 are small in number and large in relevance: real-time furnace operation
support with multi-agent RAG, LLM-driven discrete-event simulation, manufacturing
decision support, BOM digitisation, autonomous test-case generation in embedded
systems manufacturing, LLM-generated safety justifications from sensor signals,
conversational process mining over production data, facility layout design.
These are industrial-copilot papers the corpus did not contain, and could not
have contained: the domain filter had no manufacturing term to let them through.
CE16/CE17/CE18 contributed 27/26/25 new D1 records respectively.

⚠ **The 0.4% is not comparable to iteration 3's 310%.** Iteration 4 is a *query*
iteration; iteration 3 was a *citation-graph* iteration. What the flattening
shows is that the query channel is exhausted -- new query formulations now
return almost nothing the citation graph has not already delivered. It says
nothing about whether the citation graph is exhausted, and the manuscript must
not present it as general saturation. Saturation of the two channels has to be
reported separately, and the claim that matters (D-11) is the post-screening one.

That D1 per family lands in a narrow 20-27 band across all eighteen families is
itself a restatement of Decision 1.1: the families overlap heavily because the
connectors rank rather than filter, so the same records answer many queries.

---

## 8. Open decisions

1. **Whether to run a degree-≥2 snowball round** (988 untraced seeds, ~4.7 h).
   Depends on the round-3 saturation point (§4) and on whether the reported claim
   needs to be about the whole D1 pool rather than its core.
2. **Semantic Scholar's search endpoint** (§1.3) — documented as inoperative for
   this rebuild; reshaping its query is a post-hoc protocol change and should be
   deferred to a future review rather than applied mid-corpus.
