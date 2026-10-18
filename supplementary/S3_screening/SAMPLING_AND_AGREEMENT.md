# Validation Sampling and AI–Human Agreement: Design and Pre-Specified Stopping Rule

Status: pre-specified **2026-08-29**, before any record of the rebuilt corpus was
reviewed. This document exists to be citable from the manuscript's methods
section, and to fix the analysis decisions *in advance* of seeing the data they
apply to.

---

## 1. What is being estimated, and why it needs a sample

Phase 10 screens every D1-eligible record with an AI `topical_inclusion`
decision. Phase 11 has a human reviewer independently decide a subset of the
same records, blind to the AI's label (enforced structurally by
`src/validation/blinded_queue.py`, checked by `tests/test_blinded_queue.py`).
Phase 12 compares the two.

The reported quantity is **Cohen's κ between the AI screen and the human
reviewer** on `topical_inclusion`. κ is used rather than raw percent agreement
because the include rate is far from balanced: with an include prevalence near
20%, two raters who never communicate still agree about 68% of the time by
chance alone, so raw agreement overstates concordance. κ is the proportion of
agreement beyond that chance baseline.

Two framing constraints, both of which the manuscript must state:

- This is **AI–human concordance, not inter-rater reliability.** There is one
  human coder. κ here quantifies how far an automated screen can be trusted
  against that coder's judgement; it does not establish that the coding scheme
  is reproducible across independent humans, which would need a second coder.
- Screening every D1 record by hand is not feasible for a single reviewer
  (the D1 pool runs to ~1,800 records), which is precisely why the estimate is
  sample-based and why its precision has to be stated rather than assumed.

---

## 2. Sample size by precision, not by power

The sample is sized on the **width of the confidence interval for κ**, following
the confidence-interval approach to sample size for interobserver agreement
studies (Donner & Eliasziw 1992; Rotondi & Donner 2012, implemented in the R
package `kappaSize` as `CIBinary`). Implementation:
`src/validation/sample_size.py`, tested in `tests/test_sample_size.py`.

Sizing on precision rather than on power to reject a null κ is the correct frame
here: the review is not testing "is agreement better than chance" — a test that
would be uninformative, since it nearly always rejects — it is reporting an
estimate and the uncertainty around it.

For a binary rating by two raters, the large-sample variance of the intraclass κ
is (Bloch & Kraemer 1989, *Biometrics* 45:269–287):

```
Var(κ) = (1 − κ)/n · [ (1 − κ)(1 − 2κ) + κ(2 − κ) / (2·π·(1 − π)) ]
```

where `π` is the prevalence of the positive category (the proportion of records
called `include`). Setting the Wald half-width `d = z₁₋α/₂ · √Var(κ)` and solving
for `n` gives the required sample size.

Two properties of this formula drive the design, and both are worth stating in
the manuscript because they are counter-intuitive:

1. **Cost is quadratic in precision.** Halving the interval quadruples `n`.
2. **Imbalance is expensive.** The `2π(1−π)` denominator means a screen that
   includes 15% of records needs roughly twice the sample of a balanced one for
   the same precision. A selective screen is a statistically costly one.

### Anticipated κ

The calculation requires an anticipated κ. This review uses **κ = 0.60**
(`sample_size.ANTICIPATED_KAPPA`). Recent concordance analyses of LLM-assisted
screening report κ ≈ 0.7 for systematic reviews with tight, mechanical inclusion
criteria, dropping to roughly 0.55–0.60 for scoping and narrative reviews whose
criteria are broader and more judgement-laden. This is a structured narrative
review over a deliberately broad D0 discovery pool — the harder case — so sizing
on the optimistic 0.7 would under-size the sample.

Note this assumption **cuts the safe way**: across this range, a lower
anticipated κ requires a *larger* n, so guessing low buys precision rather than
losing it.

### Precision available at each sample size

At κ = 0.60 and π = 0.20 (α = 0.05):

| n | 95% CI for κ | Supports the claim… |
|---:|---|---|
| 90 | [0.39, 0.81] (±0.21) | nothing usable — spans *fair* to *almost perfect* |
| 120 | [0.42, 0.78] (±0.18) | at least moderate, with no margin |
| 150 | [0.44, 0.76] (±0.16) | at least moderate |
| 200 | [0.46, 0.74] (±0.14) | at least moderate |
| 300 | [0.49, 0.71] (±0.11) | at least moderate, robust to κ as low as 0.55 |
| 400 | [0.50, 0.70] (±0.10) | at least moderate |

**This supersedes the previous fixed target of n = 90**
(`sampling.DEFAULT_TARGET_SIZE`). That value was a chosen round number, and at a
realistic prevalence it supports only the interval [0.39, 0.81] — which spans
three of Landis & Koch's four interpretive bands and therefore constrains
essentially nothing. Carrying it forward unexamined would have been a defect in
the earlier design.

---

## 3. Pre-specified stopping rule

Because the required `n` depends on the true κ, which is unknown until measured,
the review uses a **sequential rule fixed in advance**:

1. Review a first block of **150** records from the stratified sample (§4).
2. Compute κ and its 95% CI on all paired decisions so far.
3. **Stop** when both hold:
   - the CI lower bound exceeds **0.41** (Landis & Koch's *moderate* floor), and
   - the half-width is at most **±0.15**.
4. Otherwise extend to **300** and re-evaluate. If the rule still fails at 300,
   report the interval as observed and discuss the limitation — do not keep
   extending until the desired conclusion appears.

The rule is evaluated at each retrain batch and displayed live during
`review-cli classify assisted` (`_print_ai_agreement_panel`), so the reviewer
sees the interval tighten and stops as soon as it qualifies.

**Why this is pre-specified.** Inspecting an estimate and then deciding whether
to collect more data is optional stopping, which inflates the realised error rate
of a naively computed interval. Fixing the rule, its thresholds, and its two
candidate sample sizes *before* any record is reviewed is what keeps the reported
interval interpretable. The manuscript should report that the rule was
pre-specified, at which n it stopped, and the interval at that point.

A residual caveat to state plainly in the limitations: even pre-specified,
two-stage designs leave the final interval marginally anti-conservative relative
to a fixed-n design, because the stopping point is data-dependent. With only one
interim look and a pre-fixed extension, the distortion is small, but it is not
zero and should not be presented as if it were.

---

## 4. Stratification

The sample is stratified rather than simple-random, because a simple random draw
from a pool where ~80% of records are excluded would spend most of the
reviewer's effort confirming easy exclusions and produce few informative cases.
Strata (`sampling.STRATUM_QUOTA_FRACTIONS`, unchanged in this revision):

| Stratum | Share | Rationale |
|---|---:|---|
| `low_confidence` | 25% | AI `topical_inclusion` confidence low/medium — where the screen is least trustworthy and human review adds most |
| `excluded_borderline` | 15% | AI said exclude without high confidence — the false-negative risk that matters most in a review |
| `chemical_pse` | 20% | Title/abstract matches the chemical/PSE domain terms — the review's core domain |
| `thematic` | 20% | Matches the brief's named technical themes (safety, digital twin, multi-agent, …) |
| `random_common` | remainder | Random fill from the rest of the D1 pool, so high-confidence decisions are also represented |

Stratification uses only signal available at this point in the pipeline: the
Phase 10 AI `topical_inclusion` decision and record metadata. It deliberately
does **not** use domain/technology labels, since assigning those is exactly what
the human pass validates.

**Consequence for the estimate, which must be reported:** these strata are
sampled at deliberately unequal rates, so the sample is *not* self-weighting.
An unweighted κ over it estimates agreement in the sample, not in the D1 pool —
and it is biased downward relative to the pool, because hard cases are
over-represented by design. The manuscript should report the unweighted κ as the
primary figure (it is the conservative one, and the one the stopping rule is
defined on) and state this explicitly rather than implying pool-level
generalisability.

Sampling is deterministic: `build_validation_sample(seed=42, target_size)`
returns the same rows for the same seed and database state, and
`write_sample_files()` records the exact stratum definitions and realised
per-stratum counts (`outputs/sampling_manifest.json`,
`outputs/validation_sample.csv`).

---

## 5. What is reported

From `review-cli agreement` (`src/validation/agreement.py`):

- **n** paired decisions, and the realised per-stratum counts.
- **Raw percent agreement** — reported alongside κ, never instead of it.
- **Cohen's κ** with its 95% CI, and the n at which the stopping rule fired.
- **Confusion matrix**, and the **false-negative rate**: records the human
  included that the AI excluded. For a literature review this is the
  consequential error — a missed relevant paper never reaches synthesis, whereas
  a false positive is caught at full-text screening — so it is reported
  separately and not folded into κ.
- The **anticipated κ and prevalence** used for planning, next to the observed
  ones, so the gap between plan and outcome is visible.

Interpretation follows Landis & Koch (1977): 0.00–0.20 slight, 0.21–0.40 fair,
0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect. These bands
are conventional and arbitrary; the interval is the substantive result, and the
band label is a reading aid, not a finding.

---

## 6. References

- Bloch DA, Kraemer HC (1989). 2×2 kappa coefficients: measures of agreement or association. *Biometrics* 45:269–287.
- Donner A, Eliasziw M (1992). A goodness-of-fit approach to inference procedures for the kappa statistic. *Statistics in Medicine* 11:1511–1519.
- Landis JR, Koch GG (1977). The measurement of observer agreement for categorical data. *Biometrics* 33:159–174.
- Rotondi MA, Donner A (2012). A confidence interval approach to sample size estimation for interobserver agreement studies with multiple raters and outcomes. *Journal of Clinical Epidemiology* 65:778–784.
