# Supplementary Material S1 — PRISMA 2020 checklist

**Status of this checklist.** This review is reported as a *structured* (SALSA) review, not as a systematic review (manuscript §2.1, "Reporting"). The checklist is supplied for transparency, so that readers can see item by item what was and was not done. It is **not** a claim of PRISMA compliance.

Item descriptions below are abbreviated. The verbatim item wording is in Page et al. (2021), *BMJ* 372:n71, doi:10.1136/bmj.n71 (Table 1).

Section numbers refer to the manuscript. "Fig. 1" is the search and snowballing flow (`fig:search-flow`); "Fig. 2" is the screening and corpus-assembly flow (`fig:ai-screening`).

| Item | Topic (abbreviated) | Where reported / status |
|---|---|---|
| 1 | Identify the report as a systematic review in the title | **Not met, by design.** The title identifies the work as a structured review; §2.1 explains why it is not reported as a systematic review. |
| 2 | Abstract (PRISMA 2020 for Abstracts) | **Partly met.** The abstract states the objective, sources, corpus size, main findings, and implications; it does not follow the PRISMA for Abstracts item list. |
| 3 | Rationale | §1 |
| 4 | Objectives / questions | §1 (RQ1–RQ4); §2.1 notes that RQ4 was refined after screening began |
| 5 | Eligibility criteria; grouping for syntheses | §2.2 (domain vocabulary, Table 1 notes); §2.3 (D1: temporal scope 2023–Aug 2026, publication status); §2.4 (written criterion v1); §2.5 (two-gate criterion v2); §2.8 (grouping by six dimensions mapped to RQ1–RQ4); Table 8 (exclusive domain assignment). Full criteria: S3. |
| 6 | Information sources and dates last searched | §2.2 (Crossref, OpenAlex, Semantic Scholar — abandoned; citation snowballing); "Choice of sources" paragraph explains why Scopus and Web of Science were not used. Run dates per query: S4 `search_runs.csv`. |
| 7 | Full search strategies | Table 1 (query families CE1–CE19); full query strings and filters: S4 `search_queries.yaml` and `search_runs.csv` |
| 8 | Selection process (reviewers, independence, automation) | §2.3–§2.7; Fig. 2. AI-assisted screening (Claude Opus 5 via a Claude Code subagent) with a single, blinded human reviewer; the AI pass's precision was measured on a 500-record stratified sample (§2.4) and on a held-out check (§2.6). Criteria and batch procedure: S3. |
| 9 | Data collection process | §2.8 (six-dimension classification of the 81 human-adjudicated records by a single reviewer); Table 8 caption (evidence read as full text for 113 records, verified abstract for 25, title and venue only for 20). No investigators were contacted. |
| 10a | Outcomes sought | **Not applicable** in the meta-analytic sense. The data items are the six classification dimensions and the performance figures each record reports, quoted as reported. |
| 10b | Other variables; handling of missing information | §2.8 and S3 `classification_schema.yaml`; records read on title and venue only contribute to counts, never to a quantitative claim (Table 8 caption). |
| 11 | Study risk-of-bias assessment | **Not done.** No formal risk-of-bias tool was applied. Evidence type (Table 5) is reported as a maturity indicator, and record-level limitations are reported in the narrative where the records state them. |
| 12 | Effect measures | **Not applicable** (no quantitative synthesis). |
| 13a | Which studies enter each synthesis | §2.8; §3 (scope of each claim: 81 classified records vs. all 158) |
| 13b | Data preparation | **Not applicable.** |
| 13c | Tabulation and visual display | Tables 3–10; Figs. 3–4 |
| 13d | Synthesis methods | Narrative synthesis organised by research question (SALSA; §2.1, §2.8); no meta-analysis |
| 13e | Exploring heterogeneity | **Not applicable.** |
| 13f | Sensitivity analyses | **Not done** for the synthesis. The screening criterion itself was validated (§2.4–§2.6), including a loss probe on newly excluded records. |
| 14 | Reporting-bias assessment | **Not done.** |
| 15 | Certainty assessment | **Not done.** |
| 16a | Results of search and selection (flow diagram) | §2.3–§2.7; Figs. 1–2; Table 2 |
| 16b | Studies that seemed eligible but were excluded | **Partly met.** Counts are reported (§2.5: of 146 full texts, 22 failed and 2 could not be settled); the individual records are not listed. |
| 17 | Characteristics of included studies | S2 (all 158 records: metadata, DOI, corpus half, preprint status, Domain Gate flags, six-dimension classification of the 81); Tables 3–5 and 8 |
| 18 | Risk of bias in studies | **Not done** (see item 11). |
| 19 | Results of individual studies | §5–§8 (results quoted as each record reports them); Table 8 (strongest measured result per domain, with its baseline) |
| 20a | Characteristics of studies contributing to each synthesis | §3; opening paragraphs of §5–§8 |
| 20b–20d | Statistical syntheses, heterogeneity, sensitivity | **Not applicable** (no statistical synthesis). |
| 21 | Reporting biases | **Not done.** |
| 22 | Certainty of evidence | **Not done.** |
| 23a | Interpretation | §5–§9; Final Considerations |
| 23b | Limitations of the evidence | §3.3 (classification scoped to 81 records); §3.4 (counts are lower bounds); domain-specific limitations in §5–§8 |
| 23c | Limitations of the review process | §2.1 ("Scope", "Reporting"); §2.2 ("Choice of sources": no Boolean-precise source, no subscription-index cross-check); §2.3 (preprints); §2.6 (5,173 high-confidence excludes never re-screened) |
| 23d | Implications | §9; Final Considerations |
| 24a | Registration | **Not registered** (§2.1, "Reporting"). |
| 24b | Protocol | **No protocol was prepared in advance.** Decisions were logged as they were taken: S3 `METHODOLOGY_DECISION_JOURNAL.md`. |
| 24c | Amendments | RQ4 refined after screening began (§2.1); query families CE16–CE19 added (§2.2); screening criterion rewritten after validation (§2.5). Each is logged with its reason in S3 `METHODOLOGY_DECISION_JOURNAL.md`. |
| 25 | Support | "Funding" section |
| 26 | Competing interests | "Declaration of Competing Interest" section |
| 27 | Availability of data, code and other materials | "Data and Code Availability" section; S2–S5 |
