# Supplementary materials: GenAI copilots in chemical, process, and manufacturing engineering

[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-data)

Companion archive of the article

> **Generative AI Copilots in Chemical, Process, and Manufacturing Engineering: A Structured Review of Validation Practices and the Case for Shared, Domain-Specific Benchmarks**
> Icaro Augusto Maccari Zelioli, Nicolas Spogis, Adriano Pinto Mariano
> University of Campinas (UNICAMP) and AI4Tech, Brazil

The article is a structured (SALSA) review of 158 records on generative-AI copilots for chemical, process, and manufacturing plants, and of how those systems are evaluated. This repository holds what a reader needs to **inspect, audit, and re-run** the review: the corpus, the screening criteria, every search run, the pipeline code, and the scripts that rebuild the reference list.

## At a glance

| | |
|---|---|
| Records retrieved and eligible for topical screening | 7,850 |
| Final corpus | **158 records** (81 human-adjudicated + 77 unique full-text re-screen passes) |
| Records with the six-dimension classification | 81 |
| Search runs logged | 791 |
| Period covered | 2023 to August 2026 |
| Sources | Crossref, OpenAlex, citation snowballing (no Scopus or Web of Science; reasons in the article, Section 2.2) |

## What is where

| Path | Content | Licence |
|---|---|---|
| [`supplementary/S1_PRISMA_2020_checklist.md`](supplementary/S1_PRISMA_2020_checklist.md) | PRISMA 2020 checklist mapped to the manuscript, with an honest status per item. Supplied for transparency, **not** a compliance claim. | CC BY 4.0 |
| [`supplementary/S2_corpus_158_records.csv`](supplementary/S2_corpus_158_records.csv) | The corpus: title, authors, venue, DOI, year, preprint flag, corpus half, Domain Gate flag, whether cited individually, and the six classification dimensions for the 81 human-adjudicated records. | CC BY 4.0 |
| [`supplementary/S3_screening/`](supplementary/S3_screening) | Screening criteria v1 and v2, eligibility rules, classification schema, re-screen design, sampling and agreement analysis, and the methodology decision journal. | CC BY 4.0 |
| [`supplementary/S4_search/`](supplementary/S4_search) | Query families and strings, every search run (`search_runs.csv`), the snowballing log, the saturation analysis, and the sentinel set. | CC BY 4.0 |
| [`supplementary/S5_code/`](supplementary/S5_code) | Source code of the retrieval, screening, and analysis pipeline, tests, lock file, and `scripts/` (reference-list rebuild, S2 builder, Figures 9 and 12). | MIT |
| [`cas-refs.bib`](cas-refs.bib), [`cas-refs-noncorpus.bib`](cas-refs-noncorpus.bib) | The article's reference lists: 158 corpus records, each checked against Crossref or DataCite by DOI, and 17 other references (methodology, source coverage, open simulators, benchmarks). | CC BY 4.0 |

[`supplementary/README.md`](supplementary/README.md) describes each file in more detail and lists the regeneration commands.

## Quick start

Explore the corpus:

```python
import pandas as pd

corpus = pd.read_csv("supplementary/S2_corpus_158_records.csv")
print(len(corpus))                                  # 158
print(corpus["corpus_half"].value_counts())          # 81 human-adjudicated, 77 full-text re-screen
print(corpus.query("preprint == 'yes'")[["citekey", "doi"]])   # the 5 arXiv preprints
```

Regenerate the two figures that depend only on counts (needs `matplotlib`; run from the repository root):

```bash
python supplementary/S5_code/scripts/figures/build_figure_9_screening_synthesis_flow.py
python supplementary/S5_code/scripts/figures/build_figure_12_applications_map.py
```

The figures are written to `figs/`.

## What is not included, and why

- **Abstracts and full texts.** Publisher copyright. Every record is identified by its DOI.
- **The screening database** (`review.db`, about 555 MB). It holds publisher abstracts. Every number derived from it is reproduced in S2 to S4. Because of this, `rebuild_corpus_bib.py` and `build_corpus_spreadsheet.py` cannot be re-run from this repository alone; they are provided so the procedure can be audited.
- **Credentials.** The pipeline reads storage and API settings from a local `.env`, which is git-ignored and not part of this archive.
- **The per-batch instruction wrapper** used with the model in each AI-assisted screening pass was not archived verbatim. The written criteria given to the model are in S3.

## How the review was screened (one paragraph)

Candidate records were retrieved through open APIs and citation snowballing, then filtered on date and peer-review status (7,850 records). An AI-assisted pass labelled every record, a single blinded human reviewer adjudicated a sample, and the measured precision of the first criterion was low (19%). The criterion was rewritten as two gates (Artifact Gate and Domain Gate), applied to a census-plus-sample frame, and the passes were read in full text. The final corpus merges the human-adjudicated half with the full-text re-screen half and removes duplicates. The article reports this as a **structured review**, not a systematic one: it was not registered, screening and classification used a single human reviewer, and the high-confidence-exclude stratum was sampled, not fully re-screened.

## Citing

Please cite the article and, for the data and code, this archive. Metadata for both is in [`CITATION.cff`](CITATION.cff). A Zenodo DOI will be added here once the archive is published.

## Licences

- Code (`supplementary/S5_code/`): [MIT](LICENSE).
- Data, documentation, and reference lists (everything else): [CC BY 4.0](LICENSE-data).

## Contact

Icaro Augusto Maccari Zelioli (corresponding author), Laboratory of Optimization, Design and Advanced Control, School of Chemical Engineering, University of Campinas. i.augustomz@gmail.com
