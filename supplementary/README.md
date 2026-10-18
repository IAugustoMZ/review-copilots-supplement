# Supplementary materials

Companion files for *Generative AI Copilots in Chemical, Process, and Manufacturing Engineering: A Structured Review of Validation Practices and the Case for Shared, Domain-Specific Benchmarks*.

| ID | File(s) | Content |
|---|---|---|
| S1 | `S1_PRISMA_2020_checklist.md` | PRISMA 2020 checklist mapped to the manuscript, with an explicit status per item. Supplied for transparency; not a compliance claim. |
| S2 | `S2_corpus_158_records.csv` | The 158-record corpus: metadata, DOI, year of cited and of screened version, corpus half, preprint flag, Domain Gate flag, whether cited individually, and the six-dimension classification of the 81 human-adjudicated records. No abstracts or full texts (copyright). |
| S3 | `S3_screening/` | Written criteria given to the model in each AI-assisted pass (`screening_criteria.yaml` = screen-v1; `screening_criteria_v2.yaml` = screen-v2), D0→D1 eligibility rules, classification schema, re-screen design and batch procedure, sampling and agreement analysis, and the methodology decision journal (amendments and their reasons). The per-batch instruction wrapper used with the model was not archived verbatim. |
| S4 | `S4_search/` | Query families and strings (`search_queries.yaml`), every search run with query text, source, timestamp and record count (`search_runs.csv`, 791 runs), search and snowballing log, saturation analysis, sentinel set. |
| S5 | `S5_code/` | Source code of the retrieval, screening and analysis pipeline (`src/`, `tests/`, `pyproject.toml`, `poetry.lock`). Credentials are not included: the pipeline reads them from a local `.env`. `S5_code/scripts/` holds the scripts that rebuild the reference list (`rebuild_corpus_bib.py`), S2 (`build_corpus_spreadsheet.py`) and Figures 9 and 12 (`figures/`), plus the JSON files of manual decisions they apply. |

## Regenerating (run from the archive root, the folder that contains `supplementary/` and `cas-refs.bib`)

```
python supplementary/S5_code/scripts/rebuild_corpus_bib.py \
    --bib <pre-revision .bib> --db <review.db> \
    --out cas-refs.bib --report-dir <report folder> --cache <cache.json> \
    --exclude ehrhardt2022benchmark Tomislav2024Design \
    --rename-map supplementary/S5_code/scripts/rename_map.json \
    --author-fix supplementary/S5_code/scripts/author_fix.json \
    --override supplementary/S5_code/scripts/doi_overrides.json
python supplementary/S5_code/scripts/build_corpus_spreadsheet.py \
    --db <review.db> --screened-bib <pre-revision .bib> --out supplementary/S2_corpus_158_records.csv
python supplementary/S5_code/scripts/figures/build_figure_9_screening_synthesis_flow.py   # needs matplotlib
python supplementary/S5_code/scripts/figures/build_figure_12_applications_map.py          # needs matplotlib
```

The screening database (`review.db`, ~555 MB) is not redistributed because it holds publisher abstracts, so the first two commands cannot be re-run from this package alone. Every number derived from the database is reproduced in S2–S4. The figure scripts need no inputs.
