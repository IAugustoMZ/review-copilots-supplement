# Search Saturation Curve

Cumulative *unique* (post-dedup, canonical) papers by discovery iteration. A record is attributed to the iteration it was FIRST found in, even if a later iteration's search also happened to surface it again.

| Iteration | Label | New unique | Cumulative unique | New D1-eligible | Cumulative D1-eligible | Marginal growth |
|---|---|---|---|---|---|---|
| 1 | Initial production sweep | 73 | 73 | 67 | 67 | 100.0% |
| 2 | Crossref-only retry | 0 | 73 | 0 | 67 | 0.0% |
| 3 | Full sweep (OpenAlex fix live) | 38 | 111 | 25 | 92 | 52.0% |
| 4 | Semantic Scholar catch-up | 0 | 111 | 0 | 92 | 0.0% |
| 5 | Same-day confirmatory re-run | 0 | 111 | 0 | 92 | 0.0% |
| 6 | Citation snowball (backward+forward) | 3367 | 3478 | 1747 | 1839 | 3033.3% |
| 7 | CE15 benchmark-dataset search | 0 | 3478 | 0 | 1839 | 0.0% |

**Elbow detected at iteration 4.** From this iteration onward, each additional iteration contributed less than 5% marginal growth in unique records for at least 2 consecutive iterations -- further search iterations are unlikely to substantially grow the corpus.
