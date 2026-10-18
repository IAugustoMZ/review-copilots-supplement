"""Build Supplementary Material S2: the 158-record corpus as a CSV.

One row per corpus record, joined from:
  - cas-refs.bib (registry-verified metadata; see rebuild_corpus_bib.py),
  - the pre-revision corpus .bib (year of the screened version, used by Table 3),
  - review.db: human screening decisions and the six-dimension classification
    of the 81 human-adjudicated records,
  - cas-sc-template.tex (whether the record is cited individually).

No abstracts or full texts are written (copyright); records are identified by DOI.

Usage (from final/):
  python supplementary/S5_code/scripts/build_corpus_spreadsheet.py \
      --db REVIEW.db --screened-bib PRE_REVISION.bib --out supplementary/S2_corpus_158_records.csv
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINAL = HERE.parents[2]  # final/ (this script lives in final/supplementary/S5_code/scripts)

# Records retained although they lie outside the Domain Gate as written (Section 2.1).
OUTSIDE_DOMAIN_GATE = {
    "Ahmadi2024Automatic": "construction accident analysis",
    "Cardenas2025Accident": "maritime and diving accident analysis",
    "Cardante2026LLMBased": "software requirements modelling",
    "Mohammed2025Effect": "computing education",
    "Shi2023ReLM": "chemical reaction prediction (chemistry as a science)",
}
DIMENSIONS = ["domain", "technology", "application", "architectural_role", "validation_focus", "evidence_type"]


def parse_bib(text: str) -> dict[str, dict]:
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        f = {k.lower(): " ".join(v.split()) for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", m.group(3) + "\n", re.S)}
        f["_type"] = m.group(1).lower()
        out[m.group(2).strip()] = f
    return out


def norm(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+|[{}]", "", s or "").replace("\\&", "&")
    return re.sub(r"[^a-z0-9]", "", s.lower())


def plain(s: str) -> str:
    s = re.sub(r"\\&", "&", s or "")
    s = re.sub(r"\{\\[\"'`^~]\{?(\w)\}?\}", r"\1", s)
    return re.sub(r"[{}]", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--screened-bib", required=True, help="pre-revision corpus .bib (year of the screened version)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bib = parse_bib((FINAL / "cas-refs.bib").read_text(encoding="utf-8"))
    old = parse_bib(Path(args.screened_bib).read_text(encoding="utf-8"))
    renames = json.loads((HERE / "rename_map.json").read_text())
    inverse = {v: k for k, v in renames.items()}
    tex = (FINAL / "cas-sc-template.tex").read_text(encoding="utf-8")
    live = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
    cited = {k.strip() for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", live) for k in m.group(1).split(",")}

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    human_inc = con.execute(
        "select r.record_id, r.title from screening_decisions s join records r on r.record_id = s.record_id "
        "where s.decision_type = 'human' and s.decision = 'include'"
    ).fetchall()
    cls: dict[str, dict[str, set]] = {}
    for rid, dim, label in con.execute("select record_id, dimension, label from classifications_human"):
        if dim in DIMENSIONS:
            labels = json.loads(label) if label and label.lstrip().startswith("[") else [label]
            cls.setdefault(rid, {}).setdefault(dim, set()).update(l for l in labels if l)

    def match(title: str) -> str | None:
        t = norm(title)
        best, score = None, 0.0
        for key, f in bib.items():
            r = difflib.SequenceMatcher(None, t, norm(f.get("title", ""))).ratio()
            if r > score:
                best, score = key, r
        return best if score >= 0.90 else None

    human_key_to_rid = {}
    for rid, title in human_inc:
        k = match(title)
        if k:
            human_key_to_rid[k] = rid
    print(f"human-included records matched to the corpus: {len(human_key_to_rid)} of {len(human_inc)}")

    rows = []
    for key in sorted(bib, key=str.lower):
        f = bib[key]
        o = old.get(inverse.get(key, key), {})
        rid = human_key_to_rid.get(key)
        row = {
            "citekey": key,
            "title": plain(f.get("title", "")),
            "authors": plain(f.get("author", "")),
            "year_cited_version": f.get("year", ""),
            "year_screened_version": o.get("year", ""),
            "venue": plain(f.get("journal") or f.get("booktitle") or f.get("howpublished", "")),
            "doi": f.get("doi", ""),
            "entry_type": f["_type"],
            "preprint": "yes" if f["_type"] == "misc" else "no",
            "corpus_half": "human-adjudicated (81)" if rid else "full-text re-screen (77)",
            "domain_gate_note": ("outside Domain Gate as written; retained: " + OUTSIDE_DOMAIN_GATE[key]) if key in OUTSIDE_DOMAIN_GATE else "",
            "cited_individually": "yes" if key in cited else "no",
        }
        for dim in DIMENSIONS:
            row[dim] = "; ".join(sorted(cls.get(rid, {}).get(dim, []))) if rid else ""
        rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    halves = sum(r["corpus_half"].startswith("human") for r in rows)
    print(f"wrote {args.out}: {len(rows)} records, {halves} human-adjudicated, "
          f"{sum(r['preprint'] == 'yes' for r in rows)} preprints, {sum(r['cited_individually'] == 'yes' for r in rows)} cited")


if __name__ == "__main__":
    main()
