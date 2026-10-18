"""Rebuild the corpus BibTeX file from registry-verified metadata.

Why: the previous cas-refs.bib was assembled from two sources (API records and
PDF-filename records) and, after an alphabetical re-sort, venue / DOI / year /
record_id fields drifted between entries; many PDF-derived entries also kept
only the first author. Titles were the only reliably aligned field.

Procedure (per entry of the input .bib):
  1. Match the entry to the screening database (review.db, table `records`) by
     normalized title; collect candidate DOIs (the .bib DOI and the DB DOI).
  2. Resolve every candidate DOI against Crossref (DataCite for arXiv DOIs) and
     keep it only if the registered title matches the entry title
     (SequenceMatcher ratio >= 0.85 on normalized titles).
  3. If no candidate verifies, query Crossref by title and accept the best hit
     with ratio >= 0.90.
  4. When both a published version and a preprint verify, cite the published
     version and record the preprint in the report.
  5. Build the entry from the registry record (full author list, venue,
     volume/issue/pages, year = print year if registered, else issued year).
     Entries that cannot be verified keep DB/.bib metadata and are flagged.

Outputs: the rebuilt .bib, a CSV/Markdown verification report, and a JSON
cache of registry responses (so reruns do not hit the APIs again).

Usage:
  python rebuild_corpus_bib.py --bib IN.bib --db review.db --out OUT.bib \
      --report-dir DIR --cache CACHE.json [--rename-map MAP.json] [--only KEY ...]
"""

from __future__ import annotations

import argparse
import csv
import difflib
import html
import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "lit-review-ref-check/1.0 (mailto:i.augustomz@gmail.com)"
VERIFY_RATIO = 0.85
SEARCH_RATIO = 0.90
PUBLISHED_TYPES = {"journal-article", "proceedings-article", "book-chapter", "book", "monograph"}

# Words that must keep their capitalization under the style's sentence casing.
PROPER_NOUNS = {
    "Texas", "Mexico", "Thailand", "Germany", "Aspen", "European", "Pharmacopoeia", "Brayton", "Tennessee",
    "Eastman", "Python", "Modelica", "OpenModelica", "Bayesian", "Markov", "Lyapunov",
    "Shapley", "Fourier", "Kalman", "Gaussian", "Chinese", "English", "French",
    "GitHub", "Azure", "Claude", "Llama", "LLaMA", "Aspen", "HYSYS", "Offshore",
}


# --------------------------------------------------------------------------- bib io

def parse_bib(text: str) -> list[dict]:
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        typ, key, body = m.groups()
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S):
            fields[fm.group(1).lower()] = " ".join(fm.group(2).split())
        entries.append({"type": typ.lower(), "key": key.strip(), "fields": fields})
    return entries


def norm(s: str | None) -> str:
    s = html.unescape(s or "").replace("\\&", "&")
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\\[a-zA-Z]+|[{}]", "", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def ratio(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------------------------------- registries

class Registry:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def save(self):
        self.cache_path.write_text(json.dumps(self.cache, indent=1))

    def _get(self, url: str):
        if url in self.cache:
            return self.cache[url]
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            data = json.load(urllib.request.urlopen(req, timeout=30))
        except Exception as exc:  # 404 = unregistered DOI; keep it cached as a miss
            data = {"_error": str(exc)}
        self.cache[url] = data
        time.sleep(0.6)
        return data

    def doi(self, doi: str) -> dict | None:
        doi = doi.strip()
        if doi.lower().startswith("10.48550/"):
            d = self._get("https://api.datacite.org/dois/" + urllib.parse.quote(doi.lower(), safe="/"))
            return datacite_to_record(d) if "data" in d else None
        d = self._get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/()"))
        return crossref_to_record(d["message"]) if "message" in d else None

    def search(self, title: str) -> list[dict]:
        q = urllib.parse.quote(re.sub(r"[{}\\]", "", title))
        d = self._get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=5")
        items = d.get("message", {}).get("items", []) if "message" in d else []
        return [crossref_to_record(i) for i in items]


def _year(parts) -> int | None:
    try:
        return int(parts["date-parts"][0][0])
    except Exception:
        return None


def crossref_to_record(m: dict) -> dict:
    authors = []
    for a in m.get("author", []):
        fam, giv = a.get("family", ""), a.get("given", "")
        if fam and giv and re.fullmatch(r"[A-Z]\.?", fam):
            fam, giv = giv, fam.rstrip(".") + "."  # registry put the initial in `family`
        if fam:
            authors.append((fam, giv))
        elif a.get("name"):
            authors.append((a["name"], ""))
    issued = _year(m.get("issued", {}))
    printed = _year(m.get("published-print", {})) or _year(m.get("journal-issue", {}).get("published-print", {}))
    ct = m.get("container-title") or []
    series = ""
    if m.get("type") == "book-chapter" and len(ct) > 1:
        series, ct = ct[0], ct[1:]
    return {
        "series": series,
        "source": "crossref",
        "doi": m.get("DOI", ""),
        "title": (m.get("title") or [""])[0],
        "authors": authors,
        "container": ct[0] if ct else "",
        "volume": m.get("volume", ""),
        "issue": m.get("issue", ""),
        "pages": m.get("page", "") or m.get("article-number", ""),
        "year": issued or printed,
        "year_issued": issued,
        "year_print": printed,
        "type": m.get("type", ""),
        "publisher": m.get("publisher", ""),
        "institution": (m.get("institution") or [{}])[0].get("name", "") if m.get("institution") else "",
        "group_title": m.get("group-title", ""),
    }


def datacite_to_record(d: dict) -> dict:
    a = d["data"]["attributes"]
    authors = []
    for c in a.get("creators", []):
        if c.get("familyName"):
            authors.append((c["familyName"], c.get("givenName", "")))
        elif c.get("name"):
            fam, _, giv = c["name"].partition(", ")
            authors.append((fam, giv))
    arxiv_id = a["doi"].split("arxiv.", 1)[-1]
    return {
        "source": "datacite",
        "doi": "10.48550/arXiv." + arxiv_id,
        "title": a["titles"][0]["title"],
        "authors": authors,
        "container": "arXiv",
        "volume": "", "issue": "", "pages": "",
        "year": a.get("publicationYear"),
        "year_issued": a.get("publicationYear"),
        "year_print": None,
        "type": "posted-content",
        "publisher": "arXiv",
        "institution": "",
        "group_title": "",
        "arxiv_id": arxiv_id,
    }


# --------------------------------------------------------------------------- latex text

UNICODE_FIX = {
    "‐": "-", "‑": "-", "‒": "-", "–": "--", "—": "---",
    "‘": "`", "’": "'", "“": "``", "”": "''", " ": " ",
    " ": " ", " ": " ", "−": "-", "­": "",
}


def latex_text(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"</?(scp|i|b|em|sup|sub|mml:[a-z]+|jats:[a-z]+)[^>]*>", "", s)
    for k, v in UNICODE_FIX.items():
        s = s.replace(k, v)
    s = " ".join(s.split())
    s = re.sub(r"(?<!\\)([&%#_$])", r"\\\1", s)
    s = re.sub(r"P ?\\& ?(IDs?)\b", r"P\\&\1", s)  # registry spacing: "P & ID" / "P& IDs" -> "P&ID"
    return s


def _needs_braces(core: str) -> bool:
    return (
        bool(re.search(r"[A-Z].*[A-Z]", core))            # LLM, HAZOP, GraphRAG, ChatGPT
        or bool(re.search(r"^[a-z]+[A-Z]", core))         # iPSE
        or bool(re.search(r"[A-Z]\d|\d[A-Z]", core))      # CO2, 3D, GPT4
        or core in PROPER_NOUNS
    )


def protect_title(s: str) -> str:
    """Brace-protect tokens the style's sentence casing would wrongly lower-case.

    Hyphenated compounds are judged part by part, so "LLM-Assisted" becomes
    "{LLM}-Assisted" (rendered "LLM-assisted") and "Multi-Agent" is left alone.
    """
    s = re.sub(r"\bIndustry (\d\.0)", r"{Industry \1}", s)
    out = []
    for tok in re.split(r"(\s+)", s):
        if not tok.strip() or tok.startswith("{"):
            out.append(tok)
            continue
        parts = re.split(r"(-|/)", tok)
        rebuilt = []
        for part in parts:
            m = re.match(r"^([^A-Za-z0-9\\]*)(.*?)([^A-Za-z0-9}]*)$", part)
            lead, core, trail = m.groups() if m else ("", part, "")
            rebuilt.append(lead + ("{" + core + "}" if core and _needs_braces(core.replace("\\&", "")) else core) + trail)
        out.append("".join(rebuilt))
    return "".join(out)


def bib_name(family: str, given: str) -> str:
    family, given = latex_text(family), latex_text(given)
    if not given:
        return "{" + family + "}"
    if family.isupper() and len(family) > 3:
        family = family.title()
    return f"{family}, {given}"


# --------------------------------------------------------------------------- main

def build_index(con) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {}
    for rid, doi, yr, ven, pt, title in con.execute(
        "select record_id, doi, year, venue, publication_type, title from records"
    ):
        idx.setdefault(norm(title)[:60], []).append(
            {"record_id": rid, "doi": doi, "year": yr, "venue": ven, "ptype": pt, "title": title}
        )
    return idx


def db_authors(con, record_id: str) -> list[tuple[str, str]]:
    rows = con.execute(
        "select a.full_name, a.surname from record_authors ra join authors a on a.author_id = ra.author_id "
        "where ra.record_id = ? order by ra.position",
        (record_id,),
    ).fetchall()
    out = []
    for full, sur in rows:
        if sur and full and full.endswith(sur):
            out.append((sur, full[: -len(sur)].strip()))
        else:
            out.append((sur or full, "" if sur else ""))
    return out


def choose(entry, cands: list[dict]) -> tuple[dict | None, dict | None, str]:
    title = entry["fields"].get("title", "")
    ok = [c for c in cands if c and ratio(c["title"], title) >= VERIFY_RATIO]
    if not ok:
        return None, None, "no candidate verified"
    published = [c for c in ok if c["type"] in PUBLISHED_TYPES]
    preprints = [c for c in ok if c["type"] not in PUBLISHED_TYPES]
    best = (published or preprints)[0]
    pre = preprints[0] if (published and preprints) else None
    return best, pre, "verified"


def render(entry_key: str, rec: dict) -> str:
    f: list[tuple[str, str]] = []
    f.append(("title", protect_title(latex_text(rec["title"]))))
    f.append(("author", " and ".join(bib_name(a, g) for a, g in rec["authors"]) or "{Anonymous}"))
    f.append(("year", str(rec["year"] or "")))
    typ = rec["type"]
    if typ == "journal-article":
        btype = "article"
        f.append(("journal", latex_text(rec["container"])))
    elif typ in ("proceedings-article", "book-chapter"):
        btype = "inproceedings"
        f.append(("booktitle", latex_text(rec["container"])))
        if rec.get("series"):
            f.append(("series", latex_text(rec["series"])))
        if rec.get("publisher"):
            f.append(("publisher", latex_text(rec["publisher"])))
    else:  # preprints and anything unregistered as a publication
        btype = "misc"
        if rec.get("arxiv_id"):
            f.append(("howpublished", f"arXiv preprint arXiv:{rec['arxiv_id']}"))
        else:
            server = rec.get("institution") or rec.get("group_title") or rec.get("publisher") or rec.get("container")
            f.append(("howpublished", "Preprint, " + latex_text(server)))
    for k in ("volume", "issue", "pages"):
        v = rec.get(k)
        if v:
            name = "number" if k == "issue" else k
            v = re.sub(r"\b0+(\d)", r"\1", v) if re.fullmatch(r"[\d\s-]+", v) else v
            if k == "pages":
                v = re.sub(r"(?<=\d)-(?=\d)", "--", v)
            f.append((name, latex_text(v)))
    if rec.get("doi"):
        f.append(("doi", rec["doi"]))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f if v)
    return f"@{btype}{{{entry_key},\n{body}\n}}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rename-map")
    ap.add_argument("--exclude", nargs="*", default=[], help="keys moved elsewhere (e.g. non-corpus)")
    ap.add_argument("--override", help="JSON {key: DOI} manual DOI decisions after review")
    ap.add_argument("--author-fix", help="JSON {key: [[family, given], ...]} for registry name errors")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    entries = [e for e in parse_bib(Path(args.bib).read_text(encoding="utf-8")) if e["key"] not in args.exclude]
    if args.only:
        entries = [e for e in entries if e["key"] in args.only]
    renames = json.loads(Path(args.rename_map).read_text()) if args.rename_map else {}
    overrides = json.loads(Path(args.override).read_text()) if args.override else {}
    author_fix = json.loads(Path(args.author_fix).read_text()) if args.author_fix else {}
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    idx = build_index(con)
    reg = Registry(Path(args.cache))

    report, out_entries = [], []
    for e in entries:
        f = e["fields"]
        title = f.get("title", "")
        db_hits = sorted(idx.get(norm(title)[:60], []), key=lambda h: h["doi"] is None)
        db = db_hits[0] if db_hits else None
        cand_dois = []
        if e["key"] in overrides:
            cand_dois = [overrides[e["key"]]]
        else:
            for d in (f.get("doi"), db and db["doi"]):
                if d and d.lower() not in [c.lower() for c in cand_dois]:
                    cand_dois.append(d)
        cands = [reg.doi(d) for d in cand_dois]
        best, pre, status = choose(e, cands)
        via = "doi"
        if best is None and e["key"] not in overrides:
            hits = reg.search(title)
            hits = [h for h in hits if ratio(h["title"], title) >= SEARCH_RATIO]
            best, pre, status = choose(e, hits)
            via = "title-search" if best else "unverified"
        new_key = renames.get(e["key"], e["key"])
        if best and e["key"] in author_fix:
            best = dict(best, authors=[tuple(a) for a in author_fix[e["key"]]])
            status = "verified (author names corrected by hand)"
        if best:
            out_entries.append((new_key, render(new_key, best)))
        else:
            # fall back to DB authors/venue, keep entry, flag it
            rec = {
                "title": title, "authors": db_authors(con, db["record_id"]) if db else [],
                "year": (db and db["year"]) or f.get("year"), "container": (db and db["venue"]) or f.get("journal", ""),
                "type": "journal-article" if db and db["ptype"] == "journal-article" else "posted-content",
                "volume": "", "issue": "", "pages": "", "doi": (db and db["doi"]) or "",
                "publisher": "", "institution": "", "group_title": (db and db["venue"]) or "",
            }
            out_entries.append((new_key, render(new_key, rec)))
        report.append({
            "old_key": e["key"], "new_key": new_key, "status": status, "via": via,
            "old_doi": f.get("doi", ""), "db_doi": (db and db["doi"]) or "",
            "final_doi": best["doi"] if best else "",
            "type": best["type"] if best else "", "preprint_also": pre["doi"] if pre else "",
            "old_year": f.get("year", ""), "final_year": best["year"] if best else "",
            "year_issued": best.get("year_issued") if best else "", "year_print": best.get("year_print") if best else "",
            "old_venue": f.get("journal") or f.get("booktitle", ""),
            "final_venue": html.unescape(best["container"] or best.get("institution") or best.get("group_title") or "") if best else "",
            "old_n_authors": len(re.split(r"\s+and\s+", f.get("author", ""))) if f.get("author") else 0,
            "final_n_authors": len(best["authors"]) if best else "",
            "first_author": (best["authors"][0][0] if best and best["authors"] else ""),
            "title_ratio": round(ratio(best["title"], title), 3) if best else "",
            "title": title,
        })
        reg.save()

    out_entries.sort(key=lambda kv: kv[0].lower())
    header = (
        "% ==============================================================================\n"
        "% CORPUS -- 159 eligible records (see supplementary/reference_verification.md)\n"
        "% Rebuilt by supplementary/S5_code/scripts/rebuild_corpus_bib.py from registry-verified\n"
        "% metadata (Crossref / DataCite), matched to review.db by title.\n"
        "% Non-corpus references live in cas-refs-noncorpus.bib.\n"
        "% ==============================================================================\n\n"
    )
    Path(args.out).write_text(header + "\n".join(v for _, v in out_entries), encoding="utf-8")
    rd = Path(args.report_dir)
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / "reference_verification.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    print(f"entries: {len(report)}  verified: {sum(r['status'].startswith('verified') for r in report)}")


if __name__ == "__main__":
    main()
