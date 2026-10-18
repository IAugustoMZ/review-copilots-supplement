#!/usr/bin/env python3
"""Builds Figure 9 (screening, validation and corpus assembly flow).

Recreates the layout of the original figure_9_screening_synthesis_flow.png, whose
generator was not kept, with one correction: the final box previously read
"Final corpus: 159 records classified and synthesized". The corpus is 158 records
after a DOI-level check removed a duplicate, and only the 81 human-included
records carry the six-dimension classification (Table 2, Section 2.7).

All counts are those reported in Section 2 of the manuscript.

Regenerate (from final/):
    python supplementary/S5_code/scripts/figures/build_figure_9_screening_synthesis_flow.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "..", "figs", "figure_9_screening_synthesis_flow.png")

ACTION = ("#dce8f5", "#2b5d8a")      # screening / review action
CRITERION = ("#fce1c6", "#b8661f")   # criterion definition
RESULT = ("#dff0d8", "#3c7a3a")      # measured result / pool size

W, H = 2.6, 1.0                     # box size (axis units)
XS = [0.1, 3.15, 6.2]                 # column left edges
YS = [4.6, 2.9, 1.2]                 # row bottom edges

BOXES = [
    # (row, col, text, style, bold)
    (0, 0, "Eligible pool\n7,850 records", RESULT, False),
    (0, 1, "AI screening (Claude Opus 5)\n1,067 include · 6,545 exclude", ACTION, False),
    (0, 2, "Human review (blinded)\n630 reviewed: 81 include", ACTION, False),
    (1, 2, "Validating the AI pass\n19.0% precision — too loose", RESULT, False),
    (1, 1, "Criterion rewritten\ntwo required gates", CRITERION, False),
    (1, 0, "Targeted re-screen\n2,677 records → 348 include", RESULT, False),
    (2, 0, "Full-text verification\n146 checked → 122 confirmed", RESULT, False),
    (2, 1, "Honest validation\n32.0% precision (held-out)", RESULT, False),
    (2, 2, "Final corpus: 158 records\n81 human + 122 full-text,\nde-duplicated (Table 2)", RESULT, True),
]

# arrows between (row, col) cells, following the snake order of the original
ARROWS = [((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (1, 2)), ((1, 2), (1, 1)),
          ((1, 1), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 1), (2, 2))]


def centre(rc):
    r, c = rc
    return XS[c] + W / 2, YS[r] + H / 2


def edge_points(a, b):
    (ax_, ay), (bx, by) = centre(a), centre(b)
    if a[0] == b[0]:  # same row: horizontal
        s = 1 if bx > ax_ else -1
        return (ax_ + s * (W / 2 + 0.05), ay), (bx - s * (W / 2 + 0.05), by)
    return (ax_, ay - H / 2 - 0.05), (bx, by + H / 2 + 0.05)


def main():
    fig, ax = plt.subplots(figsize=(10, 6.15))
    ax.set_xlim(0, 8.9)
    ax.set_ylim(0, 6.3)
    ax.axis("off")
    ax.text(4.45, 5.95, "Screening, validating, and finalizing the corpus",
            ha="center", va="center", fontsize=15, fontweight="bold")
    for r, c, text, (fill, edge), bold in BOXES:
        ax.add_patch(FancyBboxPatch((XS[c], YS[r]), W, H, boxstyle="round,pad=0.0,rounding_size=0.08",
                                    facecolor=fill, edgecolor=edge, linewidth=2))
        ax.text(XS[c] + W / 2, YS[r] + H / 2, text, ha="center", va="center",
                fontsize=8.6 if text.count("\n") > 1 else 9.4,
                fontweight="bold" if bold else "normal", linespacing=1.35)
    for a, b in ARROWS:
        p, q = edge_points(a, b)
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=2, color="#333333"))
    for i, (label, (fill, edge)) in enumerate([("screening / review action", ACTION),
                                               ("criterion definition", CRITERION),
                                               ("measured result / pool size", RESULT)]):
        x = 1.1 + i * 2.8
        ax.add_patch(FancyBboxPatch((x, 0.25), 0.3, 0.3, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    facecolor=fill, edgecolor=edge, linewidth=2))
        ax.text(x + 0.42, 0.4, label, va="center", fontsize=9.4)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
