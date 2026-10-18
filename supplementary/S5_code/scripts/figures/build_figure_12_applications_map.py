#!/usr/bin/env python3
"""Builds the §7 application-domain map (figure_12).

Message: the corpus fills the region where correctness is cheap to check and the
system's output stays advisory, and empties the region where correctness needs an
engineer AND the output reaches plant equipment.

Both axes are ordinal constructs of this review, not measured scales, and the caption
must say so. Cell areas are proportional to record counts (area, never radius --
Rougier 6). Counts come from the exclusive assignment in
draft_sections/SECTION_7_RQ3_APPLICATIONS.md and sum to 158 including the 20
review/roadmap records, which carry no system and are therefore not plotted.

Figure rules applied, from data/writing_source/scientific_writingg:
  - no title inside the plot area; the title lives in the LaTeX caption  (MackFig1)
  - no gridlines, no 3-D, no chartjunk                                   (Patience 4.1; Rougier 8)
  - labels placed in-plot rather than in a legend                        (MackFig1; MackFig2 step 3)
  - greyscale-safe: fill value + outline weight carry the encoding, not hue (MackFig2)
  - >=80 % of each axis carries data                                     (MackFig1)
  - 300 dpi minimum, sized for cas-dc double-column width                (MackFig2)

Regenerate:
    python3 supplementary/S5_code/scripts/figures/build_figure_12_applications_map.py   (from final/)

Revision 2026-10-12: Design cell 30 -> 29 after removing a duplicate record
(corpus 159 -> 158; see rebuild_corpus_bib.py and the --exclude list in the S5 README).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
# output: final/figs (this script lives in final/supplementary/S5_code/scripts/figures)
FIGS = os.path.join(HERE, "..", "..", "..", "..", "figs")
PNGS = FIGS
for d in (FIGS, PNGS):
    os.makedirs(d, exist_ok=True)

# cas-dc double-column \textwidth on a4paper is ~17.8 cm = 7.0 in.
FIG_W_IN, FIG_H_IN = 7.0, 4.35
BASE_FS = 8.0

# ---------------------------------------------------------------- the cells --
# x: what is allowed to decide correctness (0 expert judgment .. 2 executable)
# y: how far the output may travel  (0 advisory text .. 2 plant actuation)
# n: records in the exclusive assignment
# "outside" places the label to the right of the block, for blocks too small to
# hold their own text.
CELLS = [
    # label,                                     x,    y,    n,  fill, outside
    ("Education and\nworkforce",                0.03, -0.14,  3,  0.93, True),
    # short form: "and documentation" fits neither inside the n=13 block nor in the
    # gap beside it (the n=28 block is 0.22 axis-units away). Table 1 and the
    # caption carry the full cell name.
    ("Knowledge\nmanagement",                    0.44,  0.26, 13,  0.88, False),
    ("Process safety and\nhazard analysis",      0.26,  0.83, 26,  0.80, False),
    ("Fault detection, diagnosis\nand maintenance", 1.30, 0.40, 28, 0.80, False),
    ("Manufacturing\nexecution",                 1.02,  1.40, 19,  0.85, False),
    ("Process control\nand operations",          1.62,  1.94, 20,  0.85, False),
    ("Design, flowsheeting\nand simulation",      1.90, 1.06, 29,  0.78, False),
]

AREA_PER_RECORD = 0.0175          # axis-units^2 per record; area, not radius
ASPECT = 1.55                     # width:height of each cell block
FS_FLOOR = 6.4                    # smallest label size before a label goes outside


def _measure(fig, ax, text, fontsize):
    """Width and height of `text` in axis data units, at `fontsize`."""
    probe = ax.text(0, 0, text, fontsize=fontsize, linespacing=1.32, alpha=0)
    bb = probe.get_window_extent(renderer=fig.canvas.get_renderer())
    probe.remove()
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return abs(x1 - x0), abs(y1 - y0)


MIN_GAP = 0.055                   # axis units; blocks closer than this read as touching


def _check_gaps(placed):
    boxes = [(lb.replace("\n", " "), x - w / 2, x + w / 2, y - h / 2, y + h / 2)
             for lb, x, y, n, w, h, fs, out in placed]
    boxes.append(("no-records region", -0.30, 0.50, 1.56, 2.40))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            dx = max(a[1] - b[2], b[1] - a[2])   # <0 means they span each other in x
            dy = max(a[3] - b[4], b[3] - a[4])
            if dx < MIN_GAP and dy < MIN_GAP:
                gap = max(dx, dy)
                verdict = "OVERLAP" if gap < 0 else f"gap {gap:.3f}"
                print(f"  TOO CLOSE: {a[0]} / {b[0]} -- {verdict}")


def main():
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))

    # limits first: _measure() converts pixels to data units, so it reads the wrong
    # scale if the axes are still on their 0--1 default when the labels are fitted
    ax.set_xlim(-0.34, 2.48)
    ax.set_ylim(-0.42, 2.46)

    # ---- the empty region, drawn first so blocks sit above it ----
    ax.add_patch(Rectangle(
        (-0.30, 1.56), 0.80, 0.84,
        facecolor="0.965", edgecolor="0.55",
        linewidth=0.9, linestyle=(0, (4, 3)), zorder=1))
    ax.text(0.10, 2.24,
            "No records",
            ha="center", va="center", fontsize=BASE_FS + 0.5,
            fontweight="bold", color="0.25", zorder=2)
    ax.text(0.10, 1.90,
            "correctness needs\nan engineer, and the\n"
            "output reaches\nplant equipment",
            ha="center", va="center", fontsize=BASE_FS - 1.4,
            color="0.35", linespacing=1.40, zorder=2)

    # ---- domain blocks ----
    # Geometry is fixed by the counts (area proportional to n), so a label either
    # fits inside its block or it does not. Measuring settles it; guessing does not.
    fig.canvas.draw()
    placed = []
    for label, x, y, n, fill, outside in CELLS:
        area = n * AREA_PER_RECORD
        h = (area / ASPECT) ** 0.5
        w = h * ASPECT
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.030",
            facecolor=str(fill), edgecolor="0.20",
            linewidth=1.05, zorder=3))

        fs = BASE_FS - 1.0
        if not outside:
            # shrink until the label and its count clear the block's inner margin,
            # then give up and move the label out
            while fs >= FS_FLOOR:
                tw, th = _measure(fig, ax, f"{label}\n$n$ = {n}", fs)
                if tw <= w - 0.085 and th <= h - 0.055:
                    break
                fs -= 0.2
            else:
                outside = True
        placed.append((label, x, y, n, w, h, fs, outside))

    for label, x, y, n, w, h, fs, outside in placed:
        if outside:
            ax.text(x + w / 2 + 0.07, y, f"{label}\n$n$ = {n}",
                    ha="left", va="center", fontsize=fs,
                    color="0.08", linespacing=1.32, zorder=4)
            continue
        # push label and count apart as the label gains lines, so a
        # multi-line name never lands on its own count
        extra = 0.034 * (label.count("\n") - 1)
        ax.text(x, y + 0.052 + extra, label, ha="center", va="center",
                fontsize=fs, color="0.08",
                linespacing=1.32, zorder=4)
        ax.text(x, y - 0.105 - extra, f"$n$ = {n}", ha="center", va="center",
                fontsize=fs + 0.4, fontweight="bold",
                color="0.08", zorder=4)

    # ---- axes as ordinal constructs ----
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Expert\njudgment", "Inherited\nlabel", "Executable"],
                       fontsize=BASE_FS - 0.4, linespacing=1.3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Advisory\ntext", "Engineering\nartefact", "Plant\nactuation"],
                       fontsize=BASE_FS - 0.4, linespacing=1.3)
    ax.set_xlabel("What is allowed to decide that an answer is correct "
                  r"$\longrightarrow$ cheaper to check",
                  fontsize=BASE_FS, labelpad=7)
    ax.set_ylabel(r"How far the output may travel $\longrightarrow$",
                  fontsize=BASE_FS, labelpad=7)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.9)
        ax.spines[side].set_color("0.35")
    ax.tick_params(length=3.2, width=0.9, color="0.35", pad=4)
    ax.grid(False)

    fig.tight_layout(pad=0.55)
    svg = os.path.join(FIGS, "figure_12_applications_map.svg")
    png = os.path.join(PNGS, "figure_12_applications_map.png")
    png_fig = os.path.join(FIGS, "figure_12_applications_map.png")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    for p in (png, png_fig):
        fig.savefig(p, format="png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    _check_gaps(placed)

    for label, x, y, n, w, h, fs, outside in placed:
        flat = label.replace("\n", " ")
        if outside:
            print(f"  label outside block: {flat} (n={n})")
        elif fs < BASE_FS - 1.0:
            print(f"  label shrunk to {fs:.1f}pt to fit: {flat} (n={n})")

    total = sum(c[3] for c in CELLS)
    print(f"wrote {svg}")
    print(f"wrote {png}")
    print(f"wrote {png_fig}")
    print(f"plotted {total} records across {len(CELLS)} cells; "
          f"+20 review/roadmap records not plotted = {total + 20} of 158")


if __name__ == "__main__":
    main()
