"""Search-saturation tracking: computes the cumulative-unique-papers-vs-
iteration curve and detects the elbow where additional search iterations
stop contributing a substantial number of new unique papers. Operates only
on canonical (post-dedup) records' earliest_seen_iteration -- a record found
by iteration 3 that turns out to be a duplicate of an iteration-1 record must
count toward iteration 1's yield, not iteration 3's, or the curve would
overstate how much each new iteration actually contributed. See
src/deduplication/dedup.py::_backfill_earliest_seen_iteration, which computes
that field, and src/db/migrations/004_iteration_tracking.sql.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, select

from review_pipeline.src.db.tables import reflect_table


@dataclass
class IterationStat:
    iteration: int
    label: str
    new_unique_records: int
    cumulative_unique_records: int
    new_d1_eligible: int
    cumulative_d1_eligible: int
    marginal_growth_pct: float  # new_unique_records / cumulative BEFORE this iteration, *100


def compute_saturation_curve(engine: Engine) -> list[IterationStat]:
    records_table = reflect_table(engine, "records")
    iterations_table = reflect_table(engine, "search_iterations")

    with engine.connect() as conn:
        canonical_rows = [
            dict(r)
            for r in conn.execute(
                select(
                    records_table.c.record_id,
                    records_table.c.earliest_seen_iteration,
                    records_table.c.d1,
                ).where(records_table.c.canonical_record_id.is_(None))
            ).mappings()
        ]
        labels = {
            row["iteration_number"]: row["label"]
            for row in conn.execute(
                select(iterations_table.c.iteration_number, iterations_table.c.label)
            ).mappings()
        }

    # Union of iterations that actually contributed a new unique record AND
    # iterations that were logged (via `iterate log`) but contributed zero --
    # a zero-yield iteration must still appear as its own row with
    # new_unique_records=0, not silently vanish from the curve. A silent gap
    # is indistinguishable from "iteration never ran" and would hide exactly
    # the signal (a flat run) the elbow detector is looking for.
    known_iterations = sorted(
        {r["earliest_seen_iteration"] for r in canonical_rows if r["earliest_seen_iteration"] is not None}
        | set(labels)
    )

    curve: list[IterationStat] = []
    cumulative = 0
    cumulative_d1 = 0
    for it in known_iterations:
        new_unique = sum(1 for r in canonical_rows if r["earliest_seen_iteration"] == it)
        new_d1 = sum(1 for r in canonical_rows if r["earliest_seen_iteration"] == it and r["d1"] == 1)
        previous_cumulative = cumulative
        cumulative += new_unique
        cumulative_d1 += new_d1
        marginal_pct = (new_unique / previous_cumulative * 100) if previous_cumulative else 100.0
        curve.append(
            IterationStat(
                iteration=it,
                label=labels.get(it, f"iteration {it}"),
                new_unique_records=new_unique,
                cumulative_unique_records=cumulative,
                new_d1_eligible=new_d1,
                cumulative_d1_eligible=cumulative_d1,
                marginal_growth_pct=round(marginal_pct, 2),
            )
        )
    return curve


def detect_elbow(curve: list[IterationStat], min_marginal_pct: float = 5.0, consecutive: int = 2) -> int | None:
    """Returns the iteration number at the START of the first run of
    `consecutive` iterations that each grew cumulative unique records by less
    than `min_marginal_pct`. None if no such run exists yet (the corpus is
    still growing substantially with each iteration)."""
    run_start: int | None = None
    run_len = 0
    for stat in curve:
        if stat.marginal_growth_pct < min_marginal_pct:
            if run_len == 0:
                run_start = stat.iteration
            run_len += 1
            if run_len >= consecutive:
                return run_start
        else:
            run_len = 0
            run_start = None
    return None


def write_saturation_report(curve: list[IterationStat], elbow: int | None, path: Path) -> None:
    lines = [
        "# Search Saturation Curve",
        "",
        (
            "Cumulative *unique* (post-dedup, canonical) papers by discovery iteration. "
            "A record is attributed to the iteration it was FIRST found in, even if a "
            "later iteration's search also happened to surface it again."
        ),
        "",
        (
            "| Iteration | Label | New unique | Cumulative unique | New D1-eligible | "
            "Cumulative D1-eligible | Marginal growth |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for s in curve:
        lines.append(
            f"| {s.iteration} | {s.label} | {s.new_unique_records} | {s.cumulative_unique_records} | "
            f"{s.new_d1_eligible} | {s.cumulative_d1_eligible} | {s.marginal_growth_pct:.1f}% |"
        )
    lines.append("")
    if elbow is not None:
        lines.append(
            f"**Elbow detected at iteration {elbow}.** From this iteration onward, each additional "
            "iteration contributed less than 5% marginal growth in unique records for at least 2 "
            "consecutive iterations -- further search iterations are unlikely to substantially grow "
            "the corpus."
        )
    else:
        lines.append(
            "**No elbow detected yet.** The corpus is still growing substantially with each new "
            "iteration; additional search iterations are still likely to yield new eligible papers."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_saturation_curve(curve: list[IterationStat], elbow: int | None, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [s.iteration for s in curve]
    cumulative = [s.cumulative_unique_records for s in curve]
    cumulative_d1 = [s.cumulative_d1_eligible for s in curve]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, cumulative, marker="o", label="Cumulative unique records (D0, post-dedup)")
    ax.plot(iterations, cumulative_d1, marker="s", label="Cumulative D1-eligible records")
    if elbow is not None:
        ax.axvline(elbow, color="red", linestyle="--", alpha=0.6, label=f"Elbow (iteration {elbow})")
    ax.set_xlabel("Discovery iteration")
    ax.set_ylabel("Cumulative unique papers")
    ax.set_title("Search saturation curve")
    ax.set_xticks(iterations)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def log_iteration(
    engine: Engine,
    iteration_number: int,
    label: str,
    description: str | None,
    connectors_used: list[str],
    started_at: str,
) -> None:
    """Idempotent upsert of iteration metadata (label/description/connectors)."""
    iterations_table = reflect_table(engine, "search_iterations")
    with engine.begin() as conn:
        existing = conn.execute(
            select(iterations_table.c.iteration_number).where(
                iterations_table.c.iteration_number == iteration_number
            )
        ).first()
        if existing:
            conn.execute(
                iterations_table.update()
                .where(iterations_table.c.iteration_number == iteration_number)
                .values(
                    label=label,
                    description=description,
                    connectors_used=",".join(connectors_used),
                    started_at=started_at,
                )
            )
        else:
            conn.execute(
                iterations_table.insert().values(
                    iteration_number=iteration_number,
                    label=label,
                    description=description,
                    connectors_used=",".join(connectors_used),
                    started_at=started_at,
                )
            )
