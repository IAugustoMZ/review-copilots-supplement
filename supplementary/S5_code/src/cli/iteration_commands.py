"""`review-cli iterate ...` -- search-saturation tracking across discovery iterations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table as RichTable

from review_pipeline.src.analysis.saturation import (
    compute_saturation_curve,
    detect_elbow,
    log_iteration,
    plot_saturation_curve,
    write_saturation_report,
)
from review_pipeline.src.common.config import OUTPUTS_DIR
from review_pipeline.src.db.engine import get_engine

iterate_app = typer.Typer(help="Track discovery iterations and the search-saturation curve.")
console = Console()


@iterate_app.command("log")
def log(
    iteration: int = typer.Option(..., "--iteration"),
    label: str = typer.Option(..., "--label"),
    description: str = typer.Option("", "--description"),
    connectors: str = typer.Option("", "--connectors", help="Comma-separated connector names used this iteration."),
) -> None:
    """Record what iteration N was (label/description/connectors used)."""
    engine = get_engine()
    log_iteration(
        engine,
        iteration_number=iteration,
        label=label,
        description=description or None,
        connectors_used=[c.strip() for c in connectors.split(",") if c.strip()],
        started_at=datetime.now(UTC).isoformat(),
    )
    console.print(f"[green]Logged iteration {iteration}: {label}[/green]")


@iterate_app.command("stats")
def stats() -> None:
    """Print the cumulative-unique-papers-vs-iteration table and the detected elbow."""
    engine = get_engine()
    curve = compute_saturation_curve(engine)
    if not curve:
        console.print("[yellow]No canonical records with earliest_seen_iteration set yet -- run dedup first.[/yellow]")
        raise typer.Exit(code=0)

    table = RichTable(title="Search saturation by iteration")
    table.add_column("Iter")
    table.add_column("Label")
    table.add_column("New unique")
    table.add_column("Cumulative unique")
    table.add_column("New D1")
    table.add_column("Cumulative D1")
    table.add_column("Marginal growth")
    for s in curve:
        table.add_row(
            str(s.iteration),
            s.label,
            str(s.new_unique_records),
            str(s.cumulative_unique_records),
            str(s.new_d1_eligible),
            str(s.cumulative_d1_eligible),
            f"{s.marginal_growth_pct:.1f}%",
        )
    console.print(table)

    elbow = detect_elbow(curve)
    if elbow is not None:
        console.print(
            f"[bold]Elbow at iteration {elbow}[/bold] -- further iterations show diminishing returns "
            "(<5% growth for 2+ consecutive iterations)."
        )
    else:
        console.print("[dim]No elbow yet -- corpus is still growing substantially.[/dim]")


@iterate_app.command("report")
def report(
    out: str = typer.Option(str(OUTPUTS_DIR / "saturation_curve.md"), "--out"),
    plot_out: str = typer.Option(str(OUTPUTS_DIR / "saturation_curve.png"), "--plot-out"),
) -> None:
    """Write the saturation report (Markdown table + elbow verdict) and the PNG plot."""
    engine = get_engine()
    curve = compute_saturation_curve(engine)
    if not curve:
        console.print("[yellow]No canonical records with earliest_seen_iteration set yet -- run dedup first.[/yellow]")
        raise typer.Exit(code=1)

    elbow = detect_elbow(curve)
    write_saturation_report(curve, elbow, Path(out))
    plot_saturation_curve(curve, elbow, Path(plot_out))
    console.print(f"[green]Report written to {out}, plot written to {plot_out}[/green]")
