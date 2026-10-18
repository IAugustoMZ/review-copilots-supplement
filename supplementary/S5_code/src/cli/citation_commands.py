"""`review-cli search snowball ...` -- Phase 19 forward/backward citation-graph
snowballing from the D1-eligible corpus."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table as RichTable

from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.search.citation_trace import run_citation_trace

snowball_app = typer.Typer(help="Phase 19: forward/backward citation-graph snowballing from the D1-eligible corpus.")
console = Console()


@snowball_app.command("run")
def run(
    direction: str = typer.Option("both", "--direction", help="backward|forward|both"),
    iteration: int = typer.Option(1, "--iteration", help="Discovery-iteration number, for saturation tracking."),
    min_codiscovery_degree: int = typer.Option(
        1,
        "--min-codiscovery-degree",
        help="Trace only seeds returned by at least this many distinct search runs (citation-graph hubs). 1 = whole D1 corpus.",
    ),
    skip_already_traced: bool = typer.Option(
        False,
        "--skip-already-traced",
        help="Drop seeds a previous round already traced -- re-tracing them re-ingests known records for no new coverage.",
    ),
) -> None:
    """Trace citations from every D1-eligible seed and persist candidates as corpus_type='snowball'."""
    engine = get_engine()
    try:
        summary = run_citation_trace(
            engine,
            direction=direction,
            iteration=iteration,
            min_codiscovery_degree=min_codiscovery_degree,
            skip_already_traced=skip_already_traced,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = RichTable(title=f"Citation snowball ({direction}) -- iteration {iteration}")
    table.add_column("Seed")
    table.add_column("Connector")
    table.add_column("Direction")
    table.add_column("Status")
    table.add_column("Candidates")
    for o in summary.outcomes:
        style = "green" if o.status == "ok" else ("yellow" if o.status == "zero_results" else "red")
        table.add_row(o.title[:60], o.connector, o.direction, f"[{style}]{o.status}[/{style}]", str(o.candidate_count))
    console.print(table)
    console.print(
        f"[bold]{summary.total_candidates} total candidate(s)[/bold] from {summary.seeds_considered} seed(s) "
        f"({summary.seeds_skipped_no_doi} skipped, no DOI)."
    )
