"""review-cli entry point.

Subcommands are wired up incrementally as each pipeline phase lands
(see docs/TECHNICAL_IMPLEMENTATION_PLAN.md, section 14 and 26). At this
scaffolding stage only `review-cli status` exists.
"""

from __future__ import annotations

import typer
from rich.console import Console

from review_pipeline.src.cli.adjudication_commands import adjudicate_app
from review_pipeline.src.cli.agreement_commands import agreement_app
from review_pipeline.src.cli.citation_commands import snowball_app
from review_pipeline.src.cli.db_commands import db_app
from review_pipeline.src.cli.dedup_commands import dedup_app
from review_pipeline.src.cli.enrichment_commands import enrich_app
from review_pipeline.src.cli.fulltext_commands import fulltext_app
from review_pipeline.src.cli.gap_search_commands import gaps_app
from review_pipeline.src.cli.human_review_commands import classify_app
from review_pipeline.src.cli.iteration_commands import iterate_app
from review_pipeline.src.cli.screening_commands import screening_app
from review_pipeline.src.cli.search_commands import search_app
from review_pipeline.src.cli.sentinel_commands import sentinel_app
from review_pipeline.src.cli.venue_commands import venue_app

app = typer.Typer(
    name="review-cli",
    help="Reproducible search, classification, and review pipeline CLI.",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")
app.add_typer(search_app, name="search")
app.add_typer(enrich_app, name="enrich")
app.add_typer(dedup_app, name="dedup")
app.add_typer(screening_app, name="screen")
app.add_typer(classify_app, name="classify")
app.add_typer(agreement_app, name="agreement")
app.add_typer(adjudicate_app, name="adjudicate")
app.add_typer(fulltext_app, name="fulltext")
app.add_typer(gaps_app, name="gaps")
app.add_typer(iterate_app, name="iterate")
search_app.add_typer(venue_app, name="venues")
search_app.add_typer(sentinel_app, name="sentinel")
search_app.add_typer(snowball_app, name="snowball")
console = Console()


@app.callback()
def _main() -> None:
    """review-cli — see subcommands below."""


@app.command()
def status() -> None:
    """Print a one-shot dashboard of pipeline state (placeholder until Phase 2 DB layer lands)."""
    console.print("[bold]review-cli[/bold] — pipeline scaffold. Database layer not yet implemented.")


if __name__ == "__main__":
    app()
