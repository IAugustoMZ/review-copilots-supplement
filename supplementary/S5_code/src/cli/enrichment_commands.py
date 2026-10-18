"""`review-cli enrich run` -- Phase 7 metadata enrichment with field-level provenance."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table as RichTable

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.enrichment.enricher import enrich_records
from review_pipeline.src.enrichment.fetchers import build_default_fetchers

enrich_app = typer.Typer(help="Metadata enrichment: fetch fresh fields per DOI, merge by source priority.")
console = Console()


@enrich_app.command("run")
def run(
    limit: int | None = typer.Option(None, "--limit", help="Enrich at most N records (default: all with a DOI)."),
) -> None:
    """Enrich every canonical record with a DOI from Crossref/OpenAlex/Semantic Scholar/Unpaywall."""
    engine = get_engine()
    try:
        fetchers = build_default_fetchers()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    run_logger = RunLogger(component="enrichment", run_id=new_id())
    summary = enrich_records(engine, fetchers, run_logger, limit=limit)

    table = RichTable(title="Phase 7: enrichment summary")
    table.add_column("Metric")
    table.add_column("Count")
    table.add_row("Records considered", str(summary.records_considered))
    table.add_row("Skipped (no DOI)", str(summary.records_skipped_no_doi))
    table.add_row("Updated", str(summary.records_updated))
    table.add_row("Unchanged (already best-source)", str(summary.records_unchanged))
    console.print(table)
    console.print(f"[dim]Log: {run_logger.path}[/dim]")

    failed = [o for o in summary.outcomes if o.sources_failed]
    if failed:
        console.print(f"[yellow]{len(failed)} record(s) had at least one source fetch fail; see log for detail.[/yellow]")
