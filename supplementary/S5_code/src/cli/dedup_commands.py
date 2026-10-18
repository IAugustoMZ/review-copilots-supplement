"""`review-cli dedup run` -- Phase 8 deduplication and preprint/published linkage."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table as RichTable

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.deduplication.dedup import run_deduplication

dedup_app = typer.Typer(help="Merge duplicate records and preprint/published pairs; logs every merge.")
console = Console()


@dedup_app.command("run")
def run() -> None:
    """Find and merge duplicate/preprint-published record groups across the current corpus."""
    engine = get_engine()
    run_logger = RunLogger(component="deduplication", run_id=new_id())
    summary = run_deduplication(engine, run_logger)

    table = RichTable(title="Phase 8: deduplication summary")
    table.add_column("Metric")
    table.add_column("Count")
    table.add_row("Records considered", str(summary.records_considered))
    table.add_row("Duplicate groups created", str(summary.duplicate_groups_created))
    table.add_row("Records merged away", str(summary.records_merged_away))
    table.add_row("Preprint/published pairs logged", str(summary.preprint_matches_logged))
    auto_linked = sum(1 for o in summary.preprint_outcomes if o.merged)
    table.add_row("  auto-linked (high confidence)", str(auto_linked))
    table.add_row("  pending human review", str(summary.preprint_matches_logged - auto_linked))
    console.print(table)
    console.print(f"[dim]Log: {run_logger.path}[/dim]")
