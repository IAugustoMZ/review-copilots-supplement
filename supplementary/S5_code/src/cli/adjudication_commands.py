"""`review-cli adjudicate ...` -- Phase 13 adjudication."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table as RichTable

from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.validation.adjudication import (
    apply_adjudication,
    auto_finalize_agreements,
    queue_for_adjudication,
)

adjudicate_app = typer.Typer(help="Phase 13: resolve AI/human disagreement, finalize classifications.")
console = Console()


@adjudicate_app.command("queue")
def queue(
    dimension: str = typer.Option("topical_inclusion", "--dimension"),
) -> None:
    """List records needing adjudication: AI != human, or either low-confidence."""
    engine = get_engine()
    candidates = queue_for_adjudication(engine, dimension)

    if not candidates:
        console.print("[green]Nothing pending adjudication for this dimension.[/green]")
        return

    table = RichTable(title=f"Adjudication queue: {dimension}")
    table.add_column("record_id")
    table.add_column("AI")
    table.add_column("Human")
    for c in candidates:
        table.add_row(
            c.record_id,
            f"{c.ai_label} ({c.ai_confidence})",
            f"{c.human_label} ({c.human_confidence})",
        )
    console.print(table)
    console.print(f"[dim]{len(candidates)} record(s) pending.[/dim]")


@adjudicate_app.command("resolve")
def resolve(
    record_id: str = typer.Option(..., "--record-id"),
    dimension: str = typer.Option("topical_inclusion", "--dimension"),
    action: str = typer.Option(
        ..., "--action", help="accept_human|accept_ai|new_value|unresolved|request_fulltext"
    ),
    adjudicator: str = typer.Option(..., "--adjudicator"),
    reason: str = typer.Option(..., "--reason"),
    new_decision: str = typer.Option(None, "--new-decision", help="Required for action=new_value."),
    new_confidence: str = typer.Option(None, "--new-confidence", help="Required for action=new_value."),
) -> None:
    """Resolve one record's disagreement. Always inserts a new
    classifications_final row + audit_events row; nothing is updated in place."""
    engine = get_engine()
    try:
        classification_id = apply_adjudication(
            engine,
            record_id,
            dimension,
            action,  # type: ignore[arg-type]
            adjudicator=adjudicator,
            reason=reason,
            new_decision=new_decision,
            new_confidence=new_confidence,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Recorded final classification {classification_id} for {record_id}.[/green]")


@adjudicate_app.command("auto-finalize")
def auto_finalize(
    dimension: str = typer.Option("topical_inclusion", "--dimension"),
) -> None:
    """Mechanically finalize records where AI and human already agree at
    non-low confidence, so manual adjudication only has to look at real
    disagreements."""
    engine = get_engine()
    summary = auto_finalize_agreements(engine, dimension)
    console.print(f"[green]Auto-finalized {summary.finalized} record(s).[/green]")
    console.print(f"[dim]Skipped (already final): {summary.skipped_already_final}[/dim]")
