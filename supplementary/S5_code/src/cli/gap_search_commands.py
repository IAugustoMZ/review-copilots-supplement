"""`review-cli gaps ...` -- Phase 15 gap-search verification."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table as RichTable

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.logging import RunLogger
from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.search.ce_search import build_default_connectors
from review_pipeline.src.search.gap_search import (
    load_gap_claims,
    resolve_gap_search,
    run_gap_search,
)

gaps_app = typer.Typer(help="Phase 15: verify the manuscript's absence/gap claims with a fresh targeted search.")
console = Console()


@gaps_app.command("list")
def list_claims() -> None:
    claims = load_gap_claims()
    table = RichTable(title="Gap claims")
    table.add_column("gap_id")
    table.add_column("source")
    table.add_column("claim")
    for c in claims:
        table.add_row(c.gap_id, c.source_location, c.claim_text.strip()[:100])
    console.print(table)


@gaps_app.command("verify")
def verify(
    gap_id: str = typer.Option(..., "--gap-id"),
) -> None:
    """Run the claim's hand-authored queries and collect candidates for review.
    Writes a gap_searches row at status=UNCERTAIN -- run `gaps resolve`
    afterward once the candidates have actually been reviewed."""
    claims = {c.gap_id: c for c in load_gap_claims()}
    if gap_id not in claims:
        console.print(f"[red]Unknown gap_id '{gap_id}'. Known: {', '.join(claims)}[/red]")
        raise typer.Exit(code=1)

    engine = get_engine()
    try:
        connectors = build_default_connectors()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    run_logger = RunLogger(component="gap_search", run_id=new_id())
    run = run_gap_search(engine, claims[gap_id], connectors, run_logger)

    console.print(f"[bold]{gap_id}[/bold]: {claims[gap_id].claim_text.strip()}")
    console.print(f"gap_search_id: {run.gap_search_id}")
    if not run.candidates:
        console.print("[green]No candidates found -- claim not contradicted by this search.[/green]")
    else:
        table = RichTable(title=f"{len(run.candidates)} candidate(s) to review")
        table.add_column("Title")
        table.add_column("Year")
        table.add_column("Venue")
        table.add_column("Matched query")
        for c in run.candidates:
            table.add_row(c.title[:70], str(c.year or "-"), c.venue or "-", c.matched_query)
        console.print(table)
    console.print("[dim]Review candidates, then run `review-cli gaps resolve` to set the final status.[/dim]")


@gaps_app.command("resolve")
def resolve(
    gap_search_id: str = typer.Option(..., "--gap-search-id"),
    status: str = typer.Option(..., "--status", help="SUPPORTED_WITHIN_UPDATED_SEARCH|COUNTEREXAMPLE_FOUND|UNCERTAIN"),
    records_reviewed: int = typer.Option(..., "--records-reviewed"),
    counterexamples: str = typer.Option("[]", "--counterexamples", help="JSON array of counterexample records."),
) -> None:
    """Set the final status after a human/subagent has reviewed the candidates."""
    engine = get_engine()
    try:
        parsed = json.loads(counterexamples)
    except json.JSONDecodeError as exc:
        console.print(f"[red]--counterexamples must be valid JSON: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    try:
        resolve_gap_search(engine, gap_search_id, status, records_reviewed, parsed)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]{gap_search_id} -> {status}[/green]")
    if status == "SUPPORTED_WITHIN_UPDATED_SEARCH":
        console.print(
            '[dim]Manuscript wording: "No eligible study was identified in the updated search" -- '
            'never "no research exists."[/dim]'
        )
