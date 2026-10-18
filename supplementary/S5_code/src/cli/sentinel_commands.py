"""`review-cli search sentinel ...` -- Phase 6 known-item retrieval check."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table as RichTable

from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.search.sentinel import (
    check_sentinels,
    known_item_retrieval_rate,
    load_sentinels_to_db,
)

sentinel_app = typer.Typer(help="Known-item / sentinel-set retrieval check (debugs the search strategy).")
console = Console()


@sentinel_app.command("load")
def load() -> None:
    """(Re)load config/sentinel_set.yaml into sentinel_records."""
    engine = get_engine()
    count = load_sentinels_to_db(engine)
    console.print(f"[green]Loaded {count} sentinel(s) from config/sentinel_set.yaml.[/green]")


@sentinel_app.command("check")
def check() -> None:
    """Check which sentinels the current corpus has actually retrieved."""
    engine = get_engine()
    results = check_sentinels(engine)
    if not results:
        console.print("[yellow]No sentinels loaded. Run `review-cli search sentinel load` first.[/yellow]")
        raise typer.Exit(code=1)

    table = RichTable(title="Sentinel retrieval check")
    table.add_column("Title")
    table.add_column("Matched")
    table.add_column("Generic")
    table.add_column("CE search")
    table.add_column("Venue search")
    table.add_column("Sources")
    for r in results:
        style = "green" if r.matched else "red"
        table.add_row(
            r.title[:60],
            f"[{style}]{'yes' if r.matched else 'MISS'}[/{style}]",
            "yes" if r.retrieved_generic else "-",
            "yes" if r.retrieved_ce_search else "-",
            "yes" if r.retrieved_venue_search else "-",
            ", ".join(r.retrieval_sources) or "-",
        )
    console.print(table)

    rate = known_item_retrieval_rate(results)
    misses = [r for r in results if not r.matched]
    console.print(f"\n[bold]Known-item retrieval rate: {rate:.0%}[/bold] ({len(results) - len(misses)}/{len(results)})")
    if misses:
        console.print(
            f"[yellow]{len(misses)} miss(es). Document root cause in "
            f"docs/sentinel_misses.md before treating the search strategy as validated.[/yellow]"
        )
