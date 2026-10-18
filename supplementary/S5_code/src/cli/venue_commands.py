"""`review-cli search venues ...` -- Phase 5 venue-level search diagnostic."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table as RichTable

from review_pipeline.src.common.config import OUTPUTS_DIR
from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.search.ce_search import build_default_connectors
from review_pipeline.src.search.venue_search import (
    generate_venue_search_log,
    load_venue_search_config,
    run_all_venue_searches,
)

venue_app = typer.Typer(help="Venue-level search diagnostic (recall check, not an inclusion mechanism).")
console = Console()


@venue_app.command("list")
def list_venues() -> None:
    config = load_venue_search_config()
    table = RichTable(title="Configured venues")
    table.add_column("Venue")
    for v in config.venues:
        table.add_row(v.venue)
    console.print(table)


@venue_app.command("run")
def run(
    iteration: int = typer.Option(1, "--iteration", help="Discovery-iteration number, for saturation tracking."),
) -> None:
    """Run the GenAI/LLM-only query against every configured venue."""
    try:
        connectors = build_default_connectors()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    engine = get_engine()
    results = run_all_venue_searches(engine, connectors=connectors, iteration=iteration)

    total = 0
    for venue_name, connector_results in results.items():
        for _run_id, connector_name, response in connector_results:
            n = len(response.records) if response.status == "ok" else 0
            total += n
            if n:
                console.print(f"[green]{venue_name} / {connector_name}: {n} candidate(s)[/green]")

    log_md = generate_venue_search_log(results)
    log_path = OUTPUTS_DIR / "venue_search_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_md, encoding="utf-8")
    console.print(f"\n[bold]{total}[/bold] total candidate(s) across all venues. Log written to {log_path}")
