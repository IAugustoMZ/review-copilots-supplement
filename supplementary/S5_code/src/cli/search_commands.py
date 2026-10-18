"""`review-cli search ...` -- CE1-14 family search execution (Phase 4)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table as RichTable

from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.search.ce_search import build_default_connectors, run_ce_family
from review_pipeline.src.search.query_builder import load_search_queries_config

search_app = typer.Typer(help="Run CE1-14 chemical/process-engineering search families.")
console = Console()


@search_app.command("families")
def list_families() -> None:
    """List configured CE1-14 query families."""
    config = load_search_queries_config()
    table = RichTable(title="CE1-14 query families")
    table.add_column("Key")
    table.add_column("Theme")
    table.add_column("Theme terms")
    for key, family in config.families.items():
        table.add_row(key, family.theme_label, ", ".join(family.theme_block))
    console.print(table)


@search_app.command("run")
def run(
    family: str = typer.Option(..., "--family", help="CE family key, e.g. CE6"),
    iteration: int = typer.Option(1, "--iteration", help="Discovery-iteration number, for saturation tracking."),
) -> None:
    """Run one CE family across all configured connectors and persist results."""
    config = load_search_queries_config()
    if family not in config.families:
        console.print(f"[red]Unknown family '{family}'. Known families: {', '.join(config.families)}[/red]")
        raise typer.Exit(code=1)

    engine = get_engine()
    try:
        connectors = build_default_connectors(config)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    results = run_ce_family(engine, family, connectors=connectors, config=config, iteration=iteration)

    table = RichTable(title=f"{family}: {config.families[family].theme_label}")
    table.add_column("Connector")
    table.add_column("Status")
    table.add_column("Records (post-filter)")
    for _search_run_id, connector_name, response in results:
        style = "green" if response.status == "ok" else ("yellow" if response.status == "zero_results" else "red")
        table.add_row(connector_name, f"[{style}]{response.status}[/{style}]", str(len(response.records)))
        if response.error_detail:
            console.print(f"[red]{connector_name}: {response.error_detail}[/red]")
    console.print(table)
