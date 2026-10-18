"""`review-cli fulltext ...` -- Phase 14 OA resolution + arXiv access fallback."""

from __future__ import annotations

import os
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table as RichTable

from azure.storage.blob import StandardBlobTier

from review_pipeline.src.common.config import OUTPUTS_DIR, load_yaml
from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.fulltext.blob_sync import (
    DEFAULT_COVERAGE_REPORT_PATH as DEFAULT_BLOB_SYNC_REPORT_PATH,
)
from review_pipeline.src.fulltext.blob_sync import (
    connection_string_from_env,
    container_name_from_env,
    sync_to_blob,
    write_sync_report,
)
from review_pipeline.src.fulltext.download import (
    DOWNLOAD_STATUSES,
    get_target_record_ids,
    resolve_and_download,
    write_coverage_report,
)
from review_pipeline.src.fulltext.extract import extract_text_from_url, save_extracted_text
from review_pipeline.src.fulltext.resolve import resolve_fulltext
from review_pipeline.src.search.connectors.http_client import build_client

DEFAULT_COVERAGE_REPORT_PATH = OUTPUTS_DIR / "fulltext_coverage.md"

fulltext_app = typer.Typer(help="Phase 14: legal OA resolution + arXiv fulltext-access fallback.")
console = Console()


def _contact_email() -> str:
    connectors_cfg = load_yaml("search_queries.yaml").get("connectors", {})
    email_env = connectors_cfg.get("contact_email_env", "REVIEW_PIPELINE_CONTACT_EMAIL")
    email = os.environ.get(email_env)
    if not email:
        console.print(f"[red]Set {email_env} first.[/red]")
        raise typer.Exit(code=1)
    return email


@fulltext_app.command("resolve")
def resolve(
    record_id: str = typer.Option(..., "--record-id"),
) -> None:
    """Resolve one record's full-text access, walking OA sources in priority order."""
    engine = get_engine()
    contact_email = _contact_email()
    client = build_client(contact_email)
    summary = resolve_fulltext(engine, record_id, client, contact_email)

    table = RichTable(title=f"Fulltext resolution: {record_id}")
    table.add_column("Source")
    table.add_column("Accessible")
    table.add_column("URL")
    for a in summary.attempts:
        table.add_row(a.source, "yes" if a.accessible else "no", a.url or "-")
    console.print(table)
    console.print(f"[dim]Not implemented (no connector): {', '.join(summary.unimplemented_sources)}[/dim]")
    if summary.resolved_source:
        console.print(f"[green]Resolved via: {summary.resolved_source}[/green]")
    else:
        console.print("[yellow]No accessible full text found.[/yellow]")


@fulltext_app.command("extract")
def extract(
    record_id: str = typer.Option(..., "--record-id"),
    url: str = typer.Option(..., "--url", help="A URL already known accessible (e.g. from `fulltext resolve`)."),
) -> None:
    """Extract plain text from an already-resolved OA URL into data/verified/."""
    contact_email = _contact_email()
    client = build_client(contact_email)
    text = extract_text_from_url(client, url)
    if text is None:
        console.print("[red]Extraction failed (fetch or parse error).[/red]")
        raise typer.Exit(code=1)
    save_extracted_text(record_id, text)
    console.print(f"[green]Saved {len(text)} chars to data/verified/{record_id}.txt[/green]")


@fulltext_app.command("arxiv-review-queue")
def arxiv_review_queue() -> None:
    """List MEDIUM-confidence arXiv fulltext-surrogate candidates awaiting human review."""
    engine = get_engine()
    candidates_table = reflect_table(engine, "fulltext_surrogate_candidates")
    with engine.connect() as conn:
        rows = conn.execute(
            candidates_table.select().where(candidates_table.c.status == "pending_human_review")
        ).mappings().all()

    if not rows:
        console.print("[green]Nothing pending.[/green]")
        return

    table = RichTable(title="arXiv fulltext-surrogate review queue")
    table.add_column("record_id")
    table.add_column("arxiv_id")
    table.add_column("confidence")
    table.add_column("title")
    for r in rows:
        table.add_row(r["record_id"], r["arxiv_id"], r["match_confidence"], r["arxiv_title"][:60])
    console.print(table)


_STATUS_MARKERS = {
    "downloaded": "[green]✓[/green]",
    "oa_unverified_license": "[yellow]~[/yellow]",
    "oa_blocked_by_publisher": "[magenta]x[/magenta]",
    "no_oa_found": "[dim]-[/dim]",
    "download_failed": "[red]![/red]",
}


@fulltext_app.command("fetch-corpus")
def fetch_corpus(
    out_report: str = typer.Option(str(DEFAULT_COVERAGE_REPORT_PATH), "--out-report"),
    delay: float = typer.Option(1.0, "--delay", help="Seconds to sleep between records (politeness across many publisher domains)."),
    only_record_ids: str = typer.Option(
        None, "--only-record-ids", help="Path to a newline-separated file of record_ids to restrict to (e.g. to re-run only previous failures)."
    ),
) -> None:
    """Walk every AI-screened topical_inclusion=include record, downloading
    its PDF when a Creative-Commons-licensed or arXiv source is found
    (see src/fulltext/download.py), and writing a coverage report for
    everything else. This is the practical scope ahead of Phase 11 human
    validation finishing -- the alternative (the full D1 pool) is mostly
    not yet topically screened."""
    engine = get_engine()
    contact_email = _contact_email()
    client = build_client(contact_email)

    targets = get_target_record_ids(engine)
    if only_record_ids:
        wanted = {line.strip() for line in Path(only_record_ids).read_text().splitlines() if line.strip()}
        targets = [r for r in targets if r["record_id"] in wanted]
    console.print(f"[cyan]{len(targets)} AI-included record(s) to resolve.[/cyan]")

    outcomes = []
    for i, record in enumerate(targets, start=1):
        outcome = resolve_and_download(engine, record["record_id"], client, contact_email)
        outcomes.append(outcome)
        console.print(f"  {_STATUS_MARKERS[outcome.status]} ({i}/{len(targets)}) {outcome.record_id} -- {outcome.status}")
        if i < len(targets):
            time.sleep(delay)

    write_coverage_report(outcomes, Path(out_report))

    table = RichTable(title="Full-text PDF fetch summary")
    table.add_column("Status")
    table.add_column("Count")
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    for status in DOWNLOAD_STATUSES:
        table.add_row(status, str(counts.get(status, 0)))
    console.print(table)
    console.print(f"[green]Coverage report written to {out_report}[/green]")


@fulltext_app.command("blob-sync")
def blob_sync(
    out_report: str = typer.Option(str(DEFAULT_BLOB_SYNC_REPORT_PATH), "--out-report"),
) -> None:
    """Upload every PDF in data/fulltext_pdfs/ to Azure Blob Storage (Cool
    tier), skipping any blob that already matches the local file. Run this
    any time after `fetch-corpus` to push newly-downloaded PDFs off disk and
    out of git -- see src/fulltext/blob_sync.py."""
    connection_string = connection_string_from_env()
    if not connection_string:
        console.print("[red]Set AZURE_STORAGE_CONNECTION_STRING first -- copy .env.example to .env and fill it in.[/red]")
        raise typer.Exit(code=1)
    container_name = container_name_from_env()
    if not container_name:
        console.print("[red]Set AZURE_STORAGE_CONTAINER first -- copy .env.example to .env and fill it in.[/red]")
        raise typer.Exit(code=1)

    outcomes = sync_to_blob(connection_string, container_name, tier=StandardBlobTier.COOL)
    write_sync_report(outcomes, container_name, StandardBlobTier.COOL, Path(out_report))

    table = RichTable(title=f"Blob sync summary -- container '{container_name}' (Cool tier)")
    table.add_column("Status")
    table.add_column("Count")
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    for status in ("uploaded", "updated", "skipped", "failed"):
        table.add_row(status, str(counts.get(status, 0)))
    console.print(table)
    console.print(f"[green]Sync report written to {out_report}[/green]")
    if counts.get("failed", 0):
        raise typer.Exit(code=1)
