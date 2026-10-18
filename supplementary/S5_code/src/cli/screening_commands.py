"""`review-cli screen ...` -- Phase 9 D0->D1 eligibility, Phase 10 AI screening
(via a Claude Code subagent, not an API key)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table as RichTable

from review_pipeline.src.common.config import OUTPUTS_DIR
from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.screening.ai_screening import (
    export_batch,
    ingest_decisions,
    load_decisions,
    write_export,
)
from review_pipeline.src.screening.eligibility import (
    add_documented_exception,
    recompute_eligibility,
)
from review_pipeline.src.screening.rescreen import (
    CENSUS_STRATA,
    DEFAULT_HIGH_EXCLUDE_SAMPLE,
    DEFAULT_SEED,
    V2_PROMPT_VERSION,
    build_frame,
    export_rescreen_batch,
    frame_progress,
    read_manifest,
    write_manifest,
)

screening_app = typer.Typer(help="D0-D3 eligibility layers (Phase 9) and screening (Phase 10-13, later).")
console = Console()

DEFAULT_AI_EXPORT_PATH = OUTPUTS_DIR / "screening" / "ai_export.json"
DEFAULT_RESCREEN_MANIFEST = OUTPUTS_DIR / "rescreen_manifest.json"
DEFAULT_RESCREEN_EXPORT_PATH = OUTPUTS_DIR / "screening" / "rescreen_export.json"


@screening_app.command("recompute-eligibility")
def recompute() -> None:
    """Recompute D1 membership and core/contextual tags for every record."""
    engine = get_engine()
    summary = recompute_eligibility(engine)

    table = RichTable(title="Phase 9: D0 -> D1 eligibility")
    table.add_column("Metric")
    table.add_column("Count")
    table.add_row("Records considered (D0)", str(summary.records_considered))
    table.add_row("D1 eligible", str(summary.d1_eligible))
    table.add_row("  core", str(summary.core_count))
    table.add_row("  contextual", str(summary.contextual_count))
    for reason, count in sorted(summary.exclusion_reason_counts.items()):
        table.add_row(f"excluded: {reason}", str(count))
    console.print(table)


@screening_app.command("add-exception")
def add_exception(
    record_id: str = typer.Option(..., "--record-id"),
    rule_text: str = typer.Option(..., "--rule-text", help="The rule being invoked, e.g. 'foundational pre-2023 work'."),
    justification: str = typer.Option(..., "--justification"),
    approved_by: str = typer.Option(..., "--approved-by", help="Human reviewer name/id approving this exception."),
) -> None:
    """Register a human-approved D1 exception for one record (e.g. a
    foundational pre-2023 paper). Run recompute-eligibility afterward to apply it."""
    engine = get_engine()
    exception_id = add_documented_exception(engine, record_id, rule_text, justification, approved_by)
    console.print(f"[green]Recorded exception {exception_id} for record {record_id}.[/green]")
    console.print("[dim]Run `review-cli screen recompute-eligibility` to apply it.[/dim]")


@screening_app.command("ai-export")
def ai_export(
    limit: int | None = typer.Option(None, "--limit", help="Export at most N pending records."),
    out: str = typer.Option(str(DEFAULT_AI_EXPORT_PATH), "--out"),
) -> None:
    """Export D1-eligible, not-yet-AI-screened records for a subagent to judge
    against config/screening_criteria.yaml. Writes title/abstract/metadata only
    -- no AI labels exist yet to leak, this batch IS the input to Phase 10."""
    engine = get_engine()
    batch = export_batch(engine, limit=limit)
    write_export(batch, Path(out))
    console.print(f"[green]Exported {len(batch)} record(s) to {out}[/green]")
    if not batch:
        console.print("[dim]Nothing pending -- either D1 is empty or every D1 record already has an AI decision.[/dim]")


@screening_app.command("ai-ingest")
def ai_ingest(
    path: str = typer.Argument(..., help="JSON file of screening decisions (see ScreeningDecision schema)."),
    coder: str = typer.Option(..., "--coder", help="Model/agent identifier, e.g. 'claude-sonnet-5'."),
    prompt_version: str = typer.Option("v1", "--prompt-version"),
) -> None:
    """Ingest a subagent's screening decisions. Only ever writes decision_type='ai'
    rows -- never final_screening_decision (see ai_screening.py docstring)."""
    engine = get_engine()
    load_result = load_decisions(Path(path))
    if load_result.errors:
        console.print(f"[yellow]{len(load_result.errors)} entr(ies) failed validation and were NOT ingested:[/yellow]")
        for err in load_result.errors:
            console.print(f"  [red]{err}[/red]")

    summary = ingest_decisions(engine, load_result.decisions, coder=coder, prompt_version=prompt_version)

    table = RichTable(title="Phase 10: AI screening ingest")
    table.add_column("Metric")
    table.add_column("Count")
    table.add_row("Ingested", str(summary.ingested))
    for decision, count in sorted(summary.decision_counts.items()):
        table.add_row(f"  {decision}", str(count))
    table.add_row("Skipped (unknown record_id)", str(len(summary.skipped_unknown_record_id)))
    console.print(table)
    if summary.skipped_unknown_record_id:
        console.print(f"[yellow]Unknown record_ids: {summary.skipped_unknown_record_id}[/yellow]")


@screening_app.command("rescreen-frame")
def rescreen_frame(
    high_exclude_sample: int = typer.Option(
        DEFAULT_HIGH_EXCLUDE_SAMPLE,
        "--high-exclude-sample",
        help="Records to draw from ai_exclude_high, the one stratum too large to census.",
    ),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    out: str = typer.Option(str(DEFAULT_RESCREEN_MANIFEST), "--out"),
) -> None:
    """Build the Phase 10b re-screen frame: census the three dense v1 strata,
    probability-sample the sparse one. Deterministic given --seed; safe to
    re-run, it rebuilds from the v1 decisions rather than appending."""
    engine = get_engine()
    frame = build_frame(engine, high_exclude_sample=high_exclude_sample, seed=seed)
    write_manifest(frame, Path(out))

    table = RichTable(title=f"Phase 10b re-screen frame (seed {frame.seed})")
    for column in ("Stratum", "Mode", "Population N", "Frame n", "Weight"):
        table.add_column(column)
    for stratum, (taken) in frame.frame_counts.items():
        mode = "census" if stratum in CENSUS_STRATA else "sample"
        weight = frame.weights.get(stratum, 1.0)
        table.add_row(stratum, mode, str(frame.population_counts[stratum]), str(taken), f"{weight:.2f}")
    table.add_row("[bold]total[/bold]", "", str(sum(frame.population_counts.values())), f"[bold]{len(frame.members)}[/bold]", "")
    console.print(table)
    console.print(f"[green]Wrote {out}[/green]")


@screening_app.command("rescreen-export")
def rescreen_export(
    limit: int | None = typer.Option(None, "--limit", help="Export at most N pending frame members."),
    manifest: str = typer.Option(str(DEFAULT_RESCREEN_MANIFEST), "--manifest"),
    out: str = typer.Option(str(DEFAULT_RESCREEN_EXPORT_PATH), "--out"),
) -> None:
    """Export the next batch of frame members that have no v2 decision yet.
    Exports title/abstract/metadata only -- never the v1 label, so the v2 pass
    screens the record rather than ratifying a decision it can see."""
    engine = get_engine()
    frame = read_manifest(Path(manifest))
    batch = export_rescreen_batch(engine, frame, limit=limit)
    write_export(batch, Path(out))

    progress = frame_progress(engine, frame)
    console.print(f"[green]Exported {len(batch)} record(s) to {out}[/green]")
    console.print(f"[dim]Frame progress: {progress.done}/{progress.total} re-screened under {V2_PROMPT_VERSION}.[/dim]")
    if not batch:
        console.print("[dim]Nothing pending -- every frame member already has a v2 decision.[/dim]")


@screening_app.command("rescreen-progress")
def rescreen_progress(
    manifest: str = typer.Option(str(DEFAULT_RESCREEN_MANIFEST), "--manifest"),
) -> None:
    """Per-stratum progress of the Phase 10b re-screen."""
    engine = get_engine()
    progress = frame_progress(engine, read_manifest(Path(manifest)))

    table = RichTable(title=f"Phase 10b progress ({V2_PROMPT_VERSION})")
    for column in ("Stratum", "Re-screened", "Frame n", "%"):
        table.add_column(column)
    for stratum, (done, total) in progress.per_stratum.items():
        table.add_row(stratum, str(done), str(total), f"{100 * done / total:.0f}%" if total else "-")
    overall = f"{100 * progress.done / progress.total:.0f}%" if progress.total else "-"
    table.add_row("[bold]total[/bold]", f"[bold]{progress.done}[/bold]", str(progress.total), f"[bold]{overall}[/bold]")
    console.print(table)
