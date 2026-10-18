"""`review-cli agreement ...` -- Phase 12 AI-human agreement metrics."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table as RichTable

from review_pipeline.src.common.config import OUTPUTS_DIR
from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.validation.agreement import (
    compute_dimension_agreement,
    write_agreement_csv,
    write_agreement_report,
)

agreement_app = typer.Typer(help="AI-human agreement metrics (never 'inter-rater reliability' unqualified).")
console = Console()

DEFAULT_CSV_PATH = OUTPUTS_DIR / "validation" / "ai_human_agreement.csv"
DEFAULT_REPORT_PATH = OUTPUTS_DIR / "validation" / "ai_human_agreement_report.md"


@agreement_app.command("compute")
def compute(
    dimension: str = typer.Option("topical_inclusion", "--dimension", help="Comma-separated list also accepted."),
) -> None:
    """Compute AI-human agreement for one or more dimensions and write CSV + report."""
    engine = get_engine()
    dimensions = [d.strip() for d in dimension.split(",") if d.strip()]
    results = [compute_dimension_agreement(engine, d) for d in dimensions]

    write_agreement_csv(results, DEFAULT_CSV_PATH)
    write_agreement_report(results, DEFAULT_REPORT_PATH)

    for r in results:
        table = RichTable(title=f"AI-human agreement: {r.dimension}")
        table.add_column("Metric")
        table.add_column("Value")
        if r.kind == "single_label" and r.single:
            s = r.single
            table.add_row("n", str(s.n))
            table.add_row("percent agreement", f"{s.percent_agreement:.1%}" if s.n else "n/a")
            table.add_row("cohen kappa", f"{s.cohen_kappa:.3f}" if s.cohen_kappa is not None else "n/a")
            table.add_row("cohen kappa_max (ceiling)", f"{s.kappa_max:.3f}" if s.kappa_max is not None else "n/a")
            table.add_row("PABAK", f"{s.pabak:.3f}" if s.pabak is not None else "n/a")
            table.add_row("Gwet AC1", f"{s.gwet_ac1:.3f}" if s.gwet_ac1 is not None else "n/a")
            table.add_row(
                "AUC (ordinal score)",
                f"{s.weighted_auc:.3f}" if s.weighted_auc is not None else (f"{s.auc:.3f}" if s.auc is not None else "n/a"),
            )
            table.add_row(
                "false negative rate",
                f"{s.false_negative_rate:.1%}" if s.false_negative_rate is not None else "n/a",
            )
        elif r.kind == "multi_label" and r.multi:
            m = r.multi
            table.add_row("n", str(m.n))
            table.add_row("mean jaccard", f"{m.mean_jaccard:.3f}" if m.n else "n/a")
            table.add_row("micro F1", f"{m.micro_f1:.3f}" if m.n else "n/a")
            table.add_row("macro F1", f"{m.macro_f1:.3f}" if m.n else "n/a")
        console.print(table)

    console.print(f"[dim]Wrote {DEFAULT_CSV_PATH} and {DEFAULT_REPORT_PATH}[/dim]")
