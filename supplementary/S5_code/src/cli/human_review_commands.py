"""`review-cli classify ...` -- Phase 11 blinded human-validation workflow.

Screen layout/commands follow docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 15:
[f] full abstract, [m] metadata, [o] OA URL, [s] skip, [q] quit/save, Enter to decide.

An 'include' decision is followed by the 6-dimension classification step
(src/validation/dimension_classification.py): for each dimension, a numbered
category list is shown once and the reviewer says which numbers are high,
medium, and low relevance -- not one confidence prompt per category. Typing
'q' at any of those prompts quits immediately; everything answered so far
this session (including partial dimension progress on the current record) is
already committed to the DB, and the next `classify human` run for the same
--reviewer resumes at the first unanswered dimension of that same record
(current_record_id stays set on reviewer_sessions until the whole record --
decision plus every applicable dimension -- is done; see
finalize_record_review()).

`classify human` defaults to a tiered "required" review scope instead of the
full D1 pool (1800+ records): every record where the AI topical_inclusion
decision is 'uncertain' or not high-confidence (src/validation/sampling.py::
mandatory_review_ids -- these can't stand as a final decision on their own),
unioned with the stratified ~90-record validation sample (plan section 17)
used to compute formal AI-human agreement on a sample that also covers
high-confidence decisions. AI's high-confidence decisions outside that union
are accepted without individual re-review; see draft_paper/
cas-sc-template-v2.tex section 2.6 for the methodology text this must stay
consistent with. `classify build-sample` builds/rebuilds the validation
sample explicitly, or it's auto-built on first `classify human` run if
missing. Pass --sample validation for just the stratified sample, or
--sample all to fall back to the full D1 pool.

The topical decision's own prompts are lighter-weight than raw free text
where a pattern already exists: Confidence accepts the single-letter
shortcuts h/m/l as well as the full word; Rationale shows up to 5
previously-used rationale strings for that same decision label as a
numbered quick-pick (recovered from classifications_human -- see
_top_rationales/_prompt_rationale), with '*' to type a new one instead
(falls back to a plain free-text prompt the first few times, before any
history exists for that label); and Exclusion reason is a numbered pick
from config/screening_criteria.yaml's list, again with '*' for a reason not
in that list (_prompt_exclusion_reason). None of this changes what gets
saved -- ScreeningDecision still ends up with the exact same fields either
way, this just cuts down on retyping the same handful of answers record
after record.

`classify assisted` is the same review loop with one addition: weak
classical-ML models (src/validation/ml_assist.py), trained only on human
decisions made so far, suggest a topical_inclusion label AND, for an
included record, a suggestion for each of the 6 classification dimensions.
Every item -- the topical decision, the single-value evidence_type
dimension, and each of the 5 multi-value dimensions -- offers the same
accept (Enter) / override (c) step: for a multi-value dimension, the model's
raw per-category probabilities (src/validation/ml_assist.py::
bucket_suggestion) are turned into one proposed high/medium/low
classification shown as a whole, which the reviewer either accepts as-is or
rejects in favor of the normal manual prompt -- there is no partial edit of
a rejected suggestion, only accept-whole or answer-manually. Either way this
is never a replacement for the human decision, just a faster or
better-informed starting point once enough labels exist to train on for
that particular item (a dimension/category with too little data falls back
to the plain manual prompt on its own, uninformed). Retrains every
--batch-size (default 20) decisions within a session, automatically -- no
action needed -- and the per-record banner shows a live countdown of
decisions remaining until the next retrain. Logs every suggestion plus the
reviewer's actual final call to ml_assist_suggestions for override-rate
tracking (`classify ml-stats`), broken down per item -- "override" means the
reviewer rejected the suggestion outright (pressed c), for every item
including multi-value dimensions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import typer
from pydantic import ValidationError
from rich import box as rich_box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table as RichTable
from rich.text import Text
from sqlalchemy import func, select

from review_pipeline.src.common.config import load_yaml
from review_pipeline.src.db.engine import get_engine
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.screening.ai_screening import ScreeningDecision
from review_pipeline.src.validation import ml_assist, ml_model_metrics
from review_pipeline.src.validation.abstract_summary import generate_abstract_summary
from review_pipeline.src.validation.agreement import paired_decisions, single_label_agreement
from review_pipeline.src.validation.blinded_queue import next_for_human_review, set_current_record
from review_pipeline.src.validation.dimension_classification import (
    DimensionSchema,
    answered_dimensions,
    load_classification_schema,
    save_dimension_classification,
    save_single_value_classification,
)
from review_pipeline.src.validation.human_review import (
    finalize_record_review,
    save_human_decision,
    skip_current_record,
)
from review_pipeline.src.validation.ml_model_metrics import (
    DEFAULT_DIMENSION_METRICS_PATH,
    DEFAULT_DIMENSION_METRICS_PLOT_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_METRICS_PLOT_PATH,
)
from review_pipeline.src.validation.sample_size import achieved_half_width
from review_pipeline.src.validation.sampling import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SAMPLE_CSV_PATH,
    DEFAULT_SEED,
    DEFAULT_TARGET_SIZE,
    EXTENDED_STRATA_ORDER,
    benchmark_related_ids,
    build_label_stratified_sample,
    build_validation_sample,
    extend_sample_with_random_excludes,
    load_validation_sample,
    mandatory_review_ids,
    write_sample_files,
)
from review_pipeline.src.validation.validate_v2 import (
    DEFAULT_INCLUDE_N,
    DEFAULT_LOSS_PROBE_N,
    DEFAULT_V2_MANIFEST_PATH,
    DEFAULT_V2_SAMPLE_CSV_PATH,
    DEFAULT_V2_RESULTS_JSON,
    DEFAULT_V2_RESULTS_PATH,
    V2_INCLUDE,
    V2_LOSS_PROBE,
    V2_STRATA_ORDER,
    analyse_v2_validation,
    build_v2_validation_sample,
    proportion_half_width,
    load_v2_validation_sample,
    write_v2_results,
    write_v2_sample_files,
)
from review_pipeline.src.validation.validate_v2 import DEFAULT_SEED as DEFAULT_V2_SEED

classify_app = typer.Typer(help="Phase 11: blinded human validation of AI-screened records.")
console = Console()

VALID_SAMPLES: tuple[str, ...] = ("required", "validation", "v2", "benchmark", "all")


def _exclusion_reasons_for(sample: str) -> list[str]:
    """The reason vocabulary the reviewer is offered.

    Phase 12b (--sample v2) validates decisions made under screening_criteria_v2,
    so the reviewer must be able to reach v2's own reasons -- above all
    `context_only`, which is the whole point of the v2 narrowing. Offering the v1
    list there would force every domain-failure back into `off_topic` and make
    the reviewer's rationales incomparable to the AI's on exactly the axis being
    measured.
    """
    filename = "screening_criteria_v2.yaml" if sample == "v2" else "screening_criteria.yaml"
    criteria = load_yaml(filename)
    return [next(iter(r.keys())) for r in criteria.get("exclusion_reasons", [])] or ["off_topic"]


class _QuitReview(Exception):
    """Raised by a dimension prompt when the reviewer types 'q'. Caught in
    human() -- everything answered before this point is already committed."""


def _print_record_screen(record: dict) -> None:
    console.print(f"\n[bold]{record['title']}[/bold]")
    console.print(f"[dim]{record.get('venue') or 'venue unknown'} ({record.get('year') or 'year unknown'})[/dim]")
    console.print(generate_abstract_summary(record.get("abstract")))


def _handle_command(cmd: str, record: dict) -> None:
    if cmd == "f":
        console.print(f"\n[bold]Full abstract:[/bold]\n{record.get('abstract') or '(none available)'}")
    elif cmd == "m":
        console.print(
            f"\n[bold]Metadata:[/bold] publication_status={record.get('publication_status')}, "
            f"keywords={record.get('keywords') or '(none)'}"
        )
    elif cmd == "o":
        console.print(
            f"\n[bold]OA:[/bold] status={record.get('oa_status') or 'unknown'}, url={record.get('oa_url') or '(none)'}"
        )


def _prompt_choice(label: str, choices: list[str]) -> str:
    """typer's own Choice param type isn't part of its public API on the
    version pinned here (no `typer.Choice`, and click isn't a separate
    installed dependency) -- so choices are validated with a plain re-prompt
    loop instead of `type=...`."""
    while True:
        value = typer.prompt(f"{label} ({'/'.join(choices)})").strip().lower()
        if value in choices:
            return value
        console.print(f"[red]Invalid choice {value!r} -- must be one of: {', '.join(choices)}[/red]")


_CONFIDENCE_ALIASES = {"h": "high", "m": "medium", "l": "low"}


def _prompt_confidence() -> str:
    """Accepts the full word or its single-letter shortcut (h/m/l)."""
    while True:
        raw = typer.prompt("Confidence (high/medium/low, or h/m/l)").strip().lower()
        value = _CONFIDENCE_ALIASES.get(raw, raw)
        if value in ("high", "medium", "low"):
            return value
        console.print(f"[red]Invalid choice {raw!r} -- must be high/medium/low (or h/m/l).[/red]")


def _top_rationales(engine, decision_label: str, limit: int = 5) -> list[str]:
    """The `limit` most-frequently-used past rationale strings for this same
    decision label (include/exclude/uncertain), most common first -- so a
    reviewer who keeps writing the same handful of reasons can pick one
    instead of retyping it."""
    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        rows = conn.execute(
            select(classifications_human.c.rationale, func.count().label("n"))
            .where(
                classifications_human.c.dimension == "topical_inclusion",
                classifications_human.c.label == decision_label,
                classifications_human.c.rationale.isnot(None),
                classifications_human.c.rationale != "",
            )
            .group_by(classifications_human.c.rationale)
            .order_by(func.count().desc())
            .limit(limit)
        ).all()
    return [row[0] for row in rows]


def _prompt_rationale(engine, decision_label: str, default: str | None = None) -> str:
    """Shows up to 5 previously-used rationales for this decision label as
    numbered quick-picks (see _top_rationales), plus '*' to type a new one.
    Falls back to a plain free-text prompt when there's no history yet for
    this label. `default` (only set in the ML-accept flow) is what blank
    Enter returns."""
    top = _top_rationales(engine, decision_label)
    if not top:
        return typer.prompt("Rationale", default=default) if default is not None else typer.prompt("Rationale")

    console.print("[bold]Rationale[/bold] [dim](most used for this decision so far)[/dim]")
    for i, text in enumerate(top, start=1):
        console.print(f"  {i}. {text}")
    blank_hint = " (blank=accept default)" if default is not None else ""
    while True:
        raw = typer.prompt(f"  Number, or * to type a new one{blank_hint}", default="", show_default=False).strip()
        if raw == "" and default is not None:
            return default
        if raw == "*":
            return typer.prompt("New rationale", default=default) if default is not None else typer.prompt("New rationale")
        if raw.isdigit() and 1 <= int(raw) <= len(top):
            return top[int(raw) - 1]
        console.print(f"[red]Invalid entry {raw!r} -- pick a number, or * to type a new one.[/red]")


def _prompt_exclusion_reason(exclusion_reasons: list[str]) -> str:
    """Numbered pick from config/screening_criteria.yaml's exclusion reasons,
    plus '*' to type a different one not in that list."""
    console.print("[bold]Exclusion reason[/bold]")
    for i, reason in enumerate(exclusion_reasons, start=1):
        console.print(f"  {i}. {reason}")
    console.print("  * type a different reason")
    while True:
        raw = typer.prompt("  Number, or *", default="", show_default=False).strip()
        if raw == "*":
            return typer.prompt("New exclusion reason").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(exclusion_reasons):
            return exclusion_reasons[int(raw) - 1]
        console.print(f"[red]Invalid entry {raw!r} -- pick a number, or * to type a different reason.[/red]")


def _prompt_decision(engine, record_id: str, exclusion_reasons: list[str]) -> ScreeningDecision | None:
    """Returns None if the ScreeningDecision itself fails validation (e.g. a
    reason that doesn't match the decision) -- the caller re-prompts from
    scratch in that case."""
    decision = _prompt_choice("Decision", ["include", "exclude", "uncertain"])
    confidence = _prompt_confidence()
    rationale = _prompt_rationale(engine, decision)
    exclusion_reason = None
    uncertainty_reason = None
    if decision == "exclude":
        exclusion_reason = _prompt_exclusion_reason(exclusion_reasons)
    elif decision == "uncertain":
        uncertainty_reason = typer.prompt("Uncertainty reason")

    try:
        return ScreeningDecision(
            record_id=record_id,
            decision=decision,
            confidence=confidence,
            rationale=rationale,
            exclusion_reason=exclusion_reason,
            uncertainty_reason=uncertainty_reason,
        )
    except ValidationError as exc:
        console.print(f"[red]{exc}[/red]")
        return None


def _prompt_decision_accept(
    engine, record_id: str, exclusion_reasons: list[str], suggested_label: str
) -> ScreeningDecision | None:
    """Same as _prompt_decision but the decision itself is fixed to the
    accepted suggestion -- only confidence/rationale (and exclusion/
    uncertainty reason where relevant) are still the reviewer's own call."""
    confidence = _prompt_confidence()
    rationale = _prompt_rationale(
        engine, suggested_label, default=f"Confirmed ML-suggested decision ({suggested_label})."
    )
    exclusion_reason = None
    uncertainty_reason = None
    if suggested_label == "exclude":
        exclusion_reason = _prompt_exclusion_reason(exclusion_reasons)
    elif suggested_label == "uncertain":
        uncertainty_reason = typer.prompt("Uncertainty reason", default="Model-suggested uncertain; concurring.")

    try:
        return ScreeningDecision(
            record_id=record_id,
            decision=suggested_label,
            confidence=confidence,
            rationale=rationale,
            exclusion_reason=exclusion_reason,
            uncertainty_reason=uncertainty_reason,
        )
    except ValidationError as exc:
        console.print(f"[red]{exc}[/red]")
        return None


def _run_topical_decision_loop(
    record: dict, exclusion_reasons: list[str], reviewer: str, engine, sample_membership: str | None
) -> str:
    """Returns 'quit', 'skip', or the decision label ('include'/'exclude'/'uncertain')."""
    while True:
        cmd = typer.prompt("[f/m/o/s/q or Enter to decide]", default="", show_default=False).strip().lower()
        if cmd == "q":
            return "quit"
        if cmd == "s":
            return "skip"
        if cmd in ("f", "m", "o"):
            _handle_command(cmd, record)
            continue
        decision = _prompt_decision(engine, record["record_id"], exclusion_reasons)
        if decision is None:
            continue
        save_human_decision(engine, decision, reviewer_id=reviewer, sample_membership=sample_membership)
        console.print("[green]Saved.[/green]")
        return decision.decision


def _run_topical_decision_loop_assisted(
    record: dict,
    exclusion_reasons: list[str],
    reviewer: str,
    engine,
    sample_membership: str | None,
    suggestion: ml_assist.Suggestion | None,
) -> tuple[str, ml_assist.Suggestion | None]:
    """Same job as _run_topical_decision_loop, plus showing (and logging) a
    model suggestion when one exists. Returns (outcome, suggestion) so the
    caller can log the suggestion-vs-final-decision pair -- 'quit'/'skip'
    outcomes carry suggestion=None back since there's nothing to log for a
    record no decision was actually saved for."""
    if suggestion is not None:
        console.print(
            f"\n[cyan]ML suggestion:[/cyan] {suggestion.label} "
            f"(p={suggestion.probability:.2f}, trained on {suggestion.training_size} human decisions so far)"
        )
        nav_label = "[Enter=accept suggestion / c=change decision / f/m/o/s/q]"
    else:
        nav_label = "[f/m/o/s/q or Enter to decide]"

    while True:
        cmd = typer.prompt(nav_label, default="", show_default=False).strip().lower()
        if cmd == "q":
            return "quit", None
        if cmd == "s":
            return "skip", None
        if cmd in ("f", "m", "o"):
            _handle_command(cmd, record)
            continue
        if cmd == "" and suggestion is not None:
            decision = _prompt_decision_accept(engine, record["record_id"], exclusion_reasons, suggestion.label)
        elif cmd == "c" and suggestion is not None or cmd == "" and suggestion is None:
            decision = _prompt_decision(engine, record["record_id"], exclusion_reasons)
        else:
            console.print(f"[red]Invalid input {cmd!r}.[/red]")
            continue
        if decision is None:
            continue
        save_human_decision(engine, decision, reviewer_id=reviewer, sample_membership=sample_membership)
        console.print("[green]Saved.[/green]")
        return decision.decision, suggestion


def _get_saved_topical_label(engine, record_id: str, reviewer: str) -> str:
    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        row = conn.execute(
            classifications_human.select().where(
                classifications_human.c.record_id == record_id,
                classifications_human.c.coder == reviewer,
                classifications_human.c.dimension == "topical_inclusion",
            )
        ).mappings().first()
    return row["label"]


def _prompt_numbers(dim: DimensionSchema, level_label: str) -> list[str]:
    raw = typer.prompt(
        f"  {level_label} relevance -- numbers, comma-separated (blank for none, q to quit)",
        default="",
        show_default=False,
    ).strip()
    if raw.lower() == "q":
        raise _QuitReview
    chosen: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit() or not (1 <= int(tok) <= len(dim.categories)):
            console.print(f"[red]Ignoring invalid entry: {tok!r}[/red]")
            continue
        chosen.append(dim.categories[int(tok) - 1])
    return chosen


def _prompt_dimension_multi(dim: DimensionSchema) -> dict[str, list[str]]:
    """High/medium are typed in; low is whatever's left over. Every category
    not picked for high or medium is auto-attributed to low -- there's no
    separate low prompt, so the reviewer never has to enumerate the
    (usually longer) low-relevance list by hand."""
    console.print(f"\n[bold]{dim.name.replace('_', ' ').title()}[/bold] [dim](select any that apply)[/dim]")
    for i, cat in enumerate(dim.categories, start=1):
        console.print(f"  {i}. {cat}")
    high = _prompt_numbers(dim, "High")
    medium = _prompt_numbers(dim, "Medium")
    low = [cat for cat in dim.categories if cat not in high and cat not in medium]
    if low:
        console.print(f"  [dim]Low (auto-filled): {', '.join(low)}[/dim]")
    return {"high": high, "medium": medium, "low": low}


def _prompt_dimension_single(dim: DimensionSchema) -> str | None:
    console.print(f"\n[bold]{dim.name.replace('_', ' ').title()}[/bold] [dim](pick one)[/dim]")
    for i, cat in enumerate(dim.categories, start=1):
        console.print(f"  {i}. {cat}")
    raw = typer.prompt("  Number (blank for none, q to quit)", default="", show_default=False).strip()
    if raw.lower() == "q":
        raise _QuitReview
    if not raw:
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(dim.categories):
        return dim.categories[int(raw) - 1]
    console.print(f"[red]Invalid entry {raw!r} -- treating as none.[/red]")
    return None


def _format_multi_suggestion(suggestion: ml_assist.MultiLabelSuggestion) -> str:
    """Every trained category's raw probability, highest first -- shown
    alongside the bucketed high/medium/low proposal (ml_assist.
    bucket_suggestion) as supporting detail, in case the reviewer wants the
    exact numbers before deciding whether to accept it."""
    if not suggestion.category_probabilities:
        return "(no trained categories)"
    ranked = sorted(suggestion.category_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{cat} (p={prob:.2f})" for cat, prob in ranked)


def _format_buckets(buckets: dict[str, list[str]]) -> str:
    return "; ".join(f"{level}={', '.join(buckets[level]) or 'none'}" for level in ("high", "medium", "low"))


def _prompt_accept_or_override() -> str:
    """Returns 'accept' or 'override'; raises _QuitReview on 'q'."""
    while True:
        raw = typer.prompt("  [Enter=accept / c=change / q=quit]", default="", show_default=False).strip().lower()
        if raw == "q":
            raise _QuitReview
        if raw == "":
            return "accept"
        if raw == "c":
            return "override"
        console.print(f"[red]Invalid input {raw!r} -- Enter to accept, c to change, q to quit.[/red]")


def _print_model_metrics_table(title: str, metrics: dict[str, ml_model_metrics.ModelMetrics], active_model: str) -> None:
    """Renders one row per scored model type (ml_assist.MODEL_TYPES), marking
    whichever one actually drove suggestions -- shared by the live retrain
    banner in `classify assisted` and the standalone `classify ml-metrics`/
    `classify ml-backfill-metrics` commands so all three show numbers the
    same way."""
    table = RichTable(title=title)
    table.add_column("Model")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision (macro)", justify="right")
    table.add_column("Recall (macro)", justify="right")
    table.add_column("F1 (macro)", justify="right")
    table.add_column("Cohen's kappa", justify="right")
    table.add_column("n", justify="right")
    for model_type in ml_assist.MODEL_TYPES:
        m = metrics.get(model_type)
        if m is None:
            continue
        label = model_type.replace("_", " ")
        if model_type == active_model:
            label = f"[bold]{label} (active)[/bold]"
        table.add_row(
            label,
            f"{m.accuracy:.0%}",
            f"{m.precision_macro:.0%}",
            f"{m.recall_macro:.0%}",
            f"{m.f1_macro:.0%}",
            f"{m.cohen_kappa:.2f}",
            str(m.n),
        )
    console.print(table)


_CLASSIFIER_ABBREVIATIONS = {
    "logistic_regression": "LR",
    "decision_tree": "DT",
    "random_forest": "RF",
    "gradient_boosting": "GB",
}


def _short_model_label(model_type: str) -> str:
    """'tfidf_logistic_regression' -> 'tf/LR'. Only for the per-dimension
    grid, which needs 8 numeric columns to fit next to a dimension name;
    every other table has room for the real names and uses them."""
    spec = ml_assist.MODEL_SPECS.get(model_type)
    if not spec:
        return model_type
    feature_kind, classifier_kind = spec
    return f"{'tf' if feature_kind == 'tfidf' else 'em'}/{_CLASSIFIER_ABBREVIATIONS.get(classifier_kind, classifier_kind)}"


def _print_per_dimension_f1_table(
    title: str, per_dimension_metrics: dict[str, dict[str, ml_model_metrics.ModelMetrics]], active_model: str
) -> None:
    """Macro F1 per (dimension, model type) -- the breakdown behind the
    aggregate table the model-selection prompt actually reads.

    The aggregate averages across the 6 dimensions because ONE model type
    drives all of them, so a model can lead on average while being the worst
    available choice on a particular dimension. The reviewer picking that
    model should be able to see the trade they are making; the averaged row
    alone cannot show it. n is omitted here (it differs per dimension and is
    in the JSON) -- this table is for comparing columns, not for reporting."""
    if not per_dimension_metrics:
        return
    scored_types = [mt for mt in ml_assist.MODEL_TYPES if any(mt in d for d in per_dimension_metrics.values())]
    # A dimension name plus 8 model columns does not fit an 80-column terminal
    # with full box drawing; SIMPLE_HEAD drops the vertical rules and keeps it
    # readable at that width, where the boxed version truncates columns away.
    table = RichTable(title=title, box=rich_box.SIMPLE_HEAD, padding=(0, 0), pad_edge=False)
    # Deliberately wrappable: on a narrow terminal Rich shrinks this column
    # and wraps the longest dimension name, instead of dropping the rightmost
    # model column -- losing a model from a model-comparison table is the
    # worse failure, and a silent one.
    table.add_column("Dimension")
    for model_type in scored_types:
        # Full names would wrap 8 columns into unreadable slivers; the legend
        # below spells them out once.
        header = _short_model_label(model_type)
        # min_width so the header isn't truncated to fit 4-character cells;
        # the extra character over len(header) is the inter-column gap, since
        # this table carries no padding of its own.
        table.add_column(
            f"[bold]{header}[/bold]" if model_type == active_model else header, justify="right", min_width=7
        )
    for dimension, metrics in per_dimension_metrics.items():
        row = [f"{metrics[mt].f1_macro:.0%}" if mt in metrics else "--" for mt in scored_types]
        table.add_row(dimension, *row)
    console.print(table)
    console.print(
        "[dim]tf=tfidf, em=SPECTER embeddings; LR=logistic regression, DT=decision tree, "
        "RF=random forest, GB=gradient boosting. Bold column = active model.[/dim]"
    )


def _print_dimension_scoring_gaps(
    engine, schema: list[DimensionSchema], evaluation: ml_model_metrics.DimensionEvaluation | None
) -> None:
    """Names every dimension that produced no metrics this batch, and why.

    Without this the retrain banner shows the topical_inclusion table and
    then simply nothing for the dimensions -- the reviewer sees only
    include/exclude numbers and has no way to tell an "all 6 dimensions are
    fine" run from a run where the dimension evaluation returned None and
    the model-selection prompt never fired. Both of the defects recorded in
    ml_model_metrics's module docstring hid behind exactly that silence for
    weeks."""
    gaps = ml_model_metrics.dimension_scoring_diagnostics(engine, schema)
    if not gaps:
        return
    scored = set(evaluation.per_dimension_metrics) if evaluation else set()
    header = (
        "No classification dimension could be scored this batch"
        if not scored
        else f"Scored {len(scored)} dimension(s); the rest could not be scored"
    )
    console.print(f"[yellow]{header} -- no model comparison for:[/yellow]")
    for dimension, reason in gaps:
        console.print(f"  [yellow]{dimension}[/yellow]: {reason}")


# The interval the sampling plan (docs/SAMPLING_AND_AGREEMENT.md) commits to
# reporting, and the floor the claim "agreement is at least moderate" needs the
# lower bound to clear (Landis & Koch's moderate band starts at 0.41).
#
# ⚠ D-36: that floor is UNREACHABLE on this validation, and not because of the
# screener. The two coders' marginals differ so far (AI 51% include, human 12%,
# and the human never used `uncertain` at all) that kappa's own ceiling is
# kappa_max = 0.164. A rule that waits for a lower bound above 0.41 would tell
# the reviewer to keep reviewing forever. The panel below therefore reports
# kappa against its ceiling and stops on PRECISION alone, saying plainly when
# the moderate-agreement claim is arithmetically out of reach rather than
# quietly failing the test every batch.
AGREEMENT_TARGET_HALF_WIDTH = 0.15
MODERATE_AGREEMENT_FLOOR = 0.41


def _print_ai_agreement_panel(engine, dimension: str = "topical_inclusion") -> None:
    """Live AI-human agreement on the decisions made so far, with the CI that
    decides when reviewing can stop.

    This is the number the manuscript actually reports, so showing it per batch
    (rather than only at the end via `review-cli agreement`) is what makes the
    pre-registered stopping rule operable: the reviewer can see the interval
    tighten and stop as soon as it clears the floor, instead of committing
    up-front to an n picked before any agreement was observed.

    Deliberately reads only decisions already made. Nothing here reaches the
    review screen for a record still to be decided, so it cannot unblind the
    queue (blinded_queue.py's guarantee) -- the reviewer sees their own past
    performance in aggregate, never the AI's label for a pending record.
    """
    pairs = paired_decisions(engine, dimension)
    result = single_label_agreement(pairs)
    if result.n == 0:
        console.print("[dim]No AI-human paired decisions yet -- agreement starts once screened records are reviewed.[/dim]")
        return

    table = RichTable(title=f"AI-human agreement so far ({dimension})")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Paired decisions (n)", str(result.n))
    table.add_row("Raw agreement", f"{result.percent_agreement:.0%}")

    if result.cohen_kappa is None:
        table.add_row("Cohen's kappa", "n/a (only one class present)")
        console.print(table)
        return

    kappa = result.cohen_kappa
    table.add_row("Cohen's kappa", f"{kappa:.3f}")
    ceiling = result.kappa_max
    if ceiling:
        table.add_row("kappa_max (ceiling here)", f"{ceiling:.3f}  ({kappa / ceiling:.0%} of it reached)")
    if result.gwet_ac1 is not None:
        table.add_row("Gwet's AC1", f"{result.gwet_ac1:.3f}")
    if result.auc is not None:
        table.add_row("AUC (label x confidence)", f"{result.auc:.3f}")

    # Prevalence drives the variance, so use the observed one rather than the
    # planning assumption -- this is the realised interval, not the projected.
    prevalence = sum(1 for p in pairs if p.human_label == "include") / result.n
    if 0.0 < prevalence < 1.0 and -1.0 < kappa < 1.0:
        half_width = achieved_half_width(result.n, kappa, prevalence)
        low, high = kappa - half_width, kappa + half_width
        table.add_row("95% CI", f"[{low:.2f}, {high:.2f}]  (±{half_width:.2f})")
        table.add_row("Observed include rate", f"{prevalence:.0%}")
        console.print(table)

        ceiling_blocks_the_floor = ceiling is not None and ceiling <= MODERATE_AGREEMENT_FLOOR
        if ceiling_blocks_the_floor:
            console.print(
                f"[yellow]The {MODERATE_AGREEMENT_FLOOR} floor is unreachable here[/yellow] -- with these "
                f"marginals kappa cannot exceed {ceiling:.3f} no matter how the remaining records are "
                f"judged (D-36). Kappa is measuring the AI's threshold, not its judgement; read AC1, "
                f"AUC and the precision/recall pair instead, and stop on precision alone."
            )
            if half_width <= AGREEMENT_TARGET_HALF_WIDTH:
                console.print(
                    f"[bold green]Precision target met[/bold green] -- half-width ±{half_width:.2f} is "
                    f"within ±{AGREEMENT_TARGET_HALF_WIDTH}. Further review refines the estimate but is "
                    f"not required by the sampling plan."
                )
        elif low > MODERATE_AGREEMENT_FLOOR and half_width <= AGREEMENT_TARGET_HALF_WIDTH:
            console.print(
                f"[bold green]Stopping rule met[/bold green] -- the interval clears {MODERATE_AGREEMENT_FLOOR} "
                f"and is within ±{AGREEMENT_TARGET_HALF_WIDTH}. Reviewing more records refines this "
                f"estimate but is no longer required by the sampling plan."
            )
        else:
            console.print(
                f"[dim]Stopping rule not met yet (need lower bound > {MODERATE_AGREEMENT_FLOOR} "
                f"and half-width <= {AGREEMENT_TARGET_HALF_WIDTH}).[/dim]"
            )
    else:
        console.print(table)


def _prompt_active_model(
    metrics: dict[str, ml_model_metrics.ModelMetrics], current: str, what: str = "decisions"
) -> str:
    """Lets the reviewer pick which trained model type drives `what` until
    the next retrain -- Enter keeps the current one. Only offers model types
    that were actually scored this batch (i.e. present in `metrics`). Shared
    by the topical_inclusion prompt (what="decisions", the default) and the
    dimension-suggestion prompt (what="all 6 classification dimensions")."""
    available = [mt for mt in ml_assist.MODEL_TYPES if mt in metrics]
    console.print(f"[dim]Numbers: {', '.join(f'{i}={mt}' for i, mt in enumerate(available, start=1))}[/dim]")
    while True:
        raw = typer.prompt(
            f"  Which model should make {what} until the next retrain? (Enter=keep {current})",
            default="",
            show_default=False,
        ).strip()
        if raw == "":
            return current
        if raw.isdigit() and 1 <= int(raw) <= len(available):
            return available[int(raw) - 1]
        if raw in available:
            return raw
        console.print(f"[red]Invalid entry {raw!r} -- pick a number or one of: {', '.join(available)}.[/red]")


def _missing_dimension_names(schema: list[DimensionSchema], dim_models: dict[str, ml_assist.DimensionModel]) -> list[str]:
    """schema dimension names with no trained model in dim_models, in schema
    order -- named explicitly in the assisted() banners below instead of a
    bare 'others', since a whole dimension going quietly missing (e.g. every
    category failing ml_assist.MIN_EXAMPLES_PER_CLASS at once, which severe
    label imbalance -- most records sharing nearly the same categories --
    makes easy to hit) is otherwise indistinguishable from everything being
    fine until someone goes looking."""
    return [dim.name for dim in schema if dim.name not in dim_models]


def _classify_dimensions(
    engine,
    record: dict,
    reviewer: str,
    schema: list[DimensionSchema],
    already: set[str],
    dim_models: dict[str, ml_assist.DimensionModel] | None = None,
) -> bool:
    """Runs the 6-dimension classification step for an 'include' record.
    Returns True if every dimension got answered, False if the reviewer quit
    mid-way (progress already committed either way). When dim_models is
    given (classify assisted): every dimension with a trained model offers
    the same Enter-to-accept / c-to-change step. A multi-value dimension's
    raw per-category probabilities are bucketed into one proposed
    high/medium/low classification (ml_assist.bucket_suggestion) shown as a
    whole; accepting takes that bucketing as-is, changing drops into the
    normal manual high/medium/low prompt from scratch (no partial edit). A
    single-value dimension (just evidence_type) accepts/changes the single
    top-probability category the same way it always did. A dimension with no
    model (not enough labeled data yet) always falls back to the plain
    manual prompt, same as classify human."""
    pending_dims = [dim for dim in schema if dim.name not in already]
    try:
        for i, dim in enumerate(pending_dims, start=1):
            console.print(f"[dim]-- classification dimension {i} of {len(pending_dims)} --[/dim]")
            model = dim_models.get(dim.name) if dim_models else None
            suggestion = ml_assist.suggest_dimension(model, record)

            if isinstance(suggestion, ml_assist.MultiLabelSuggestion):
                buckets = ml_assist.bucket_suggestion(suggestion)
                console.print(
                    f"[cyan]ML suggestion -- {dim.name.replace('_', ' ')}:[/cyan] {_format_buckets(buckets)} "
                    f"[dim](trained on {suggestion.training_size} human classifications; "
                    f"raw: {_format_multi_suggestion(suggestion)})[/dim]"
                )
                choice = _prompt_accept_or_override()
                levels = buckets if choice == "accept" else _prompt_dimension_multi(dim)
                save_dimension_classification(engine, record["record_id"], reviewer, dim.name, levels)
                ml_assist.log_suggestion(
                    engine,
                    record["record_id"],
                    reviewer,
                    suggested_label=json.dumps(suggestion.category_probabilities, sort_keys=True),
                    suggested_probability=(
                        sum(suggestion.category_probabilities.values()) / len(suggestion.category_probabilities)
                        if suggestion.category_probabilities
                        else 0.0
                    ),
                    training_size=suggestion.training_size,
                    final_label=json.dumps(levels, sort_keys=True),
                    dimension=dim.name,
                    was_override=(choice == "override"),
                )
            elif isinstance(suggestion, ml_assist.SingleLabelSuggestion):
                console.print(
                    f"[cyan]ML suggestion -- {dim.name.replace('_', ' ')}:[/cyan] "
                    f"{suggestion.label or '(none)'} (p={suggestion.probability:.2f}) "
                    f"[dim](trained on {suggestion.training_size} human classifications)[/dim]"
                )
                category = (
                    (suggestion.label or None)
                    if _prompt_accept_or_override() == "accept"
                    else _prompt_dimension_single(dim)
                )
                save_single_value_classification(engine, record["record_id"], reviewer, dim.name, category)
                ml_assist.log_suggestion(
                    engine,
                    record["record_id"],
                    reviewer,
                    suggested_label=suggestion.label,
                    suggested_probability=suggestion.probability,
                    training_size=suggestion.training_size,
                    final_label=category or "",
                    dimension=dim.name,
                )
            elif dim.multi_value:
                levels = _prompt_dimension_multi(dim)
                save_dimension_classification(engine, record["record_id"], reviewer, dim.name, levels)
            else:
                category = _prompt_dimension_single(dim)
                save_single_value_classification(engine, record["record_id"], reviewer, dim.name, category)
    except _QuitReview:
        console.print(
            "[dim]Session saved -- dimension progress kept for this record. "
            "Resume anytime with the same --reviewer.[/dim]"
        )
        return False

    console.print("[green]Classification complete for this record.[/green]")
    return True


@classify_app.command("build-sample")
def build_sample(
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    target_size: int = typer.Option(DEFAULT_TARGET_SIZE, "--target-size"),
    allocation: str = typer.Option(
        "thematic",
        "--allocation",
        help=(
            "'thematic' (default): quotas by disagreement-likelihood and topic -- good at "
            "finding screening errors, but the include count is incidental, so precision of "
            "the include label is not estimable to a stated half-width. "
            "'by-ai-label': stratify on the Phase 10 AI decision, over-sampling include and "
            "uncertain, so include precision IS estimable -- reweight by N_h/n_h (in the manifest) "
            "for any population-level statistic."
        ),
    ),
) -> None:
    """Builds (or rebuilds) the stratified validation sample `classify human`
    reviews by default -- see docs/TECHNICAL_IMPLEMENTATION_PLAN.md section 17
    and src/validation/sampling.py. Deterministic for a given (seed, DB
    state), so re-running with the same seed after new records are added
    reshuffles the sample; run this deliberately, not automatically, once a
    sample is already in active use."""
    normalized = allocation.replace("-", "_")
    if normalized not in ("thematic", "by_ai_label"):
        console.print(f"[red]Unknown --allocation {allocation!r}; use 'thematic' or 'by-ai-label'.[/red]")
        raise typer.Exit(code=2)

    engine = get_engine()
    if normalized == "by_ai_label":
        result = build_label_stratified_sample(engine, seed=seed, target_size=target_size)
        strata = ("ai_include", "ai_exclude", "ai_uncertain")
    else:
        result = build_validation_sample(engine, seed=seed, target_size=target_size)
        strata = ("low_confidence", "excluded_borderline", "chemical_pse", "thematic", "random_common")
    write_sample_files(result, DEFAULT_SAMPLE_CSV_PATH, DEFAULT_MANIFEST_PATH)

    table = RichTable(
        title=f"Validation sample ({allocation}) -- {len(result.rows)}/{result.pool_size} D1 records (seed={seed})"
    )
    table.add_column("Stratum")
    table.add_column("Count")
    table.add_column("Population")
    table.add_column("Weight N/n")
    for stratum in strata:
        n = result.stratum_counts.get(stratum, 0)
        big_n = result.population_counts.get(stratum)
        table.add_row(
            stratum,
            str(n),
            str(big_n) if big_n else "-",
            f"{big_n / n:.1f}" if big_n and n else "-",
        )
    console.print(table)
    console.print(f"[green]Sample written to {DEFAULT_SAMPLE_CSV_PATH}, manifest to {DEFAULT_MANIFEST_PATH}[/green]")
    console.print("[dim]Run `classify human --reviewer <name>` next -- it uses this sample by default.[/dim]")


@classify_app.command("extend-sample")
def extend_sample(
    size: int = typer.Option(100, "--size", help="How many high-confidence excludes to draw at random."),
    seed: int = typer.Option(DEFAULT_SEED + 1, "--seed"),
) -> None:
    """Close the one unvalidated stratum: draw `--size` AI-`exclude` records of
    HIGH confidence with equal probability, and re-label the existing exclude
    rows as the borderline frame they were really drawn from (D-35).

    Unlike `build-sample`, this is additive -- every record already in the
    sample stays in it, under the same decisions, so adjudications already made
    are untouched. What changes is that each exclude cell now carries a weight
    for a population it was actually drawn from at random, which is what makes
    a corpus-level recall estimate defensible.

    Review the new records with `classify assisted --sample validation`.
    """
    existing = load_validation_sample()
    if not existing:
        console.print(
            f"[red]No sample at {DEFAULT_SAMPLE_CSV_PATH}[/red] -- "
            "run `classify build-sample --allocation by-ai-label` first."
        )
        raise typer.Exit(code=1)

    engine = get_engine()
    result = extend_sample_with_random_excludes(engine, existing, size=size, seed=seed)
    added = len(result.rows) - len(existing)
    write_sample_files(result, DEFAULT_SAMPLE_CSV_PATH, DEFAULT_MANIFEST_PATH)

    table = RichTable(
        title=f"Extended validation sample -- {len(result.rows)} records (+{added} new, seed={seed})"
    )
    table.add_column("Stratum")
    table.add_column("Count", justify="right")
    table.add_column("Population", justify="right")
    table.add_column("Weight N/n", justify="right")
    for stratum in EXTENDED_STRATA_ORDER:
        n = result.stratum_counts.get(stratum, 0)
        big_n = result.population_counts.get(stratum)
        table.add_row(
            stratum,
            str(n),
            str(big_n) if big_n else "-",
            f"{big_n / n:.1f}" if big_n and n else "-",
        )
    console.print(table)
    console.print(f"[green]Sample written to {DEFAULT_SAMPLE_CSV_PATH}, manifest to {DEFAULT_MANIFEST_PATH}[/green]")
    console.print(
        f"[dim]Next: `classify assisted --reviewer <name> --sample validation` -- {added} unreviewed records.[/dim]"
    )


@classify_app.command("build-sample-v2")
def build_sample_v2(
    include_n: int = typer.Option(
        DEFAULT_INCLUDE_N, "--include-n", help="Draws from the unadjudicated v2 includes."
    ),
    loss_probe_n: int = typer.Option(
        DEFAULT_LOSS_PROBE_N, "--loss-probe-n", help="Draws from the v1-include/v2-exclude set."
    ),
    seed: int = typer.Option(DEFAULT_V2_SEED, "--seed"),
) -> None:
    """Draw the Phase 12b sample: the unbiased test of the v2 screening criterion
    (D-47, src/validation/validate_v2.py).

    Phase 10b's headline precision of 56.9% is a resubstitution estimate -- v2 was
    written after inspecting the Phase 11 records it is then scored against. This
    draws from records that were NOT adjudicated in Phase 11, so the resulting
    precision is measured on evidence v2's author never saw.

    Two strata, and they answer different questions with different strength:
    `v2_include` gives an unbiased precision to a stated half-width;
    `v2_loss_probe` gives a one-sided BOUND on corpus lost to the narrowing, not
    a recall estimate -- 30 draws cannot estimate a rate (D-35).

    Review the result with `classify assisted --sample v2`.
    """
    engine = get_engine()
    result = build_v2_validation_sample(
        engine, include_n=include_n, loss_probe_n=loss_probe_n, seed=seed
    )
    write_v2_sample_files(result, DEFAULT_V2_SAMPLE_CSV_PATH, DEFAULT_V2_MANIFEST_PATH)

    table = RichTable(
        title=f"Phase 12b validation sample -- {len(result.members)} records (seed={seed})"
    )
    table.add_column("Stratum")
    table.add_column("n", justify="right")
    table.add_column("Population N", justify="right")
    table.add_column("Weight N/n", justify="right")
    table.add_column("Half-width at p=0.5", justify="right")
    for stratum in V2_STRATA_ORDER:
        n = result.sample_counts.get(stratum, 0)
        big_n = result.population_counts.get(stratum, 0)
        table.add_row(
            stratum,
            str(n),
            str(big_n) if big_n else "-",
            f"{big_n / n:.2f}" if big_n and n else "-",
            f"±{proportion_half_width(n, big_n) * 100:.1f}pp" if n and big_n else "-",
        )
    console.print(table)
    console.print(
        f"[green]Sample written to {DEFAULT_V2_SAMPLE_CSV_PATH}, "
        f"manifest to {DEFAULT_V2_MANIFEST_PATH}[/green]"
    )
    console.print(
        "[dim]Next: `classify assisted --reviewer <name> --sample v2`. "
        "The include stratum is also a down payment -- all of its population must be "
        "adjudicated before synthesis regardless.[/dim]"
    )


@classify_app.command("report-v2")
def report_v2() -> None:
    """Score the adjudicated Phase 12b sample and write the results files (D-50).

    Every Phase 12b figure the manuscript quotes comes from here. Nothing is
    transcribed by hand: `outputs/validation/phase12b_results.{md,json}` are the
    citable artefacts, and re-running this after any further adjudication
    refreshes them.
    """
    engine = get_engine()
    result = analyse_v2_validation(engine)
    write_v2_results(result, DEFAULT_V2_RESULTS_PATH, DEFAULT_V2_RESULTS_JSON)

    inc = result.strata[V2_INCLUDE]
    probe = result.strata[V2_LOSS_PROBE]
    lo, hi = inc.exact_ci

    table = RichTable(title="Phase 12b -- v2 criterion, measured without resubstitution")
    table.add_column("Quantity")
    table.add_column("Value", justify="right")
    table.add_row("precision of a v2 include", f"{inc.includes}/{inc.n} = {inc.rate:.1%}")
    table.add_row("  exact 95% CI", f"{lo:.1%} - {hi:.1%}")
    table.add_row("  FPC half-width", f"±{inc.half_width * 100:.1f}pp")
    table.add_row(
        "resubstitution estimate (biased)",
        f"{result.resubstitution_includes}/{result.resubstitution_n} = "
        f"{result.resubstitution_rate:.1%}",
    )
    table.add_row("  optimistic bias", f"{result.bias * 100:.1f}pp")
    table.add_row("loss probe (BOUND, not a rate)", f"{probe.includes}/{probe.n} = {probe.rate:.1%}")
    table.add_row(
        "  one-sided 95% upper",
        f"{probe.upper_bound:.1%} (<= {probe.upper_bound * probe.population_n:.0f} "
        f"of {probe.population_n})",
    )
    console.print(table)

    if any(result.unadjudicated.values()):
        pending = ", ".join(f"{k}: {v}" for k, v in result.unadjudicated.items() if v)
        console.print(f"[yellow]Provisional -- still unadjudicated: {pending}[/yellow]")
    else:
        console.print(f"[green]Complete. Report at {DEFAULT_V2_RESULTS_PATH}[/green]")
    console.print(
        f"[dim]The loss probe is an upper bound on corpus lost to the narrowing. "
        f"Do not quote {probe.rate:.1%} as a false-negative rate (D-35, D-48).[/dim]"
    )


@classify_app.command("human")
def human(
    reviewer: str = typer.Option(..., "--reviewer", help="Reviewer id/name."),
    sample: str = typer.Option(
        "required",
        "--sample",
        help="'required' (default -- every AI-uncertain/non-high-confidence D1 record, unioned "
        "with the stratified validation sample from `classify build-sample`), 'validation' "
        "(just the stratified validation sample, exactly as built -- 400 records under D-29), "
        "'v2' (the Phase 12b sample from `classify build-sample-v2` -- the unbiased test of the "
        "v2 criterion, offered with v2's exclusion reasons), 'benchmark' (every D1 record AI-tagged "
        "evidence_type=benchmark or matching benchmark/dataset keywords -- for prioritizing "
        "the benchmark-dataset gap-analysis thread), or 'all' (the full D1 pool).",
    ),
) -> None:
    """Blinded human classification loop. Never reads AI labels (see
    src/validation/blinded_queue.py's structural guarantee)."""
    engine = get_engine()
    schema = load_classification_schema()

    if sample not in VALID_SAMPLES:
        console.print(
            f"[red]Unknown --sample {sample!r} -- must be one of "
            f"{', '.join(repr(s) for s in VALID_SAMPLES)}.[/red]"
        )
        raise typer.Exit(code=1)
    exclusion_reasons = _exclusion_reasons_for(sample)

    sample_membership_by_record: dict[str, str] | None = None
    if sample == "v2":
        sample_membership_by_record = load_v2_validation_sample()
        if sample_membership_by_record is None:
            console.print(
                f"[red]No Phase 12b sample at {DEFAULT_V2_SAMPLE_CSV_PATH}[/red] -- "
                "run `classify build-sample-v2` first. It is not auto-built: it is a "
                "pre-registered draw and must be created deliberately, once."
            )
            raise typer.Exit(code=1)
    elif sample == "benchmark":
        sample_membership_by_record = benchmark_related_ids(engine)
        console.print(
            f"[dim]Benchmark-related set: {len(sample_membership_by_record)} D1 record(s) "
            f"(AI evidence_type=benchmark or benchmark/dataset keyword match).[/dim]"
        )
    elif sample in ("required", "validation"):
        sample_membership_by_record = load_validation_sample(DEFAULT_SAMPLE_CSV_PATH)
        if sample_membership_by_record is None:
            console.print(
                f"[yellow]No validation sample found -- building one now "
                f"(seed={DEFAULT_SEED}, target_size={DEFAULT_TARGET_SIZE}).[/yellow]"
            )
            result = build_validation_sample(engine, seed=DEFAULT_SEED, target_size=DEFAULT_TARGET_SIZE)
            write_sample_files(result, DEFAULT_SAMPLE_CSV_PATH, DEFAULT_MANIFEST_PATH)
            sample_membership_by_record = {row.record_id: row.stratum for row in result.rows}
            console.print(
                f"[green]Built a {len(sample_membership_by_record)}-record sample "
                f"(see {DEFAULT_SAMPLE_CSV_PATH}).[/green]"
            )
        if sample == "required":
            mandatory = mandatory_review_ids(engine)
            merged = dict(sample_membership_by_record)
            for record_id, reason in mandatory.items():
                merged.setdefault(record_id, reason)
            sample_membership_by_record = merged
            console.print(
                f"[dim]Required-review set: {len(sample_membership_by_record)} records "
                f"({len(mandatory)} AI-uncertain/non-high-confidence + validation sample). "
                f"Pass --sample validation for just the sample, or --sample all for the full D1 pool.[/dim]"
            )
    allowed_ids = set(sample_membership_by_record) if sample_membership_by_record is not None else None

    while True:
        record = next_for_human_review(engine, reviewer, allowed_record_ids=allowed_ids)
        if record is None:
            console.print("[green]Queue empty -- nothing left to review.[/green]")
            return

        counts = _progress_counts(engine, allowed_ids)
        pool_label = {
            "required": "required-review set",
            "validation": "validation sample",
            "v2": "Phase 12b v2-validation sample",
            "benchmark": "benchmark-related set",
            "all": "D1 pool",
        }[sample]
        console.print(
            f"\n[bold cyan]── {counts.reviewed}/{counts.d1_total} reviewed so far in the {pool_label} "
            f"({counts.d1_total - counts.reviewed} remaining) ──[/bold cyan]"
        )

        set_current_record(engine, reviewer, record["record_id"])
        already = answered_dimensions(engine, record["record_id"], reviewer)
        sample_membership = (
            sample_membership_by_record.get(record["record_id"]) if sample_membership_by_record else None
        )

        if "topical_inclusion" not in already:
            _print_record_screen(record)
            outcome = _run_topical_decision_loop(record, exclusion_reasons, reviewer, engine, sample_membership)
            if outcome == "quit":
                console.print("[dim]Session saved. Resume anytime with the same --reviewer.[/dim]")
                return
            if outcome == "skip":
                skip_current_record(engine, reviewer)
                continue
            decision_label = outcome
        else:
            decision_label = _get_saved_topical_label(engine, record["record_id"], reviewer)
            console.print(f"\n[dim]Resuming {record['title']!r} -- decision already recorded: {decision_label}.[/dim]")

        if decision_label != "include":
            finalize_record_review(engine, reviewer)
            continue

        if not _classify_dimensions(engine, record, reviewer, schema, already):
            return
        finalize_record_review(engine, reviewer)


@classify_app.command("assisted")
def assisted(
    reviewer: str = typer.Option(..., "--reviewer", help="Reviewer id/name."),
    sample: str = typer.Option(
        "required",
        "--sample",
        help="Same meaning as `classify human`'s --sample: 'required' (default), 'validation', "
        "'v2', 'benchmark', or 'all'.",
    ),
    batch_size: int = typer.Option(
        20, "--batch-size", help="Retrain the suggestion model every this-many decisions made this session."
    ),
) -> None:
    """Same blinded review loop as `classify human`, plus a weak ML suggestion
    for the topical_inclusion decision and, for an included record, for each
    of the 6 classification dimensions (src/validation/ml_assist.py). Every
    item -- the topical decision, the single-value evidence_type dimension,
    and each of the 5 multi-value dimensions -- offers the same
    Enter-accepts-as-is / c-to-change step: for a multi-value dimension, raw
    per-category probabilities are bucketed into one proposed high/medium/low
    classification (ml_assist.bucket_suggestion) shown as a whole, accepted
    whole or rejected in favor of the normal manual prompt. Trained only on
    human decisions so far -- never on AI screening labels, so this doesn't
    weaken the blinding guarantee, it just makes confirming (or informing) a
    decision faster once enough labels exist. Falls back to the plain
    unassisted prompt, per-item, until there's enough labeled data for that
    specific decision or dimension (see ml_assist.MIN_TOTAL_EXAMPLES).

    Eight model types (ml_assist.MODEL_SPECS: logistic regression, decision
    tree, random forest, and gradient boosting, each over both TF-IDF and
    SPECTER-embedding features -- src/validation/embeddings.py) are trained
    side by side every retrain, each through a basic CV + hyperparameter-
    selection pass (ml_assist._fit_with_cv), but only one -- the "active"
    model -- actually generates the suggestions shown above. It starts as
    ml_assist.DEFAULT_ACTIVE_MODEL (tfidf_logistic_regression -- the one
    model that never depends on the embedding backend -- or whichever model
    the reviewer picked last time, if a metrics history file already
    exists). Right before each retrain, every model type is fit ONCE on
    every human decision made so far (not just this batch's) and scored by
    its own out-of-fold cross-validation predictions from that same fit
    (ml_model_metrics.cumulative_metrics) -- a single ~--batch-size batch is
    both small and, given how imbalanced human decisions are, easily
    unrepresentative on its own, and even a single random holdout split of
    the full cumulative set stays noisy batch-to-batch (a fresh unrelated
    resample every retrain); out-of-fold CV uses every example exactly once,
    each scored by a model that never trained on it, so the reported numbers
    get both larger-sample-size and more stable as review continues. Every
    model type is scored this way even though only the active one was
    actually shown -- and the reviewer is then shown all of them side by
    side and asked which one should drive suggestions until the next
    retrain. If the embedding backend is unavailable (e.g. no network to
    download SPECTER), the embedding_* models are silently skipped and the
    tfidf_* ones still work. Every batch's metrics are appended to
    ml_model_metrics.DEFAULT_METRICS_PATH and a trend plot is regenerated at
    ml_model_metrics.DEFAULT_METRICS_PLOT_PATH -- see `classify ml-metrics`
    and `classify ml-backfill-metrics`.

    The 6 classification dimensions go through the SAME eight-model
    comparison (ml_assist.train_dimensions_all), tracked in a SEPARATE file/
    plot (ml_model_metrics.DEFAULT_DIMENSION_METRICS_PATH/
    DEFAULT_DIMENSION_METRICS_PLOT_PATH) -- but unlike topical_inclusion,
    there's a single active model type applied uniformly to ALL 6
    dimensions (not one choice per dimension), picked from each model
    type's out-of-fold CV score averaged ACROSS the 6 dimensions
    (ml_model_metrics.cumulative_dimension_metrics). These skip the
    hyperparameter search topical_inclusion's models go through
    (ml_assist._fit_without_search, not _fit_with_cv) -- there can be dozens
    of (category, model_type) combinations across the 6 dimensions, and
    they're meant as a fast second opinion, not a tuned model, so searching
    per combination would be real, avoidable retrain cost. Right after the
    topical retrain prompt, a second table and a second
    "which model for all 6 classification dimensions?" prompt appear the
    same way."""
    engine = get_engine()
    schema = load_classification_schema()

    if sample not in VALID_SAMPLES:
        console.print(
            f"[red]Unknown --sample {sample!r} -- must be one of "
            f"{', '.join(repr(s) for s in VALID_SAMPLES)}.[/red]"
        )
        raise typer.Exit(code=1)
    exclusion_reasons = _exclusion_reasons_for(sample)

    sample_membership_by_record: dict[str, str] | None = None
    if sample == "v2":
        sample_membership_by_record = load_v2_validation_sample()
        if sample_membership_by_record is None:
            console.print(
                f"[red]No Phase 12b sample at {DEFAULT_V2_SAMPLE_CSV_PATH}[/red] -- "
                "run `classify build-sample-v2` first. It is not auto-built: it is a "
                "pre-registered draw and must be created deliberately, once."
            )
            raise typer.Exit(code=1)
    elif sample == "benchmark":
        sample_membership_by_record = benchmark_related_ids(engine)
        console.print(
            f"[dim]Benchmark-related set: {len(sample_membership_by_record)} D1 record(s) "
            f"(AI evidence_type=benchmark or benchmark/dataset keyword match).[/dim]"
        )
    elif sample in ("required", "validation"):
        sample_membership_by_record = load_validation_sample(DEFAULT_SAMPLE_CSV_PATH)
        if sample_membership_by_record is None:
            result = build_validation_sample(engine, seed=DEFAULT_SEED, target_size=DEFAULT_TARGET_SIZE)
            write_sample_files(result, DEFAULT_SAMPLE_CSV_PATH, DEFAULT_MANIFEST_PATH)
            sample_membership_by_record = {row.record_id: row.stratum for row in result.rows}
        if sample == "required":
            mandatory = mandatory_review_ids(engine)
            merged = dict(sample_membership_by_record)
            for record_id, reason in mandatory.items():
                merged.setdefault(record_id, reason)
            sample_membership_by_record = merged
    allowed_ids = set(sample_membership_by_record) if sample_membership_by_record is not None else None

    metrics_history = ml_model_metrics.load_history(DEFAULT_METRICS_PATH)
    dimension_metrics_history = ml_model_metrics.load_dimension_history(DEFAULT_DIMENSION_METRICS_PATH)
    batch_number = (metrics_history[-1].batch + 1) if metrics_history else 1
    active_model_type = (
        ml_model_metrics.load_current_active_model(DEFAULT_METRICS_PATH) or ml_assist.DEFAULT_ACTIVE_MODEL
    )
    if active_model_type not in ml_assist.MODEL_TYPES:
        # A name from before the tfidf_*/embedding_* model rename, or any
        # other stale/unrecognized value -- fall back rather than KeyError
        # below. DEFAULT_ACTIVE_MODEL is tfidf-only, so it's always a safe
        # landing spot regardless of what broke.
        active_model_type = ml_assist.DEFAULT_ACTIVE_MODEL
    active_dimension_model_type = (
        ml_model_metrics.load_current_active_dimension_model(DEFAULT_DIMENSION_METRICS_PATH)
        or ml_assist.DEFAULT_ACTIVE_MODEL
    )
    if active_dimension_model_type not in ml_assist.MODEL_TYPES:
        active_dimension_model_type = ml_assist.DEFAULT_ACTIVE_MODEL

    # Topical decisions are committed to the DB the moment they're made (see
    # save_human_decision), so nothing is ever lost when a session is
    # aborted -- but decisions_since_retrain/batch_examples used to live only
    # in this function's local variables, so they silently reset to empty on
    # every new invocation. Reloading the pending batch here instead means a
    # decision made in a session that later quit (or crashed) still counts
    # toward triggering -- and gets scored by -- the next retrain, instead of
    # the countdown resetting to a fresh batch_size and that decision quietly
    # never being evaluated.
    batch_examples: list[tuple[str, str]] = ml_model_metrics.load_pending_batch(DEFAULT_METRICS_PATH)
    decisions_since_retrain = len(batch_examples)

    # If a batch is already partway done, train suggestion models on exactly
    # the data that existed before that batch started (not everything
    # currently in the DB) -- otherwise the pending decisions above would
    # leak into their own evaluation once this batch's retrain fires (see
    # ml_assist.train_all's max_examples docstring).
    trained_through = len(ml_assist._labeled_examples(engine)) - decisions_since_retrain
    models = ml_assist.train_all(engine, max_examples=trained_through)
    if models and active_model_type not in models:
        # Most likely an embedding_* model that couldn't be trained this
        # session (SPECTER unavailable -- see the notice below); fall back
        # to the tfidf default rather than silently reviewing unassisted.
        active_model_type = ml_assist.DEFAULT_ACTIVE_MODEL
    model = models.get(active_model_type) if models else None
    if models:
        missing_embedding_models = [
            mt for mt, (feature_kind, _ck) in ml_assist.MODEL_SPECS.items()
            if feature_kind == "embedding" and mt not in models
        ]
        if missing_embedding_models:
            console.print(
                "[yellow]Embedding-based suggestion models are unavailable this session "
                "(SPECTER embedding backend failed to load -- check network access / "
                "model download) -- continuing with the tfidf-based models only.[/yellow]"
            )
    if model is None:
        console.print(
            f"[yellow]Not enough labeled data yet for a suggestion model "
            f"(need >={ml_assist.MIN_TOTAL_EXAMPLES} decisions across >=2 classes, "
            f">={ml_assist.MIN_EXAMPLES_PER_CLASS} each) -- reviewing unassisted for now.[/yellow]"
        )
    else:
        console.print(
            f"[dim]Suggestion model ({active_model_type}) trained on {model.training_size} human decisions "
            f"({dict(model.class_counts)}). Retrains every {batch_size} decisions this session.[/dim]"
        )
    if decisions_since_retrain:
        console.print(
            f"[dim]Resuming batch {batch_number} -- {decisions_since_retrain} decision"
            f"{'s' if decisions_since_retrain != 1 else ''} already made toward it (from a previous session).[/dim]"
        )
    dimension_models_by_type = ml_assist.train_dimensions_all(engine, schema)
    if dimension_models_by_type and active_dimension_model_type not in {
        mt for dim_type_models in dimension_models_by_type.values() for mt in dim_type_models
    }:
        active_dimension_model_type = ml_assist.DEFAULT_ACTIVE_MODEL
    dim_models = {
        name: dim_type_models[active_dimension_model_type]
        for name, dim_type_models in dimension_models_by_type.items()
        if active_dimension_model_type in dim_type_models
    }
    if dim_models:
        missing = _missing_dimension_names(schema, dim_models)
        console.print(
            f"[dim]Dimension suggestion models ({active_dimension_model_type}) trained for: "
            f"{', '.join(sorted(dim_models))}"
            + (f" (no model yet for: {', '.join(missing)} -- not enough labeled data).[/dim]" if missing else " (all 6 dimensions covered).[/dim]")
        )

    while True:
        record = next_for_human_review(engine, reviewer, allowed_record_ids=allowed_ids)
        if record is None:
            console.print("[green]Queue empty -- nothing left to review.[/green]")
            return

        counts = _progress_counts(engine, allowed_ids)
        pool_label = {
            "required": "required-review set",
            "validation": "validation sample",
            "v2": "Phase 12b v2-validation sample",
            "benchmark": "benchmark-related set",
            "all": "D1 pool",
        }[sample]
        console.print(
            f"\n[bold cyan]── {counts.reviewed}/{counts.d1_total} reviewed so far in the {pool_label} "
            f"({counts.d1_total - counts.reviewed} remaining) ──[/bold cyan]"
        )
        if model is not None or dim_models:
            until_retrain = batch_size - decisions_since_retrain
            console.print(
                f"[dim]{until_retrain} decision{'s' if until_retrain != 1 else ''} until suggestion "
                f"models retrain (batch {batch_number}).[/dim]"
            )

        set_current_record(engine, reviewer, record["record_id"])
        already = answered_dimensions(engine, record["record_id"], reviewer)
        sample_membership = (
            sample_membership_by_record.get(record["record_id"]) if sample_membership_by_record else None
        )

        if "topical_inclusion" not in already:
            _print_record_screen(record)
            suggestion = ml_assist.suggest(model, record)
            outcome, used_suggestion = _run_topical_decision_loop_assisted(
                record, exclusion_reasons, reviewer, engine, sample_membership, suggestion
            )
            if outcome == "quit":
                console.print("[dim]Session saved. Resume anytime with the same --reviewer.[/dim]")
                return
            if outcome == "skip":
                skip_current_record(engine, reviewer)
                continue
            decision_label = outcome
            if used_suggestion is not None:
                ml_assist.log_suggestion(
                    engine,
                    record["record_id"],
                    reviewer,
                    suggested_label=used_suggestion.label,
                    suggested_probability=used_suggestion.probability,
                    training_size=used_suggestion.training_size,
                    final_label=decision_label,
                )
            batch_examples.append((ml_assist.record_text(record), decision_label))
            decisions_since_retrain += 1
            ml_model_metrics.save_pending_batch(batch_examples, DEFAULT_METRICS_PATH)
            if decisions_since_retrain >= batch_size:
                console.print(f"\n[bold cyan]── Retrain triggered -- batch {batch_number} ──[/bold cyan]")
                # Agreement first, model metrics second: this is the review's
                # actual reported outcome, whereas the model tables below are
                # about how good the suggestions are getting. Printing it here
                # also means it is on screen before the retrain prompts ask the
                # reviewer to pick a model, so a batch that already satisfies
                # the stopping rule is visible before they commit to another.
                _print_ai_agreement_panel(engine)
                # Every decision made toward this batch is already committed
                # to the DB (save_human_decision), so the full cumulative
                # example set here already includes it -- cumulative_metrics
                # fits every model ONCE on all of it and scores each by its
                # own out-of-fold cross-validation predictions from that fit
                # (see ml_model_metrics module docstring), not just this
                # batch's own ~batch_size decisions or a single random
                # holdout split of the cumulative set -- both were noisier.
                # The SAME fit becomes the new production `models` below --
                # no separate re-fit needed.
                cumulative_examples = ml_assist._labeled_examples(engine)
                cumulative_texts = [text for text, _label in cumulative_examples]
                cumulative_labels = [label for _text, label in cumulative_examples]
                evaluation = ml_model_metrics.cumulative_metrics(cumulative_texts, cumulative_labels)
                if evaluation is not None:
                    _print_model_metrics_table(
                        f"Batch {batch_number}: model performance (out-of-fold CV) over all "
                        f"{len(cumulative_examples)} decisions to date",
                        evaluation.metrics,
                        active_model_type,
                    )
                    metrics_history = ml_model_metrics.record_batch(
                        ml_model_metrics.BatchRecord(
                            batch=batch_number,
                            timestamp=datetime.now(UTC).isoformat(),
                            n_evaluated=evaluation.n,
                            n_trained_on=evaluation.n,
                            metrics=evaluation.metrics,
                            active_model=active_model_type,
                        ),
                        path=DEFAULT_METRICS_PATH,
                        plot_path=DEFAULT_METRICS_PLOT_PATH,
                    )
                    console.print(
                        f"[dim]Metrics written to {DEFAULT_METRICS_PATH}, plot updated at "
                        f"{DEFAULT_METRICS_PLOT_PATH}.[/dim]"
                    )

                # Dimension-suggestion models are evaluated the same
                # out-of-fold-CV way, but per dimension, then averaged
                # ACROSS the 6 dimensions into one aggregate score per model
                # type -- because one active model type drives all 6
                # dimensions' suggestions, not one choice per dimension (see
                # ml_model_metrics module docstring).
                dimension_evaluation = ml_model_metrics.cumulative_dimension_metrics(engine, schema)
                if dimension_evaluation is not None:
                    _print_model_metrics_table(
                        f"Batch {batch_number}: model performance averaged across the "
                        f"{len(dimension_evaluation.per_dimension_metrics)} scored classification dimension(s)",
                        dimension_evaluation.aggregate_metrics,
                        active_dimension_model_type,
                    )
                    _print_per_dimension_f1_table(
                        f"Batch {batch_number}: macro F1 per dimension (the breakdown behind the average above)",
                        dimension_evaluation.per_dimension_metrics,
                        active_dimension_model_type,
                    )
                    dimension_metrics_history = ml_model_metrics.record_dimension_batch(
                        ml_model_metrics.DimensionBatchRecord(
                            batch=batch_number,
                            timestamp=datetime.now(UTC).isoformat(),
                            per_dimension_metrics=dimension_evaluation.per_dimension_metrics,
                            aggregate_metrics=dimension_evaluation.aggregate_metrics,
                            active_dimension_model=active_dimension_model_type,
                        ),
                        path=DEFAULT_DIMENSION_METRICS_PATH,
                        plot_path=DEFAULT_DIMENSION_METRICS_PLOT_PATH,
                    )
                    console.print(
                        f"[dim]Dimension metrics written to {DEFAULT_DIMENSION_METRICS_PATH}, plot updated at "
                        f"{DEFAULT_DIMENSION_METRICS_PLOT_PATH}.[/dim]"
                    )
                _print_dimension_scoring_gaps(engine, schema, dimension_evaluation)

                # The models cumulative_metrics just fit (on all of
                # cumulative_texts/cumulative_labels, i.e. every decision up
                # to and including this batch) ARE the new production
                # models -- no separate re-fit needed.
                models = evaluation.models if evaluation is not None else None
                dimension_models_by_type = ml_assist.train_dimensions_all(engine, schema)
                decisions_since_retrain = 0
                batch_examples = []
                ml_model_metrics.save_pending_batch(batch_examples, DEFAULT_METRICS_PATH)
                batch_number += 1
                if models:
                    trained_on = next(iter(models.values())).training_size
                    console.print(f"[cyan]Retrained suggestion models on {trained_on} human decisions.[/cyan]")
                    if metrics_history and metrics_history[-1].metrics:
                        active_model_type = _prompt_active_model(metrics_history[-1].metrics, active_model_type)
                        ml_model_metrics.save_current_active_model(active_model_type, DEFAULT_METRICS_PATH)
                    model = models.get(active_model_type)
                    console.print(f"[cyan]Active suggestion model: {active_model_type}.[/cyan]")
                else:
                    model = None
                if dimension_models_by_type:
                    if dimension_metrics_history and dimension_metrics_history[-1].aggregate_metrics:
                        active_dimension_model_type = _prompt_active_model(
                            dimension_metrics_history[-1].aggregate_metrics,
                            active_dimension_model_type,
                            what="suggestions for all 6 classification dimensions",
                        )
                        ml_model_metrics.save_current_active_dimension_model(
                            active_dimension_model_type, DEFAULT_DIMENSION_METRICS_PATH
                        )
                    if active_dimension_model_type not in {
                        mt for dim_type_models in dimension_models_by_type.values() for mt in dim_type_models
                    }:
                        active_dimension_model_type = ml_assist.DEFAULT_ACTIVE_MODEL
                    dim_models = {
                        name: dim_type_models[active_dimension_model_type]
                        for name, dim_type_models in dimension_models_by_type.items()
                        if active_dimension_model_type in dim_type_models
                    }
                    missing = _missing_dimension_names(schema, dim_models)
                    console.print(
                        f"[cyan]Retrained dimension suggestion models ({active_dimension_model_type}) for: "
                        f"{', '.join(sorted(dim_models))}"
                        + (f" (still no model for: {', '.join(missing)} -- not enough labeled data).[/cyan]" if missing else " (all 6 dimensions covered).[/cyan]")
                    )
                else:
                    dim_models = {}
        else:
            decision_label = _get_saved_topical_label(engine, record["record_id"], reviewer)
            console.print(f"\n[dim]Resuming {record['title']!r} -- decision already recorded: {decision_label}.[/dim]")

        if decision_label != "include":
            finalize_record_review(engine, reviewer)
            continue

        if not _classify_dimensions(engine, record, reviewer, schema, already, dim_models=dim_models):
            return
        finalize_record_review(engine, reviewer)


@classify_app.command("ml-stats")
def ml_stats(
    recent_window: int = typer.Option(20, "--recent-window", help="Size of the 'recent' trailing window reported."),
) -> None:
    """Override-rate report for `classify assisted`'s suggestions so far --
    the evidence for deciding whether to extend ML assist from the
    required-review set to the full D1 pool."""
    engine = get_engine()
    stats = ml_assist.assist_stats(engine, recent_window=recent_window)
    if stats.total == 0:
        console.print("[yellow]No ML-assisted decisions logged yet -- run `classify assisted` first.[/yellow]")
        return

    overall_rate = stats.overrides / stats.total
    console.print(f"Suggestions shown and decided: {stats.total}")
    console.print(f"Overridden overall: {stats.overrides} ({overall_rate:.0%})")
    if stats.recent_total:
        recent_rate = stats.recent_overrides / stats.recent_total
        console.print(
            f"Last {stats.recent_total} decisions: {stats.recent_overrides} overridden ({recent_rate:.0%})"
        )

    dimensions = ml_assist.logged_dimensions(engine)
    if len(dimensions) > 1:
        table = RichTable(title="By item")
        table.add_column("Item")
        table.add_column("Shown", justify="right")
        table.add_column("Overridden", justify="right")
        table.add_column("Rate", justify="right")
        for dimension in dimensions:
            dim_stats = ml_assist.assist_stats(engine, recent_window=recent_window, dimension=dimension)
            rate = dim_stats.overrides / dim_stats.total if dim_stats.total else 0.0
            table.add_row(dimension, str(dim_stats.total), str(dim_stats.overrides), f"{rate:.0%}")
        console.print(table)


def _print_trend_table(title: str, model_types: tuple[str, ...], rows: list[tuple[int, str, dict[str, float]]]) -> None:
    """Shared by ml_metrics's two trend tables: `rows` is
    (batch, active_model, {model_type: f1_macro}) -- '-' for a model_type
    absent from that batch's metrics."""
    trend = RichTable(title=title)
    trend.add_column("Batch", justify="right")
    trend.add_column("Active model")
    for model_type in model_types:
        trend.add_column(model_type.replace("_", " "), justify="right")
    for batch, active_model, f1_by_type in rows:
        trend.add_row(
            str(batch),
            active_model,
            *(f"{f1_by_type[mt]:.0%}" if mt in f1_by_type else "-" for mt in model_types),
        )
    console.print(trend)


@classify_app.command("ml-metrics")
def ml_metrics() -> None:
    """Reports the per-batch, per-model-type metrics history for both
    suggestion systems (src/validation/ml_model_metrics.py): topical_inclusion
    and the 6 classification dimensions, each comparing the same eight model
    types (ml_assist.MODEL_SPECS: logistic regression, decision tree, random
    forest, and gradient boosting, each over TF-IDF and SPECTER-embedding
    features). topical_inclusion is fit once every retrain on every decision
    made to date and scored by its own out-of-fold cross-validation
    predictions from that fit (see cumulative_metrics); the 6 dimensions are
    scored the same way per dimension and then averaged ACROSS the 6 into
    one aggregate score per model type (cumulative_dimension_metrics), since
    a single active model type drives suggestions for all 6 dimensions at
    once. Read-only -- run
    `classify assisted` to add more batches live, or `classify
    ml-backfill-metrics` to reconstruct history from decisions already in the
    database."""
    history = ml_model_metrics.load_history(DEFAULT_METRICS_PATH)
    if not history:
        console.print(
            f"[yellow]No metrics history yet at {DEFAULT_METRICS_PATH} -- run `classify assisted` through at "
            f"least one retrain, or `classify ml-backfill-metrics` to reconstruct history from past decisions.[/yellow]"
        )
    else:
        latest = history[-1]
        _print_model_metrics_table(
            f"topical_inclusion -- batch {latest.batch}/{len(history)} (latest) -- active model: {latest.active_model}",
            latest.metrics,
            latest.active_model,
        )
        _print_trend_table(
            "topical_inclusion -- F1 (macro) by batch",
            ml_assist.MODEL_TYPES,
            [(b.batch, b.active_model, {mt: m.f1_macro for mt, m in b.metrics.items()}) for b in history],
        )
        console.print(f"[dim]Full history: {DEFAULT_METRICS_PATH} -- trend plot: {DEFAULT_METRICS_PLOT_PATH}[/dim]")

    dimension_history = ml_model_metrics.load_dimension_history(DEFAULT_DIMENSION_METRICS_PATH)
    if not dimension_history:
        console.print(
            f"[yellow]No dimension metrics history yet at {DEFAULT_DIMENSION_METRICS_PATH} -- run `classify "
            f"assisted` through at least one retrain with enough 'include' records classified, or `classify "
            f"ml-backfill-metrics` to reconstruct history from past decisions.[/yellow]"
        )
        return

    latest_dim = dimension_history[-1]
    _print_model_metrics_table(
        f"6-dimension average -- batch {latest_dim.batch}/{len(dimension_history)} (latest) -- "
        f"active model: {latest_dim.active_dimension_model}",
        latest_dim.aggregate_metrics,
        latest_dim.active_dimension_model,
    )
    _print_per_dimension_f1_table(
        f"6-dimension breakdown -- batch {latest_dim.batch} macro F1 per dimension",
        latest_dim.per_dimension_metrics,
        latest_dim.active_dimension_model,
    )
    _print_trend_table(
        "6-dimension average -- F1 (macro) by batch",
        ml_assist.MODEL_TYPES,
        [
            (b.batch, b.active_dimension_model, {mt: m.f1_macro for mt, m in b.aggregate_metrics.items()})
            for b in dimension_history
        ],
    )
    console.print(
        f"[dim]Full history: {DEFAULT_DIMENSION_METRICS_PATH} -- trend plot: {DEFAULT_DIMENSION_METRICS_PLOT_PATH}[/dim]"
    )


@classify_app.command("ml-backfill-metrics")
def ml_backfill_metrics(
    batch_size: int = typer.Option(
        20, "--batch-size", help="Same meaning as `classify assisted`'s --batch-size -- use the same value here "
        "to reconstruct history consistent with the batches a live session would have retrained on."
    ),
) -> None:
    """Reconstructs the full per-batch, per-model-type metrics history for
    BOTH suggestion systems -- topical_inclusion and the 6 classification
    dimensions -- from every human decision made so far (ordered by when it
    was actually decided), overwriting whatever's currently in
    ml_model_metrics.DEFAULT_METRICS_PATH/DEFAULT_METRICS_PLOT_PATH and
    DEFAULT_DIMENSION_METRICS_PATH/DEFAULT_DIMENSION_METRICS_PLOT_PATH. Use
    this once to backfill history for decisions made before this feature
    existed, or any time --batch-size needs to be corrected retroactively --
    safe to re-run, it always rebuilds from scratch rather than appending."""
    engine = get_engine()
    history = ml_model_metrics.backfill_batches(
        engine, batch_size=batch_size, path=DEFAULT_METRICS_PATH, plot_path=DEFAULT_METRICS_PLOT_PATH
    )
    if not history:
        console.print(
            f"[yellow]Not enough human topical_inclusion decisions yet to reconstruct even one batch "
            f"(need >={ml_assist.MIN_TOTAL_EXAMPLES} decisions across >=2 classes before the first batch "
            f"boundary).[/yellow]"
        )
    else:
        console.print(f"[green]Reconstructed {len(history)} batch(es) of topical_inclusion metrics "
                       f"(batch_size={batch_size}).[/green]")
        _print_model_metrics_table(
            f"topical_inclusion -- batch {history[-1].batch}/{len(history)} (latest, backfilled)",
            history[-1].metrics,
            history[-1].active_model,
        )
        console.print(f"[dim]Written to {DEFAULT_METRICS_PATH}, plot at {DEFAULT_METRICS_PLOT_PATH}.[/dim]")

    schema = load_classification_schema()
    dimension_history = ml_model_metrics.backfill_dimension_batches(
        engine, schema, batch_size=batch_size, path=DEFAULT_DIMENSION_METRICS_PATH, plot_path=DEFAULT_DIMENSION_METRICS_PLOT_PATH
    )
    if not dimension_history:
        console.print(
            "[yellow]Not enough classified 'include' records yet to reconstruct even one dimension batch.[/yellow]"
        )
        return

    console.print(
        f"[green]Reconstructed {len(dimension_history)} batch(es) of 6-dimension-average metrics "
        f"(batch_size={batch_size}).[/green]"
    )
    _print_model_metrics_table(
        f"6-dimension average -- batch {dimension_history[-1].batch}/{len(dimension_history)} (latest, backfilled)",
        dimension_history[-1].aggregate_metrics,
        dimension_history[-1].active_dimension_model,
    )
    console.print(
        f"[dim]Written to {DEFAULT_DIMENSION_METRICS_PATH}, plot at {DEFAULT_DIMENSION_METRICS_PLOT_PATH}.[/dim]"
    )


@dataclass
class ProgressCounts:
    d1_total: int
    reviewed: int
    included: int
    fully_classified: int
    sessions: list[dict]


def _progress_counts(engine, allowed_record_ids: set[str] | None = None) -> ProgressCounts:
    """Shared by the inline per-record banner in human() and the
    progress/watch commands, so both always agree on the same numbers.
    allowed_record_ids restricts d1_total (and everything derived from it) to
    a subset -- e.g. human()'s active validation sample -- instead of the
    full D1 pool; progress()/watch() leave it unset to report overall corpus
    status regardless of which sample is currently in use."""
    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    records_table = reflect_table(engine, "records")
    classifications_human = reflect_table(engine, "classifications_human")
    required_dims = {d.name for d in load_classification_schema()}

    with engine.connect() as conn:
        sessions = [dict(r) for r in conn.execute(reviewer_sessions.select()).mappings().all()]
        d1_ids = {
            row[0]
            for row in conn.execute(
                select(records_table.c.record_id).where(
                    records_table.c.d1 == 1, records_table.c.canonical_record_id.is_(None)
                )
            )
        }
        if allowed_record_ids is not None:
            d1_ids &= allowed_record_ids
        rows = conn.execute(
            select(
                classifications_human.c.record_id,
                classifications_human.c.dimension,
                classifications_human.c.label,
            )
        ).all()

    dims_by_record: dict[str, set[str]] = {}
    topical_by_record: dict[str, str] = {}
    for record_id, dimension, label in rows:
        dims_by_record.setdefault(record_id, set()).add(dimension)
        if dimension == "topical_inclusion":
            topical_by_record[record_id] = label

    reviewed = set(topical_by_record) & d1_ids
    included = {rid for rid in reviewed if topical_by_record[rid] == "include"}
    fully_classified = {rid for rid in included if required_dims <= dims_by_record.get(rid, set())}

    return ProgressCounts(
        d1_total=len(d1_ids),
        reviewed=len(reviewed),
        included=len(included),
        fully_classified=len(fully_classified),
        sessions=sessions,
    )


def _build_progress_view(engine) -> Group:
    counts = _progress_counts(engine)

    table = RichTable(title="Phase 11: human review progress")
    table.add_column("Reviewer")
    table.add_column("Completed")
    table.add_column("In progress")
    table.add_column("Last session")
    for s in counts.sessions:
        table.add_row(
            s["reviewer_id"], str(s["records_completed"]), s["current_record_id"] or "-", s["last_session_at"] or "-"
        )

    summary = Text.from_markup(
        f"D1 eligible: {counts.d1_total}\n"
        f"Reviewed (topical decision made): {counts.reviewed}\n"
        f"  ↳ included: {counts.included}\n"
        f"Fully dimension-classified: {counts.fully_classified}/{counts.included} of included"
    )
    return Group(table, summary)


@classify_app.command("progress")
def progress() -> None:
    """Per-reviewer progress + overall D1 vs human-reviewed counts, one-shot."""
    engine = get_engine()
    console.print(_build_progress_view(engine))


@classify_app.command("watch")
def watch(
    interval: float = typer.Option(2.0, "--interval", help="Refresh interval in seconds."),
) -> None:
    """Live-refreshing progress dashboard -- run in a second terminal while
    `classify human` runs elsewhere. Read-only; never writes. Ctrl-C to stop."""
    engine = get_engine()
    with Live(_build_progress_view(engine), console=console, refresh_per_second=4) as live:
        try:
            while True:
                time.sleep(interval)
                live.update(_build_progress_view(engine))
        except KeyboardInterrupt:
            pass
