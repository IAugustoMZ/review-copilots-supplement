from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from review_pipeline.src.cli import human_review_commands
from review_pipeline.src.cli.app import app
from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.validation import ml_assist, ml_model_metrics
from review_pipeline.src.validation.dimension_classification import (
    load_classification_schema,
    save_dimension_classification,
)

runner = CliRunner()


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _use_tmp_sample_paths(monkeypatch, tmp_path: Path) -> None:
    """Every test here builds a tiny in-memory D1 pool -- 'validation' sample
    mode (the human() default) would otherwise auto-build a sample file
    under the real project's outputs/ dir. Point it at tmp_path instead so
    tests never touch real repo state; since the pool is smaller than
    target_size, the auto-built sample still includes every inserted record,
    so existing assertions are unaffected."""
    monkeypatch.setattr(human_review_commands, "DEFAULT_SAMPLE_CSV_PATH", tmp_path / "validation_sample.csv")
    monkeypatch.setattr(human_review_commands, "DEFAULT_MANIFEST_PATH", tmp_path / "sampling_manifest.json")


def _use_tmp_metrics_paths(monkeypatch, tmp_path: Path) -> None:
    """Same idea as _use_tmp_sample_paths -- points both the topical_inclusion
    and the 6-dimension ml-assist metrics files/plots at tmp_path so tests
    never touch the real outputs/ dir."""
    monkeypatch.setattr(human_review_commands, "DEFAULT_METRICS_PATH", tmp_path / "ml_model_metrics.json")
    monkeypatch.setattr(human_review_commands, "DEFAULT_METRICS_PLOT_PATH", tmp_path / "ml_model_metrics.png")
    monkeypatch.setattr(human_review_commands, "DEFAULT_DIMENSION_METRICS_PATH", tmp_path / "ml_dimension_model_metrics.json")
    monkeypatch.setattr(human_review_commands, "DEFAULT_DIMENSION_METRICS_PLOT_PATH", tmp_path / "ml_dimension_model_metrics.png")


def _insert_d1_record(engine, record_id: str = "r1") -> str:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    title = "A Paper About GenAI Copilots"
    with engine.begin() as conn:
        conn.execute(
            records_table.insert().values(
                record_id=record_id,
                title=title,
                title_normalized=normalize_title(title),
                d1=1,
                created_at=now,
                updated_at=now,
            )
        )
    return record_id


# One multi-value dimension prompt = two lines (high, medium numbers) --
# low is auto-filled with whatever's left over, no separate low prompt.
# Schema order: domain, technology, application, architectural_role,
# validation_focus (all multi-value) then evidence_type (single-value).
def _dim_answers(*, pick_first_high: bool = True) -> list[str]:
    line = ["1", ""] if pick_first_high else ["", ""]
    return line * 5  # 5 multi-value dimensions


def test_progress_command_runs_against_temp_db(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _insert_d1_record(engine, "r1")
    _insert_d1_record(engine, "r2")

    result = runner.invoke(app, ["classify", "progress"])

    assert result.exit_code == 0
    assert "D1 eligible: 2" in result.stdout


def test_human_full_include_flow_writes_topical_and_all_dimensions(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    record_id = _insert_d1_record(engine)

    inputs = [
        "",  # nav prompt -> Enter to decide
        "include",
        "high",
        "on-topic industrial copilot",
        *_dim_answers(),  # domain, technology, application, architectural_role, validation_focus
        "1",  # evidence_type single pick
    ]
    result = runner.invoke(app, ["classify", "human", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    assert result.exit_code == 0, result.stdout

    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        rows = {
            row["dimension"]: row
            for row in conn.execute(classifications_human.select()).mappings().all()
        }
    assert set(rows) == {
        "topical_inclusion",
        "domain",
        "technology",
        "application",
        "architectural_role",
        "validation_focus",
        "evidence_type",
    }
    assert rows["topical_inclusion"]["label"] == "include"
    assert rows["evidence_type"]["label"] != ""

    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    with engine.connect() as conn:
        session = dict(conn.execute(reviewer_sessions.select()).mappings().one())
    assert session["records_completed"] == 1
    assert session["current_record_id"] is None
    assert session["reviewer_id"] == "alice"
    assert record_id in {r["record_id"] for r in [rows["topical_inclusion"]]}


def test_human_quit_mid_dimension_then_resume_skips_answered_dimensions(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine)

    # decision + domain fully answered, then quit during technology's medium prompt
    first_inputs = [
        "",
        "include",
        "high",
        "on-topic",
        "1", "",  # domain: high=[1], medium=[] -- low auto-filled with the rest; fully answered, saved
        "1",      # technology: high=[1]
        "q",      # quit during technology's medium prompt -- technology NOT saved
    ]
    first_result = runner.invoke(
        app, ["classify", "human", "--reviewer", "alice"], input="\n".join(first_inputs) + "\n"
    )
    assert first_result.exit_code == 0, first_result.stdout

    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        dims_after_quit = {
            row["dimension"] for row in conn.execute(classifications_human.select()).mappings().all()
        }
    assert dims_after_quit == {"topical_inclusion", "domain"}

    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    with engine.connect() as conn:
        session = dict(conn.execute(reviewer_sessions.select()).mappings().one())
    assert session["records_completed"] == 0
    assert session["current_record_id"] is not None  # record still in progress

    # resume: decision + domain are skipped entirely (no input consumed for them);
    # picks up at technology and finishes the rest
    second_inputs = [
        "1", "",  # technology
        "1", "",  # application
        "1", "",  # architectural_role
        "1", "",  # validation_focus
        "1",      # evidence_type
    ]
    second_result = runner.invoke(
        app, ["classify", "human", "--reviewer", "alice"], input="\n".join(second_inputs) + "\n"
    )
    assert second_result.exit_code == 0, second_result.stdout

    with engine.connect() as conn:
        dims_final = {row["dimension"] for row in conn.execute(classifications_human.select()).mappings().all()}
    assert dims_final == {
        "topical_inclusion",
        "domain",
        "technology",
        "application",
        "architectural_role",
        "validation_focus",
        "evidence_type",
    }
    with engine.connect() as conn:
        session = dict(conn.execute(reviewer_sessions.select()).mappings().one())
    assert session["records_completed"] == 1
    assert session["current_record_id"] is None


def test_human_defaults_to_auto_built_required_review_set(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine, "r1")

    inputs = ["", "exclude", "high", "not relevant", "1"]  # confidence, rationale, exclusion reason #1=off_topic
    result = runner.invoke(app, ["classify", "human", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    assert result.exit_code == 0, result.stdout
    assert "No validation sample found" in result.stdout
    assert "Required-review set" in result.stdout
    assert "required-review set" in result.stdout  # inline progress banner
    sample_csv = tmp_path / "validation_sample.csv"
    assert sample_csv.exists()

    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        row = dict(conn.execute(classifications_human.select()).mappings().one())
    assert row["sample_membership"] is not None


def test_human_sample_validation_excludes_mandatory_only_records(tmp_path: Path, monkeypatch) -> None:
    """A record that's only mandatory (AI-uncertain/non-high-confidence) and
    not in the stratified sample must not be pulled in under --sample
    validation -- that union only happens for the 'required' default."""
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine, "r1")

    inputs = ["", "exclude", "high", "not relevant", "1"]  # confidence, rationale, exclusion reason #1=off_topic
    result = runner.invoke(
        app, ["classify", "human", "--reviewer", "alice", "--sample", "validation"], input="\n".join(inputs) + "\n"
    )

    assert result.exit_code == 0, result.stdout
    assert "Required-review set" not in result.stdout
    assert "validation sample" in result.stdout


def test_human_sample_all_bypasses_validation_sample(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine, "r1")

    inputs = ["", "exclude", "high", "not relevant", "1"]  # confidence, rationale, exclusion reason #1=off_topic
    result = runner.invoke(
        app, ["classify", "human", "--reviewer", "alice", "--sample", "all"], input="\n".join(inputs) + "\n"
    )

    assert result.exit_code == 0, result.stdout
    assert "No validation sample found" not in result.stdout
    assert not (tmp_path / "validation_sample.csv").exists()

    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        row = dict(conn.execute(classifications_human.select()).mappings().one())
    assert row["sample_membership"] is None


def test_build_sample_command_writes_csv_and_manifest(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine, "r1")
    _insert_d1_record(engine, "r2")

    result = runner.invoke(app, ["classify", "build-sample", "--seed", "1", "--target-size", "10"])

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "validation_sample.csv").exists()
    assert (tmp_path / "sampling_manifest.json").exists()


def test_human_exclude_decision_skips_dimension_step(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine)

    inputs = ["", "exclude", "high", "not relevant", "1"]  # confidence, rationale, exclusion reason #1=off_topic
    result = runner.invoke(app, ["classify", "human", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    assert result.exit_code == 0, result.stdout
    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        dims = {row["dimension"] for row in conn.execute(classifications_human.select()).mappings().all()}
    assert dims == {"topical_inclusion"}

    reviewer_sessions = reflect_table(engine, "reviewer_sessions")
    with engine.connect() as conn:
        session = dict(conn.execute(reviewer_sessions.select()).mappings().one())
    assert session["records_completed"] == 1
    assert session["current_record_id"] is None


def test_confidence_prompt_accepts_single_letter_shortcut(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine)

    inputs = ["", "include", "h", "on-topic", *_dim_answers(), "1"]
    result = runner.invoke(app, ["classify", "human", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    assert result.exit_code == 0, result.stdout
    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        row = dict(
            conn.execute(
                classifications_human.select().where(classifications_human.c.dimension == "topical_inclusion")
            )
            .mappings()
            .one()
        )
    assert row["confidence"] == "high"


def test_rationale_prompt_offers_past_reasons_as_numbered_quick_pick(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine, "r1")
    _insert_d1_record(engine, "r2")

    first_inputs = ["", "exclude", "high", "not about GenAI copilots", "1"]
    runner.invoke(app, ["classify", "human", "--reviewer", "alice"], input="\n".join(first_inputs) + "\n")

    # "1" for rationale picks the just-used reason from the quick-pick menu instead of retyping it
    second_inputs = ["", "exclude", "high", "1", "1"]
    result = runner.invoke(app, ["classify", "human", "--reviewer", "alice"], input="\n".join(second_inputs) + "\n")

    assert result.exit_code == 0, result.stdout
    assert "most used for this decision" in result.stdout

    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        rows = conn.execute(
            classifications_human.select().where(classifications_human.c.dimension == "topical_inclusion")
        ).mappings().all()
    rationales = {row["record_id"]: row["rationale"] for row in rows}
    assert rationales["r2"] == "not about GenAI copilots"


def test_rationale_prompt_star_allows_typing_a_new_reason(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine, "r1")
    _insert_d1_record(engine, "r2")

    first_inputs = ["", "exclude", "high", "not about GenAI copilots", "1"]
    runner.invoke(app, ["classify", "human", "--reviewer", "alice"], input="\n".join(first_inputs) + "\n")

    second_inputs = ["", "exclude", "high", "*", "a brand new reason", "1"]
    result = runner.invoke(app, ["classify", "human", "--reviewer", "alice"], input="\n".join(second_inputs) + "\n")

    assert result.exit_code == 0, result.stdout
    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        rows = conn.execute(
            classifications_human.select().where(classifications_human.c.dimension == "topical_inclusion")
        ).mappings().all()
    rationales = {row["record_id"]: row["rationale"] for row in rows}
    assert rationales["r2"] == "a brand new reason"


def test_exclusion_reason_prompt_is_numbered(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine)

    inputs = ["", "exclude", "high", "wrong tech", "3"]  # #3 in screening_criteria.yaml = wrong_technology
    result = runner.invoke(app, ["classify", "human", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    assert result.exit_code == 0, result.stdout
    screening_decisions = reflect_table(engine, "screening_decisions")
    with engine.connect() as conn:
        row = dict(conn.execute(screening_decisions.select()).mappings().one())
    assert row["exclusion_reason"] == "wrong_technology"


def test_exclusion_reason_prompt_star_allows_custom_reason(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine)

    inputs = ["", "exclude", "high", "custom reason", "*", "duplicate of another record"]
    result = runner.invoke(app, ["classify", "human", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    assert result.exit_code == 0, result.stdout
    screening_decisions = reflect_table(engine, "screening_decisions")
    with engine.connect() as conn:
        row = dict(conn.execute(screening_decisions.select()).mappings().one())
    assert row["exclusion_reason"] == "duplicate of another record"


INCLUDE_TRAINING_TITLES = [
    "A large language model copilot for chemical process troubleshooting",
    "Generative AI agent for industrial plant operations support",
    "LLM-based assistant for distillation column monitoring",
    "Retrieval-augmented copilot for manufacturing knowledge transfer",
]
EXCLUDE_TRAINING_TITLES = [
    "A survey of medieval agricultural irrigation techniques",
    "Ceramic glaze chemistry in Bronze Age pottery",
    "Folk music preservation in rural communities",
    "History of cartography in the 18th century",
]


def _insert_d1_record_titled(engine, record_id: str, title: str) -> str:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            records_table.insert().values(
                record_id=record_id,
                title=title,
                title_normalized=normalize_title(title),
                d1=1,
                created_at=now,
                updated_at=now,
            )
        )
    return record_id


def _seed_ml_training_pool(engine) -> None:
    """Enough human-labeled, clearly-separable examples for
    ml_assist.train() to fit a usable model -- see ml_assist.MIN_TOTAL_EXAMPLES
    / MIN_EXAMPLES_PER_CLASS."""
    classifications_human = reflect_table(engine, "classifications_human")
    now = datetime.now(UTC).isoformat()
    for i, title in enumerate(INCLUDE_TRAINING_TITLES):
        rid = _insert_d1_record_titled(engine, f"train_inc{i}", title)
        with engine.begin() as conn:
            conn.execute(
                classifications_human.insert().values(
                    classification_id=new_id(),
                    record_id=rid,
                    dimension="topical_inclusion",
                    label="include",
                    confidence="high",
                    rationale="seed",
                    coder="seed",
                    timestamp=now,
                )
            )
    for i, title in enumerate(EXCLUDE_TRAINING_TITLES):
        rid = _insert_d1_record_titled(engine, f"train_exc{i}", title)
        with engine.begin() as conn:
            conn.execute(
                classifications_human.insert().values(
                    classification_id=new_id(),
                    record_id=rid,
                    dimension="topical_inclusion",
                    label="exclude",
                    confidence="high",
                    rationale="seed",
                    coder="seed",
                    timestamp=now,
                )
            )


def test_assisted_falls_back_to_unassisted_without_enough_training_data(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _use_tmp_metrics_paths(monkeypatch, tmp_path)
    _insert_d1_record(engine, "r1")

    inputs = ["", "exclude", "high", "not relevant", "1"]  # confidence, rationale, exclusion reason #1=off_topic
    result = runner.invoke(app, ["classify", "assisted", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    assert result.exit_code == 0, result.stdout
    assert "Not enough labeled data yet" in result.stdout
    assert "ML suggestion" not in result.stdout


def test_assisted_accept_suggestion_records_decision_with_no_override(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _use_tmp_metrics_paths(monkeypatch, tmp_path)
    _seed_ml_training_pool(engine)
    target = _insert_d1_record_titled(
        engine, "target1", "A generative AI copilot assistant for industrial plant reactor monitoring"
    )

    inputs = ["", "high", "", *_dim_answers(), "1"]  # accept suggestion, confidence, rationale default, dims
    result = runner.invoke(app, ["classify", "assisted", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    assert result.exit_code == 0, result.stdout
    assert "ML suggestion: include" in result.stdout

    ml_assist_suggestions = reflect_table(engine, "ml_assist_suggestions")
    with engine.connect() as conn:
        row = dict(conn.execute(ml_assist_suggestions.select()).mappings().one())
    assert row["record_id"] == target
    assert row["suggested_label"] == "include"
    assert row["final_label"] == "include"
    assert row["was_override"] == 0


def test_assisted_change_overrides_suggestion(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _use_tmp_metrics_paths(monkeypatch, tmp_path)
    _seed_ml_training_pool(engine)
    _insert_d1_record_titled(
        engine, "target1", "A generative AI copilot assistant for industrial plant reactor monitoring"
    )

    # training pool seeded "exclude" rationale history ("seed"), so the rationale
    # prompt now offers a numbered quick-pick -- "*" types a new one instead
    inputs = ["c", "exclude", "high", "*", "actually off-topic", "1"]
    result = runner.invoke(app, ["classify", "assisted", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    assert result.exit_code == 0, result.stdout

    ml_assist_suggestions = reflect_table(engine, "ml_assist_suggestions")
    with engine.connect() as conn:
        row = dict(conn.execute(ml_assist_suggestions.select()).mappings().one())
    assert row["suggested_label"] == "include"
    assert row["final_label"] == "exclude"
    assert row["was_override"] == 1


def test_ml_stats_reports_no_data_message(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)

    result = runner.invoke(app, ["classify", "ml-stats"])

    assert result.exit_code == 0, result.stdout
    assert "No ML-assisted decisions logged yet" in result.stdout


def test_ml_stats_reports_override_rate_after_assisted_session(tmp_path: Path, monkeypatch) -> None:
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _use_tmp_metrics_paths(monkeypatch, tmp_path)
    _seed_ml_training_pool(engine)
    _insert_d1_record_titled(
        engine, "target1", "A generative AI copilot assistant for industrial plant reactor monitoring"
    )
    inputs = ["", "high", "", *_dim_answers(), "1"]
    runner.invoke(app, ["classify", "assisted", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    result = runner.invoke(app, ["classify", "ml-stats"])

    assert result.exit_code == 0, result.stdout
    assert "Suggestions shown and decided: 1" in result.stdout
    assert "Overridden overall: 0" in result.stdout


# Extra domain-dimension examples layered on top of _seed_ml_training_pool's
# 4-per-side records, so each domain category clears
# ml_assist.MIN_POSITIVES_PER_CATEGORY (a category under that bar is skipped
# entirely and the dimension ends up with no model at all). Kept separate from
# INCLUDE_TRAINING_TITLES/EXCLUDE_TRAINING_TITLES so the topical_inclusion
# example counts those drive, which several tests assert on, stay unchanged.
EXTRA_DOMAIN_INCLUDE_TITLES = [
    "Conversational agent for reactor startup procedure guidance",
    "Foundation model assistant for process control loop tuning",
    "Copilot for refinery unit operations and fault diagnosis",
    "Language model support tool for chemical batch scheduling",
]
EXTRA_DOMAIN_EXCLUDE_TITLES = [
    "Baroque counterpoint in early keyboard repertoire",
    "Sediment stratigraphy of an inland lake basin",
    "Typography and legibility in road signage",
    "Migration patterns of coastal wading birds",
]


def _seed_domain_training(engine) -> None:
    """Domain-dimension labels over the same records/titles as
    _seed_ml_training_pool -- chemical_engineering rated high (safety low)
    for the copilot titles, safety rated high (chemical_engineering low) for
    the unrelated ones -- clearly separable enough for a per-category
    high/not-high classifier to learn from, plus EXTRA_DOMAIN_*_TITLES
    records to get each category over ml_assist.MIN_POSITIVES_PER_CATEGORY.

    Both categories have to appear at HIGH somewhere: the target is P(rated
    'high'), not category presence (ml_assist._parse_high_categories), so a
    category only ever rated medium gets no classifier and no suggestion.

    The extra records carry a topical_inclusion decision as well as their
    domain label, which is what keeps them out of the review queue -- same
    device _seed_domain_training_pool uses -- so the assisted-flow tests
    still meet the single target record they expect."""
    for i in range(len(INCLUDE_TRAINING_TITLES)):
        save_dimension_classification(
            engine, f"train_inc{i}", "seed", "domain", {"high": ["chemical_engineering"], "medium": [], "low": ["safety"]}
        )
    for i in range(len(EXCLUDE_TRAINING_TITLES)):
        save_dimension_classification(
            engine, f"train_exc{i}", "seed", "domain", {"high": ["safety"], "medium": [], "low": ["chemical_engineering"]}
        )

    classifications_human = reflect_table(engine, "classifications_human")
    now = datetime.now(UTC).isoformat()
    extras = [
        ("dom_extra_inc", EXTRA_DOMAIN_INCLUDE_TITLES, "include", {"high": ["chemical_engineering"], "medium": [], "low": ["safety"]}),
        ("dom_extra_exc", EXTRA_DOMAIN_EXCLUDE_TITLES, "exclude", {"high": ["safety"], "medium": [], "low": ["chemical_engineering"]}),
    ]
    for prefix, titles, topical_label, categories_by_level in extras:
        for i, title in enumerate(titles):
            rid = _insert_d1_record_titled(engine, f"{prefix}{i}", title)
            with engine.begin() as conn:
                conn.execute(
                    classifications_human.insert().values(
                        classification_id=new_id(), record_id=rid, dimension="topical_inclusion",
                        label=topical_label, confidence="high", rationale="seed", coder="seed", timestamp=now,
                    )
                )
            save_dimension_classification(engine, rid, "seed", "domain", categories_by_level)

    assert len(INCLUDE_TRAINING_TITLES) + len(EXTRA_DOMAIN_INCLUDE_TITLES) >= ml_assist.MIN_POSITIVES_PER_CATEGORY
    assert len(EXCLUDE_TRAINING_TITLES) + len(EXTRA_DOMAIN_EXCLUDE_TITLES) >= ml_assist.MIN_POSITIVES_PER_CATEGORY


def _seed_domain_training_pool(engine, n_per_category: int = 10) -> None:
    """n_per_category topically-decided (and thus queue-excluded) records
    rating domain=chemical_engineering high (include-side) / domain=safety
    high (exclude-side), each with the other category low -- larger than
    _seed_domain_training's 8 records so an 80/20 evaluation split
    (ml_model_metrics.cumulative_dimension_metrics) reliably leaves both
    classes represented for at least one category.

    The default also has to clear ml_assist.MIN_POSITIVES_PER_CATEGORY, since
    each of these categories is fit as its own high/not-high classifier and a
    category under that bar is skipped entirely -- with every category
    skipped there is no dimension model at all, and the assisted-flow tests
    below have nothing to assert against. Note "positives" counts records
    rating the category HIGH, not records mentioning it: every record here
    names both categories, and only the level distinguishes them
    (ml_assist._parse_high_categories)."""
    assert n_per_category >= ml_assist.MIN_POSITIVES_PER_CATEGORY
    classifications_human = reflect_table(engine, "classifications_human")
    now = datetime.now(UTC).isoformat()
    for i in range(n_per_category):
        title = f"{INCLUDE_TRAINING_TITLES[i % len(INCLUDE_TRAINING_TITLES)]} variant {i}"
        rid = _insert_d1_record_titled(engine, f"domain_inc{i}", title)
        with engine.begin() as conn:
            conn.execute(
                classifications_human.insert().values(
                    classification_id=new_id(), record_id=rid, dimension="topical_inclusion", label="include",
                    confidence="high", rationale="seed", coder="seed", timestamp=now,
                )
            )
        save_dimension_classification(engine, rid, "seed", "domain", {"high": ["chemical_engineering"], "medium": [], "low": ["safety"]})
    for i in range(n_per_category):
        title = f"{EXCLUDE_TRAINING_TITLES[i % len(EXCLUDE_TRAINING_TITLES)]} variant {i}"
        rid = _insert_d1_record_titled(engine, f"domain_exc{i}", title)
        with engine.begin() as conn:
            conn.execute(
                classifications_human.insert().values(
                    classification_id=new_id(), record_id=rid, dimension="topical_inclusion", label="exclude",
                    confidence="high", rationale="seed", coder="seed", timestamp=now,
                )
            )
        save_dimension_classification(engine, rid, "seed", "domain", {"high": ["safety"], "medium": [], "low": ["chemical_engineering"]})


def test_assisted_dimension_suggestion_accept_saves_bucketed_proposal(tmp_path: Path, monkeypatch) -> None:
    """Multi-value dimensions offer the same accept/override step as every
    other item: raw per-category probabilities are bucketed into one
    proposed high/medium/low classification (ml_assist.bucket_suggestion),
    which Enter accepts as a whole. The expected bucketing is computed
    directly from the same training data the CLI trains on, so this doesn't
    hardcode a specific probability outcome."""
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _use_tmp_metrics_paths(monkeypatch, tmp_path)
    _seed_ml_training_pool(engine)
    _seed_domain_training(engine)
    title = "A generative AI copilot assistant for industrial plant reactor monitoring"
    target = _insert_d1_record_titled(engine, "target1", title)

    schema = load_classification_schema()
    dim_models = ml_assist.train_dimensions(engine, schema)
    suggestion = ml_assist.suggest_dimension(dim_models["domain"], {"title": title, "abstract": ""})
    assert suggestion is not None
    expected_buckets = ml_assist.bucket_suggestion(suggestion)

    inputs = [
        "", "high", "",  # accept topical suggestion (include), confidence, rationale default
        "",  # accept the bucketed domain suggestion
        "1", "",  # technology -- no trained model, manual
        "1", "",  # application -- manual
        "1", "",  # architectural_role -- manual
        "1", "",  # validation_focus -- manual
        "1",  # evidence_type -- manual
    ]
    result = runner.invoke(app, ["classify", "assisted", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    assert result.exit_code == 0, result.stdout
    assert "ML suggestion -- domain" in result.stdout

    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        domain_row = dict(
            conn.execute(
                classifications_human.select().where(
                    classifications_human.c.dimension == "domain", classifications_human.c.record_id == target
                )
            ).mappings().one()
        )
    expected_categories = {c for cats in expected_buckets.values() for c in cats}
    assert set(json.loads(domain_row["label"])) == expected_categories

    ml_assist_suggestions = reflect_table(engine, "ml_assist_suggestions")
    with engine.connect() as conn:
        rows = conn.execute(
            ml_assist_suggestions.select().where(ml_assist_suggestions.c.dimension == "domain")
        ).mappings().all()
    assert len(rows) == 1
    suggested = json.loads(rows[0]["suggested_label"])
    assert set(suggested) == {"chemical_engineering", "safety"}  # every trained category, not just "present" ones
    assert json.loads(rows[0]["final_label"]) == expected_buckets
    assert rows[0]["was_override"] == 0


def test_assisted_dimension_suggestion_override_falls_back_to_manual(tmp_path: Path, monkeypatch) -> None:
    """Rejecting the bucketed domain suggestion (c) drops into the normal
    manual high/medium/low prompt from scratch, same as before -- and
    was_override now records the literal accept/override action rather than
    a category-overlap heuristic."""
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _use_tmp_metrics_paths(monkeypatch, tmp_path)
    _seed_ml_training_pool(engine)
    _seed_domain_training(engine)
    _insert_d1_record_titled(
        engine, "target1", "A generative AI copilot assistant for industrial plant reactor monitoring"
    )

    inputs = [
        "", "high", "",  # accept topical suggestion
        "c",  # reject the bucketed domain suggestion
        "2", "",  # domain -- manual: high=[2] ("process_systems_engineering"), disagreeing with the model
        "1", "",  # technology
        "1", "",  # application
        "1", "",  # architectural_role
        "1", "",  # validation_focus
        "1",  # evidence_type
    ]
    result = runner.invoke(app, ["classify", "assisted", "--reviewer", "alice"], input="\n".join(inputs) + "\n")

    assert result.exit_code == 0, result.stdout

    ml_assist_suggestions = reflect_table(engine, "ml_assist_suggestions")
    with engine.connect() as conn:
        row = dict(
            conn.execute(
                ml_assist_suggestions.select().where(ml_assist_suggestions.c.dimension == "domain")
            ).mappings().one()
        )
    # low is auto-filled with every domain category not picked for high/medium
    schema = load_classification_schema()
    domain_categories = next(dim.categories for dim in schema if dim.name == "domain")
    expected_low = [c for c in domain_categories if c != "process_systems_engineering"]
    assert json.loads(row["final_label"]) == {
        "high": ["process_systems_engineering"],
        "medium": [],
        "low": expected_low,
    }
    assert row["was_override"] == 1


def test_assisted_retrain_prints_batch_shows_metrics_and_lets_reviewer_switch_model(
    tmp_path: Path, monkeypatch
) -> None:
    """With --batch-size 1, the single topical decision made this session
    immediately crosses the retrain boundary: the batch number is printed,
    every model type (ml_assist.MODEL_SPECS) is scored against that decision
    and shown, the metrics file/plot are written, and picking a different
    model at the prompt becomes the active suggestion model."""
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _use_tmp_metrics_paths(monkeypatch, tmp_path)
    _seed_ml_training_pool(engine)
    _insert_d1_record_titled(
        engine, "target1", "A generative AI copilot assistant for industrial plant reactor monitoring"
    )

    inputs = [
        "", "high", "",        # accept topical suggestion (include), confidence, rationale default
        "2",                   # retrain prompt: pick model #2 (tfidf_decision_tree)
        *_dim_answers(), "1",  # dimension classification -- no dimension models trained yet, manual
    ]
    result = runner.invoke(
        app, ["classify", "assisted", "--reviewer", "alice", "--batch-size", "1"], input="\n".join(inputs) + "\n"
    )

    assert result.exit_code == 0, result.stdout
    assert "Retrain triggered -- batch 1" in result.stdout
    assert "model performance (out-of-fold CV) over all 9 decisions to date" in result.stdout
    # The reviewer-choice prompt lists every scored model type by its
    # underscored name -- checked here rather than the metrics table's
    # printed labels, which Rich word-wraps across cells at test-runner
    # console widths (e.g. "logistic" / "regression" on separate lines).
    for model_type in ml_assist.MODEL_TYPES:
        assert model_type in result.stdout
    assert "Active suggestion model: tfidf_decision_tree." in result.stdout

    metrics_path = tmp_path / "ml_model_metrics.json"
    plot_path = tmp_path / "ml_model_metrics.png"
    assert metrics_path.exists()
    assert plot_path.exists()

    history = ml_model_metrics.load_history(metrics_path)
    assert len(history) == 1
    assert history[0].batch == 1
    assert set(history[0].metrics) == set(ml_assist.MODEL_SPECS)
    # active_model on the recorded batch is what was active DURING it (the
    # default), not the reviewer's new pick -- that pick only takes effect
    # starting with the next batch.
    assert history[0].active_model == ml_assist.DEFAULT_ACTIVE_MODEL
    assert ml_model_metrics.load_current_active_model(metrics_path) == "tfidf_decision_tree"


def test_assisted_retrain_also_shows_dimension_model_table_and_prompts_for_a_dimension_model(
    tmp_path: Path, monkeypatch
) -> None:
    """When there's enough dimension-classified data (here: 'domain'), the
    retrain trigger also evaluates the 6-dimension model comparison
    (averaged across however many dimensions currently have enough data)
    and prompts for a SEPARATE active model choice applied uniformly to all
    6 dimensions -- independent of the topical_inclusion choice."""
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _use_tmp_metrics_paths(monkeypatch, tmp_path)
    _seed_ml_training_pool(engine)
    _seed_domain_training_pool(engine)
    _insert_d1_record_titled(engine, "target1", "A survey of ancient irrigation systems in Mesopotamia")

    inputs = [
        "c", "exclude", "high", "*", "manually excluded",  # override topical suggestion to exclude
        "1",  # exclusion reason -- excluding skips dimension classification entirely
        "2",  # topical retrain prompt: pick model #2 (tfidf_decision_tree)
        "3",  # dimension retrain prompt: pick model #3 (tfidf_random_forest)
    ]
    result = runner.invoke(
        app, ["classify", "assisted", "--reviewer", "alice", "--batch-size", "1"], input="\n".join(inputs) + "\n"
    )

    assert result.exit_code == 0, result.stdout
    assert "Retrain triggered -- batch 1" in result.stdout
    assert "model performance averaged across the" in result.stdout
    assert "Active suggestion model: tfidf_decision_tree." in result.stdout

    dim_path = tmp_path / "ml_dimension_model_metrics.json"
    assert dim_path.exists()
    assert (tmp_path / "ml_dimension_model_metrics.png").exists()

    dim_history = ml_model_metrics.load_dimension_history(dim_path)
    assert len(dim_history) == 1
    assert "domain" in dim_history[0].per_dimension_metrics
    assert dim_history[0].active_dimension_model == ml_assist.DEFAULT_ACTIVE_MODEL


def test_assisted_persists_chosen_active_model_across_sessions(tmp_path: Path, monkeypatch) -> None:
    """A model choice made at one retrain must still be the starting active
    model the NEXT `classify assisted` invocation opens with -- even though
    that choice is a different process, it's read back from the metrics file
    (ml_model_metrics.load_current_active_model), not just held in memory."""
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _use_tmp_metrics_paths(monkeypatch, tmp_path)
    _seed_ml_training_pool(engine)
    _insert_d1_record_titled(
        engine, "target1", "A generative AI copilot assistant for industrial plant reactor monitoring"
    )

    first_inputs = ["", "high", "", "3", *_dim_answers(), "1"]  # "3" = tfidf_random_forest at the retrain prompt
    first_result = runner.invoke(
        app, ["classify", "assisted", "--reviewer", "alice", "--batch-size", "1"],
        input="\n".join(first_inputs) + "\n",
    )
    assert first_result.exit_code == 0, first_result.stdout
    assert ml_model_metrics.load_current_active_model(tmp_path / "ml_model_metrics.json") == "tfidf_random_forest"

    _insert_d1_record_titled(
        engine, "target2", "Another generative AI copilot for chemical plant safety monitoring"
    )
    second_result = runner.invoke(
        app, ["classify", "assisted", "--reviewer", "alice", "--batch-size", "1"], input="q\n"
    )
    assert second_result.exit_code == 0, second_result.stdout
    assert "Suggestion model (tfidf_random_forest)" in second_result.stdout


def test_assisted_persists_chosen_active_dimension_model_across_sessions(tmp_path: Path, monkeypatch) -> None:
    """Same guarantee as test_assisted_persists_chosen_active_model_across_sessions,
    for the SEPARATE dimension-model choice (ml_model_metrics.
    load_current_active_dimension_model) -- independent file/key from the
    topical_inclusion one."""
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _use_tmp_metrics_paths(monkeypatch, tmp_path)
    _seed_ml_training_pool(engine)
    _seed_domain_training_pool(engine)
    _insert_d1_record_titled(engine, "target1", "A survey of ancient irrigation systems in Mesopotamia")

    first_inputs = [
        "c", "exclude", "high", "*", "manually excluded", "1",  # override to exclude -- skips dimension classification
        "2",  # topical retrain prompt
        "3",  # dimension retrain prompt: 3rd scored model type
    ]
    first_result = runner.invoke(
        app, ["classify", "assisted", "--reviewer", "alice", "--batch-size", "1"],
        input="\n".join(first_inputs) + "\n",
    )
    assert first_result.exit_code == 0, first_result.stdout
    dim_path = tmp_path / "ml_dimension_model_metrics.json"
    chosen = ml_model_metrics.load_current_active_dimension_model(dim_path)
    assert chosen is not None

    _insert_d1_record_titled(engine, "target2", "Ceramic glaze techniques of the Bronze Age Aegean")
    second_result = runner.invoke(
        app, ["classify", "assisted", "--reviewer", "alice", "--batch-size", "1"], input="q\n"
    )
    assert second_result.exit_code == 0, second_result.stdout
    assert f"Dimension suggestion models ({chosen})" in second_result.stdout
    assert ml_model_metrics.load_current_active_dimension_model(dim_path) == chosen


def test_assisted_batch_progress_survives_abort_and_counts_toward_next_retrain(tmp_path: Path, monkeypatch) -> None:
    """Regression test: aborting `classify assisted` mid-batch (typing 'q')
    must not reset the decisions-until-retrain countdown, and the eventual
    retrain must be scored against every decision made toward that batch --
    including ones made in a session that was aborted before the batch
    actually completed. The decision itself was never at risk (save_human_
    decision commits immediately) -- this is about the batch/retrain
    bookkeeping, which used to live only in assisted()'s local variables and
    silently reset to empty on every new invocation."""
    engine = _db(tmp_path)
    monkeypatch.setattr(human_review_commands, "get_engine", lambda: engine)
    _use_tmp_sample_paths(monkeypatch, tmp_path)
    _use_tmp_metrics_paths(monkeypatch, tmp_path)
    _seed_ml_training_pool(engine)
    target1 = _insert_d1_record_titled(engine, "target1", "A survey of ancient irrigation systems in Mesopotamia")
    _insert_d1_record_titled(engine, "target2", "Ceramic glaze techniques of the Bronze Age Aegean")

    # Session 1, --batch-size 2: accept the (exclude) suggestion for target1
    # -- an exclude decision skips dimension classification, so the queue
    # moves straight to target2's nav prompt, where 'q' aborts the session
    # before the batch (2 decisions) actually completes.
    first_inputs = ["", "high", "", "1", "q"]  # accept suggestion, confidence, rationale default, excl. reason, quit
    first_result = runner.invoke(
        app, ["classify", "assisted", "--reviewer", "alice", "--batch-size", "2"],
        input="\n".join(first_inputs) + "\n",
    )
    assert first_result.exit_code == 0, first_result.stdout
    assert "ML suggestion: exclude" in first_result.stdout
    assert "Retrain triggered" not in first_result.stdout

    classifications_human = reflect_table(engine, "classifications_human")
    with engine.connect() as conn:
        rows = conn.execute(
            classifications_human.select().where(classifications_human.c.record_id == target1)
        ).mappings().all()
    assert len(rows) == 1  # target1's decision made it to the DB despite the abort

    # A fresh session must pick up that 1 decision, not reset to a fresh
    # 2-decision countdown.
    probe_result = runner.invoke(
        app, ["classify", "assisted", "--reviewer", "alice", "--batch-size", "2"], input="q\n"
    )
    assert probe_result.exit_code == 0, probe_result.stdout
    assert "1 decision until suggestion models retrain (batch 1)" in probe_result.stdout

    # Session 2: one more decision (target2) crosses the batch-size-2
    # boundary -- the retrain's cumulative metrics must reflect BOTH
    # decisions made toward this batch (target1 AND target2), not just the
    # one made in this session -- proof that the aborted session 1's
    # decision wasn't dropped from the countdown or the eventual evaluation.
    second_inputs = ["", "high", "", "1", ""]  # accept suggestion, confidence, rationale, excl. reason, keep model
    second_result = runner.invoke(
        app, ["classify", "assisted", "--reviewer", "alice", "--batch-size", "2"],
        input="\n".join(second_inputs) + "\n",
    )
    assert second_result.exit_code == 0, second_result.stdout
    assert "Retrain triggered -- batch 1" in second_result.stdout

    history = ml_model_metrics.load_history(tmp_path / "ml_model_metrics.json")
    assert len(history) == 1
    # seed pool (8) + target1 + target2 = 10 decisions -- every example both
    # trains (in some CV folds) and is evaluated (in its own held-out fold),
    # so n_trained_on == n_evaluated == the full cumulative count.
    expected_total = len(INCLUDE_TRAINING_TITLES) + len(EXCLUDE_TRAINING_TITLES) + 2
    assert history[0].n_trained_on == expected_total
    assert history[0].n_evaluated == expected_total
    assert ml_model_metrics.load_pending_batch(tmp_path / "ml_model_metrics.json") == []
