from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from review_pipeline.src.cli import human_review_commands
from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record(engine, record_id: str, title: str = "A Paper") -> None:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            records_table.insert().values(
                record_id=record_id,
                title=title,
                title_normalized=normalize_title(title),
                year=2025,
                d1=1,
                created_at=now,
                updated_at=now,
            )
        )


def _label(engine, table_name: str, record_id: str, label: str, dimension: str = "topical_inclusion") -> None:
    """Both classification tables name the rater column `coder` -- a model name
    on the AI side, a reviewer id on the human side."""
    table = reflect_table(engine, table_name)
    with engine.begin() as conn:
        conn.execute(
            table.insert().values(
                classification_id=new_id(),
                record_id=record_id,
                dimension=dimension,
                label=label,
                rationale="test",
                coder="test-model" if table_name == "classifications_ai" else "tester",
                timestamp=datetime.now(UTC).isoformat(),
            )
        )


def _pair(engine, record_id: str, human: str, ai: str) -> None:
    _insert_record(engine, record_id)
    _label(engine, "classifications_human", record_id, human)
    _label(engine, "classifications_ai", record_id, ai)


def test_panel_reports_nothing_to_compare_before_any_paired_decision(tmp_path: Path, capsys) -> None:
    engine = _db(tmp_path)
    human_review_commands._print_ai_agreement_panel(engine)
    assert "No AI-human paired decisions yet" in capsys.readouterr().out


def test_panel_reports_kappa_and_interval_on_high_but_imperfect_agreement(tmp_path: Path, capsys) -> None:
    engine = _db(tmp_path)
    for i in range(40):
        _pair(engine, f"inc{i}", human="include", ai="include")
    for i in range(160):
        _pair(engine, f"exc{i}", human="exclude", ai="exclude")
    for i in range(6):  # a few disagreements, so kappa stays inside (-1, 1)
        _pair(engine, f"dis{i}", human="include", ai="exclude")

    human_review_commands._print_ai_agreement_panel(engine)
    out = capsys.readouterr().out

    assert "Cohen's kappa" in out
    assert "95% CI" in out
    assert "Observed include rate" in out
    assert "Stopping rule met" in out


def test_panel_omits_the_interval_at_perfect_agreement(tmp_path: Path, capsys) -> None:
    """kappa == 1 is a boundary the Bloch-Kraemer variance is not defined at.
    The panel must degrade to reporting kappa alone rather than raising or
    printing a nonsense interval."""
    engine = _db(tmp_path)
    for i in range(12):
        _pair(engine, f"r{i}", human="include", ai="include")
    for i in range(12, 30):
        _pair(engine, f"r{i}", human="exclude", ai="exclude")

    human_review_commands._print_ai_agreement_panel(engine)
    out = capsys.readouterr().out

    assert "1.000" in out
    assert "95% CI" not in out


def test_panel_flags_the_stopping_rule_as_unmet_on_a_small_disagreeing_sample(tmp_path: Path, capsys) -> None:
    engine = _db(tmp_path)
    # Deliberately noisy and small -- the interval cannot clear the floor here.
    for i in range(5):
        _pair(engine, f"a{i}", human="include", ai="include")
    for i in range(5):
        _pair(engine, f"b{i}", human="include", ai="exclude")
    for i in range(10):
        _pair(engine, f"c{i}", human="exclude", ai="exclude")

    human_review_commands._print_ai_agreement_panel(engine)
    out = capsys.readouterr().out
    assert "Stopping rule not met yet" in out


def test_panel_only_counts_records_carrying_both_labels(tmp_path: Path, capsys) -> None:
    """A record the human decided but the AI never screened (or vice versa)
    must not inflate n -- agreement is undefined for it."""
    engine = _db(tmp_path)
    for i in range(6):
        _pair(engine, f"both{i}", human="include", ai="include")

    _insert_record(engine, "human_only")
    _label(engine, "classifications_human", "human_only", "include")
    _insert_record(engine, "ai_only")
    _label(engine, "classifications_ai", "ai_only", "exclude")

    human_review_commands._print_ai_agreement_panel(engine)
    out = capsys.readouterr().out
    assert "6" in out
    assert "8" not in out.split("Paired decisions")[1].split("\n")[0]


def test_panel_says_so_when_the_marginals_put_the_floor_out_of_reach(tmp_path: Path, capsys) -> None:
    """Phase 11's shape: the AI includes far more often than the human does, so
    kappa's ceiling sits below the moderate-agreement floor. Telling the
    reviewer "not met yet" would be telling them to review forever (D-36)."""
    engine = _db(tmp_path)
    for i in range(2):
        _pair(engine, f"tp{i}", human="include", ai="include")
    for i in range(16):
        _pair(engine, f"fp{i}", human="exclude", ai="include")
    for i in range(2):
        _pair(engine, f"tn{i}", human="exclude", ai="exclude")

    human_review_commands._print_ai_agreement_panel(engine)
    out = capsys.readouterr().out

    assert "unreachable" in out
    assert "Stopping rule not met yet" not in out
    assert "kappa_max" in out


def test_panel_shows_kappa_against_its_ceiling(tmp_path: Path, capsys) -> None:
    engine = _db(tmp_path)
    for i in range(5):
        _pair(engine, f"a{i}", human="include", ai="include")
    for i in range(5):
        _pair(engine, f"b{i}", human="include", ai="exclude")
    for i in range(10):
        _pair(engine, f"c{i}", human="exclude", ai="exclude")

    human_review_commands._print_ai_agreement_panel(engine)
    out = capsys.readouterr().out

    assert "kappa_max" in out
    assert "of it reached" in out
