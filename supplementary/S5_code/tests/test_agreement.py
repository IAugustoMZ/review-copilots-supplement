from __future__ import annotations

import json

import pytest
from datetime import UTC, datetime
from pathlib import Path

from review_pipeline.src.common.ids import new_id
from review_pipeline.src.common.text_normalize import normalize_title
from review_pipeline.src.db.engine import get_engine, run_migrations
from review_pipeline.src.db.tables import reflect_table
from review_pipeline.src.validation.agreement import (
    DimensionAgreement,
    PairedDecision,
    compute_dimension_agreement,
    multi_label_agreement,
    paired_decisions,
    single_label_agreement,
    write_agreement_csv,
    write_agreement_report,
)


def _db(tmp_path: Path):
    engine = get_engine(tmp_path / "review.db")
    run_migrations(engine)
    return engine


def _insert_record(engine, record_id: str) -> None:
    records_table = reflect_table(engine, "records")
    now = datetime.now(UTC).isoformat()
    title = f"Paper {record_id}"
    with engine.begin() as conn:
        conn.execute(
            records_table.insert().values(
                record_id=record_id,
                title=title,
                title_normalized=normalize_title(title),
                publication_status="peer_reviewed_journal",
                year=2025,
                d1=1,
                created_at=now,
                updated_at=now,
            )
        )


def _insert_classification(engine, table_name: str, record_id: str, dimension: str, label: str, coder: str) -> None:
    table = reflect_table(engine, table_name)
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            table.insert().values(
                classification_id=new_id(),
                record_id=record_id,
                dimension=dimension,
                label=label,
                confidence="high",
                coder=coder,
                timestamp=now,
            )
        )


# --- single-label -----------------------------------------------------------


def test_single_label_agreement_percent_and_kappa() -> None:
    pairs = [
        PairedDecision("r1", ai_label="include", human_label="include"),
        PairedDecision("r2", ai_label="exclude", human_label="exclude"),
        PairedDecision("r3", ai_label="include", human_label="exclude"),
        PairedDecision("r4", ai_label="uncertain", human_label="uncertain"),
    ]
    result = single_label_agreement(pairs)
    assert result.n == 4
    assert result.percent_agreement == 0.75
    assert result.cohen_kappa is not None


def test_single_label_agreement_false_negative_rate() -> None:
    pairs = [
        PairedDecision("r1", ai_label="include", human_label="include"),  # true positive
        PairedDecision("r2", ai_label="exclude", human_label="include"),  # FALSE NEGATIVE
        PairedDecision("r3", ai_label="uncertain", human_label="include"),  # FALSE NEGATIVE
        PairedDecision("r4", ai_label="exclude", human_label="exclude"),  # true negative, irrelevant to FN rate
    ]
    result = single_label_agreement(pairs)
    assert result.false_negative_rate == 2 / 3
    assert set(result.false_negatives) == {"r2", "r3"}


def test_single_label_agreement_confusion_matrix_shape() -> None:
    pairs = [
        PairedDecision("r1", ai_label="include", human_label="include"),
        PairedDecision("r2", ai_label="exclude", human_label="include"),
    ]
    result = single_label_agreement(pairs)
    assert result.confusion_matrix["include"]["include"] == 1
    assert result.confusion_matrix["include"]["exclude"] == 1


def test_single_label_agreement_empty_pairs() -> None:
    result = single_label_agreement([])
    assert result.n == 0
    assert result.cohen_kappa is None
    assert result.false_negative_rate is None


def test_single_label_agreement_no_human_include_cases_gives_none_fn_rate() -> None:
    pairs = [PairedDecision("r1", ai_label="exclude", human_label="exclude")]
    result = single_label_agreement(pairs)
    assert result.false_negative_rate is None


# --- multi-label --------------------------------------------------------


def test_multi_label_agreement_jaccard_and_f1() -> None:
    pairs = [
        PairedDecision("r1", ai_label=json.dumps(["domain_A", "domain_B"]), human_label=json.dumps(["domain_A"])),
        PairedDecision("r2", ai_label=json.dumps(["domain_B"]), human_label=json.dumps(["domain_B"])),
    ]
    result = multi_label_agreement(pairs)
    assert result.n == 2
    assert 0 < result.mean_jaccard < 1
    assert "domain_A" in result.per_label
    assert "domain_B" in result.per_label


def test_multi_label_agreement_identical_sets_is_perfect() -> None:
    pairs = [PairedDecision("r1", ai_label=json.dumps(["x", "y"]), human_label=json.dumps(["x", "y"]))]
    result = multi_label_agreement(pairs)
    assert result.mean_jaccard == 1.0
    assert result.micro_f1 == 1.0


# --- paired_decisions / compute_dimension_agreement (real DB) --------------


def test_paired_decisions_only_includes_records_with_both_labels(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    for rid in ("r1", "r2", "r3"):
        _insert_record(engine, rid)
    _insert_classification(engine, "classifications_ai", "r1", "topical_inclusion", "include", "model")
    _insert_classification(engine, "classifications_human", "r1", "topical_inclusion", "exclude", "alice")
    _insert_classification(engine, "classifications_ai", "r2", "topical_inclusion", "include", "model")
    # r2 has no human label yet; r3 has neither

    pairs = paired_decisions(engine, "topical_inclusion")
    assert len(pairs) == 1
    assert pairs[0].record_id == "r1"
    assert pairs[0].ai_label == "include"
    assert pairs[0].human_label == "exclude"


def test_compute_dimension_agreement_detects_single_label(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, "r1")
    _insert_classification(engine, "classifications_ai", "r1", "topical_inclusion", "include", "model")
    _insert_classification(engine, "classifications_human", "r1", "topical_inclusion", "include", "alice")

    result = compute_dimension_agreement(engine, "topical_inclusion")
    assert result.kind == "single_label"
    assert result.single.percent_agreement == 1.0


def test_compute_dimension_agreement_detects_multi_label(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, "r1")
    _insert_classification(engine, "classifications_ai", "r1", "domain", json.dumps(["a", "b"]), "model")
    _insert_classification(engine, "classifications_human", "r1", "domain", json.dumps(["a"]), "alice")

    result = compute_dimension_agreement(engine, "domain")
    assert result.kind == "multi_label"
    assert result.multi.n == 1


def test_compute_dimension_agreement_no_data_yet(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    result = compute_dimension_agreement(engine, "topical_inclusion")
    assert result.kind == "single_label"
    assert result.single.n == 0


# --- report writers -----------------------------------------------------


def test_write_agreement_csv_and_report_do_not_crash_on_empty_and_real_data(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, "r1")
    _insert_classification(engine, "classifications_ai", "r1", "topical_inclusion", "exclude", "model")
    _insert_classification(engine, "classifications_human", "r1", "topical_inclusion", "include", "alice")

    result = compute_dimension_agreement(engine, "topical_inclusion")
    csv_path = tmp_path / "out.csv"
    report_path = tmp_path / "out.md"
    write_agreement_csv([result], csv_path)
    write_agreement_report([result], report_path)

    csv_text = csv_path.read_text()
    assert "false_negative_rate" in csv_text
    report_text = report_path.read_text()
    assert "AI-Human Agreement" in report_text
    assert "inter-rater reliability" in report_text.lower()  # the qualifying note itself
    assert "r1" in report_text  # the false negative is called out with its record_id


# --- design-weighted estimation (D-29) --------------------------------------


def _pairs_with_weights() -> list[PairedDecision]:
    """A deliberately skewed sample: the include stratum is over-sampled 5x and
    the exclude stratum under-sampled 50x, as in the real Phase 11 design."""
    pairs = []
    # 8 AI-includes, 6 of them confirmed by the human -> sample precision 75%.
    for i in range(8):
        pairs.append(
            PairedDecision(
                record_id=f"inc-{i}",
                ai_label="include",
                human_label="include" if i < 6 else "exclude",
                stratum="ai_include",
                weight=5.3,
            )
        )
    # 4 AI-excludes, 2 of which the human would have included. The two strata
    # deliberately have DIFFERENT agreement rates (6/8 vs 2/4) -- otherwise the
    # weighted and unweighted figures coincide by construction and the test
    # proves nothing.
    for i in range(4):
        pairs.append(
            PairedDecision(
                record_id=f"exc-{i}",
                ai_label="exclude",
                human_label="include" if i < 2 else "exclude",
                stratum="ai_exclude",
                weight=50.3,
            )
        )
    return pairs


def test_weighted_and_unweighted_agreement_differ_under_unequal_weights() -> None:
    result = single_label_agreement(_pairs_with_weights())

    assert result.is_weighted is True
    # 8 of 12 rows agree.
    assert result.percent_agreement == 8 / 12
    expected_weighted = (6 * 5.3 + 2 * 50.3) / (8 * 5.3 + 4 * 50.3)
    assert result.weighted_percent_agreement == pytest.approx(expected_weighted)
    assert result.weighted_percent_agreement != pytest.approx(result.percent_agreement)


def test_weighted_precision_of_the_include_label() -> None:
    result = single_label_agreement(_pairs_with_weights())

    # Every AI-include carries the same weight, so weighted precision matches
    # the sample precision here -- the point is that it is computed on weights.
    assert result.weighted_precision == pytest.approx(6 / 8)


def test_weighted_false_negative_rate_is_dominated_by_the_exclude_stratum() -> None:
    result = single_label_agreement(_pairs_with_weights())

    # 8 human-includes in the sample, 2 missed -> unweighted 2/8.
    assert result.false_negative_rate == pytest.approx(2 / 8)
    # Weighted, each miss stands for 50.3 corpus records against the 5.3 each
    # caught include stands for, so the rate is far higher.
    expected = (2 * 50.3) / (6 * 5.3 + 2 * 50.3)
    assert result.weighted_false_negative_rate == pytest.approx(expected)
    assert result.weighted_false_negative_rate > result.false_negative_rate


def test_kish_effective_n_is_below_the_row_count_under_unequal_weights() -> None:
    result = single_label_agreement(_pairs_with_weights())

    assert result.effective_n is not None
    assert result.effective_n < result.n


def test_equal_weights_leave_weighted_and_unweighted_identical() -> None:
    pairs = [
        PairedDecision(record_id="a", ai_label="include", human_label="include"),
        PairedDecision(record_id="b", ai_label="include", human_label="exclude"),
        PairedDecision(record_id="c", ai_label="exclude", human_label="exclude"),
    ]

    result = single_label_agreement(pairs)

    assert result.is_weighted is False
    assert result.weighted_percent_agreement == pytest.approx(result.percent_agreement)
    assert result.weighted_cohen_kappa == pytest.approx(result.cohen_kappa)
    assert result.effective_n == pytest.approx(result.n)


def test_report_leads_with_the_weighted_column_when_weights_are_present(tmp_path: Path) -> None:
    result = DimensionAgreement(
        dimension="topical_inclusion",
        kind="single_label",
        single=single_label_agreement(_pairs_with_weights()),
        multi=None,
    )
    out = tmp_path / "agreement.md"
    write_agreement_report([result], out)
    text = out.read_text(encoding="utf-8")

    assert "Read the weighted column" in text
    assert "design-weighted (corpus estimate)" in text
    assert "Kish effective n" in text


# --- D-36: kappa in context -------------------------------------------------


def _skewed_marginals() -> list[PairedDecision]:
    """The Phase 11 shape in miniature: the AI says include far more often than
    the human does, and the human never uses `uncertain` at all."""
    pairs = [PairedDecision(record_id=f"tp-{i}", ai_label="include", human_label="include") for i in range(2)]
    pairs += [PairedDecision(record_id=f"fp-{i}", ai_label="include", human_label="exclude") for i in range(8)]
    pairs += [PairedDecision(record_id=f"tn-{i}", ai_label="exclude", human_label="exclude") for i in range(6)]
    pairs += [PairedDecision(record_id=f"un-{i}", ai_label="uncertain", human_label="exclude") for i in range(4)]
    return pairs


def test_kappa_max_bounds_the_observed_kappa() -> None:
    result = single_label_agreement(_skewed_marginals())

    assert result.kappa_max is not None
    assert result.cohen_kappa is not None
    assert result.cohen_kappa <= result.kappa_max
    # And the ceiling itself is far below 1 -- that is the whole point.
    assert result.kappa_max < 0.5


def test_kappa_max_is_one_when_the_marginals_match() -> None:
    pairs = [
        PairedDecision(record_id="a", ai_label="include", human_label="include"),
        PairedDecision(record_id="b", ai_label="exclude", human_label="include"),
        PairedDecision(record_id="c", ai_label="include", human_label="exclude"),
        PairedDecision(record_id="d", ai_label="exclude", human_label="exclude"),
    ]

    result = single_label_agreement(pairs)

    assert result.kappa_max == pytest.approx(1.0)


def test_gwet_ac1_exceeds_kappa_when_disagreement_sits_in_the_marginals() -> None:
    result = single_label_agreement(_skewed_marginals())

    assert result.gwet_ac1 is not None
    assert result.cohen_kappa is not None
    assert result.gwet_ac1 > result.cohen_kappa


def test_pabak_is_a_linear_function_of_observed_agreement() -> None:
    pairs = _skewed_marginals()
    result = single_label_agreement(pairs)

    q = len({p.ai_label for p in pairs} | {p.human_label for p in pairs})
    assert result.pabak == pytest.approx((q * result.percent_agreement - 1) / (q - 1))


def test_auc_separates_a_good_ranker_from_a_bad_threshold() -> None:
    """Every human-include is scored above every human-exclude, but the AI's
    own cut-point still puts most of them on the wrong side."""
    pairs = [
        PairedDecision(record_id="p1", ai_label="include", human_label="include", ai_confidence="high"),
        PairedDecision(record_id="p2", ai_label="include", human_label="include", ai_confidence="medium"),
        PairedDecision(record_id="n1", ai_label="exclude", human_label="exclude", ai_confidence="high"),
        PairedDecision(record_id="n2", ai_label="exclude", human_label="exclude", ai_confidence="medium"),
    ]

    result = single_label_agreement(pairs)

    assert result.auc == pytest.approx(1.0)


def test_auc_is_a_half_when_the_score_carries_no_signal() -> None:
    pairs = [
        PairedDecision(record_id="p1", ai_label="include", human_label="include", ai_confidence="high"),
        PairedDecision(record_id="n1", ai_label="include", human_label="exclude", ai_confidence="high"),
    ]

    result = single_label_agreement(pairs)

    assert result.auc == pytest.approx(0.5)


def test_auc_uses_confidence_to_break_the_labels_ties() -> None:
    """Two AI-includes, and the human agrees with the confident one. On the
    label alone this is a coin flip; on (label, confidence) it is perfect."""
    pairs = [
        PairedDecision(record_id="p1", ai_label="include", human_label="include", ai_confidence="high"),
        PairedDecision(record_id="n1", ai_label="include", human_label="exclude", ai_confidence="low"),
    ]

    result = single_label_agreement(pairs)

    assert result.auc == pytest.approx(1.0)


def test_report_prints_the_kappa_ceiling_and_omits_landis_koch(tmp_path: Path) -> None:
    result = DimensionAgreement(
        dimension="topical_inclusion",
        kind="single_label",
        single=single_label_agreement(_skewed_marginals()),
        multi=None,
    )
    out = tmp_path / "agreement.md"
    write_agreement_report([result], out)
    text = out.read_text(encoding="utf-8")

    assert "kappa_max" in text
    assert "PABAK" in text
    assert "Gwet's AC1" in text
    assert "AUC" in text
    # The verbal bands must never be printed next to a kappa with a low ceiling.
    for band in ("slight", "fair agreement", "moderate agreement", "substantial agreement"):
        assert band not in text.lower()


def test_report_warns_about_a_stratum_that_was_drawn_but_not_adjudicated(tmp_path: Path) -> None:
    single = single_label_agreement(_pairs_with_weights())
    single.stratum_coverage = {"ai_include": (8, 8), "ai_exclude_high": (0, 100)}
    result = DimensionAgreement(dimension="topical_inclusion", kind="single_label", single=single, multi=None)

    out = tmp_path / "agreement.md"
    write_agreement_report([result], out)
    text = out.read_text(encoding="utf-8")

    assert "Incomplete strata" in text
    assert "`ai_exclude_high` 0/100" in text
    assert "ai_include" not in text.split("Incomplete strata")[1].split("\n")[0]


def test_paired_decisions_carries_the_ai_confidence(tmp_path: Path) -> None:
    engine = _db(tmp_path)
    _insert_record(engine, "r1")
    _insert_classification(engine, "classifications_ai", "r1", "topical_inclusion", "include", "claude-opus-5")
    _insert_classification(engine, "classifications_human", "r1", "topical_inclusion", "include", "izelioli")

    pairs = paired_decisions(engine, "topical_inclusion")

    assert pairs[0].ai_confidence == "high"
