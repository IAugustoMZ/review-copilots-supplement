from __future__ import annotations

import math

import pytest

from review_pipeline.src.validation.sample_size import (
    achieved_half_width,
    anticipate_kappa_from_literature,
    kappa_variance_factor,
    precision_table,
    required_n,
)


def test_required_n_matches_hand_computed_bloch_kraemer_value() -> None:
    """kappa=0.7, prevalence=0.2, d=0.10, alpha=0.05 worked by hand from
    Var(k) = (1-k)/n * [(1-k)(1-2k) + k(2-k)/(2*pi*(1-pi))]:
    factor = 0.3 * (-0.12 + 0.91/0.32) = 0.817125; n = 1.96^2 * 0.817125 / 0.01.
    """
    assert required_n(kappa=0.7, half_width=0.10, prevalence=0.2) == 314


def test_required_n_and_achieved_half_width_are_inverses() -> None:
    n = required_n(kappa=0.6, half_width=0.15, prevalence=0.25)
    # required_n ceilings, so the realised interval is at most the one asked for.
    assert achieved_half_width(n, kappa=0.6, prevalence=0.25) <= 0.15 + 1e-12


def test_precision_cost_is_quadratic() -> None:
    """Halving the half-width should roughly quadruple n -- the property that
    makes the trade-off worth showing the reviewer explicitly."""
    coarse = required_n(kappa=0.6, half_width=0.20, prevalence=0.2)
    fine = required_n(kappa=0.6, half_width=0.10, prevalence=0.2)
    assert fine == pytest.approx(4 * coarse, rel=0.02)


def test_imbalanced_prevalence_costs_sample_size() -> None:
    """The 2*pi*(1-pi) denominator: a 15%-include screen needs a bigger sample
    than a balanced one for the same precision."""
    balanced = required_n(kappa=0.6, half_width=0.12, prevalence=0.50)
    skewed = required_n(kappa=0.6, half_width=0.12, prevalence=0.15)
    assert skewed > 2 * balanced


def test_variance_factor_is_symmetric_in_prevalence() -> None:
    assert kappa_variance_factor(0.6, 0.2) == pytest.approx(kappa_variance_factor(0.6, 0.8))


def test_tighter_alpha_needs_more_records() -> None:
    assert required_n(kappa=0.6, half_width=0.12, prevalence=0.2, alpha=0.01) > required_n(
        kappa=0.6, half_width=0.12, prevalence=0.2, alpha=0.05
    )


def test_precision_table_is_monotonically_tighter() -> None:
    rows = precision_table(kappa=0.6, prevalence=0.2)
    assert [r.n for r in rows] == sorted(r.n for r in rows)
    half_widths = [r.half_width for r in rows]
    assert half_widths == sorted(half_widths, reverse=True)
    assert all(math.isfinite(h) for h in half_widths)


def test_previous_target_size_of_90_is_documented_as_imprecise() -> None:
    """Guards the finding that motivated this module: the review's earlier
    fixed n=90 supports only a very wide interval at a realistic prevalence,
    so the number cannot just be carried forward unexamined."""
    assert achieved_half_width(90, kappa=anticipate_kappa_from_literature(), prevalence=0.2) > 0.20


@pytest.mark.parametrize("prevalence", [0.0, 1.0, -0.1, 1.5])
def test_prevalence_outside_the_open_unit_interval_is_rejected(prevalence: float) -> None:
    with pytest.raises(ValueError):
        required_n(kappa=0.6, half_width=0.1, prevalence=prevalence)


def test_unsupported_alpha_is_rejected_rather_than_silently_defaulted() -> None:
    with pytest.raises(ValueError):
        required_n(kappa=0.6, half_width=0.1, prevalence=0.2, alpha=0.025)
