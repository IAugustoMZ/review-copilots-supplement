"""Precision-based sample size for the Phase 11/12 AI-human agreement estimate.

`sampling.DEFAULT_TARGET_SIZE` was a chosen round number (90). What the review
actually has to defend is the *precision* of the reported Cohen's kappa, so this
module derives n from a stated confidence-interval half-width instead, following
the confidence-interval approach to sample size for interobserver agreement
studies (Donner & Eliasziw 1992; Rotondi & Donner 2012, implemented in the R
package `kappaSize` as `CIBinary`). Sizing on the width of the interval, rather
than on power to reject some null kappa, is the right frame here: nobody is
testing "is agreement better than chance", the manuscript reports an estimate
and its uncertainty.

For a binary rating by two raters the large-sample variance of the intraclass
kappa is (Bloch & Kraemer 1989, Biometrics 45:269-287):

    Var(k) = (1 - k) / n * [ (1 - k)(1 - 2k) + k(2 - k) / (2 * pi * (1 - pi)) ]

with `pi` the prevalence of the positive category (here: the proportion of
records the AI screen called `include`). Solving the Wald half-width
d = z_{1-alpha/2} * sqrt(Var(k)) for n gives required_n() below.

Two consequences worth stating explicitly, because both bite in practice:

- Precision degrades sharply as prevalence moves away from 0.5. The
  `2*pi*(1-pi)` denominator means a screen that includes 15% of records needs
  roughly twice the sample of one that includes 50% for the same interval.
- The cost is quadratic in precision. Halving the half-width quadruples n. This
  is why the honest move is to report the n/precision trade-off and let the
  reviewer choose a point on it, rather than to quietly pick one -- see
  precision_table().
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Two-sided normal quantiles for the alpha levels a review would plausibly use.
# Avoids a scipy dependency for what is three constants.
_Z_BY_ALPHA: dict[float, float] = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}


def _z(alpha: float) -> float:
    try:
        return _Z_BY_ALPHA[round(alpha, 3)]
    except KeyError as exc:
        raise ValueError(f"alpha must be one of {sorted(_Z_BY_ALPHA)}, got {alpha}") from exc


def kappa_variance_factor(kappa: float, prevalence: float) -> float:
    """The n-independent part of Var(kappa): Var(kappa) == factor / n."""
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"prevalence must be in (0, 1), got {prevalence}")
    if not -1.0 < kappa < 1.0:
        raise ValueError(f"kappa must be in (-1, 1), got {kappa}")
    return (1.0 - kappa) * ((1.0 - kappa) * (1.0 - 2.0 * kappa) + kappa * (2.0 - kappa) / (2.0 * prevalence * (1.0 - prevalence)))


def required_n(kappa: float, half_width: float, prevalence: float, alpha: float = 0.05) -> int:
    """Records needed for a two-sided (1-alpha) CI of the given half-width.

    `kappa` is the anticipated value -- the calculation is only as good as that
    guess, which is why anticipate_kappa_from_literature() documents where ours
    comes from. Returns a ceiling, so the realised interval is never wider than
    asked for at the anticipated kappa.
    """
    if half_width <= 0:
        raise ValueError(f"half_width must be positive, got {half_width}")
    return math.ceil(_z(alpha) ** 2 * kappa_variance_factor(kappa, prevalence) / half_width**2)


def achieved_half_width(n: int, kappa: float, prevalence: float, alpha: float = 0.05) -> float:
    """Inverse of required_n: the precision a given n buys. Used to report what
    a reviewer-chosen (smaller) n actually delivers, rather than leaving the
    interval unstated."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return _z(alpha) * math.sqrt(kappa_variance_factor(kappa, prevalence) / n)


@dataclass(frozen=True)
class PrecisionOption:
    n: int
    half_width: float


def precision_table(
    kappa: float,
    prevalence: float,
    candidate_ns: tuple[int, ...] = (50, 90, 120, 150, 200, 250, 300, 400),
    alpha: float = 0.05,
) -> list[PrecisionOption]:
    """n -> achieved half-width, for presenting the trade-off as a choice.

    The default candidates bracket the previous review's 90 so the cost of that
    earlier choice is visible alongside the alternatives.
    """
    return [PrecisionOption(n=n, half_width=achieved_half_width(n, kappa, prevalence, alpha)) for n in candidate_ns]


# Anticipated kappa for the sizing calculation. Screening agreement between an
# LLM screen and a human reviewer clusters around 0.7 for systematic reviews
# with tight inclusion criteria and drops to roughly 0.55-0.60 for scoping and
# narrative reviews, whose criteria are broader and more judgement-laden
# (Cochrane-style concordance analyses of AI screening, 2025-2026). This review
# is a structured narrative review over a deliberately broad D0 discovery pool,
# which is the harder case, so sizing on the optimistic 0.7 would under-size the
# sample. 0.60 is the defensible anticipation here.
#
# Note this cuts the safe way: required_n is larger for smaller anticipated
# kappa across this range, so guessing low buys precision rather than losing it.
ANTICIPATED_KAPPA = 0.60


def anticipate_kappa_from_literature() -> float:
    """The anticipated kappa the sizing uses. A function, not a bare constant, so
    the manuscript's methods text and the CLI quote the same documented value."""
    return ANTICIPATED_KAPPA
