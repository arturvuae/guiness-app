"""Phase 5 of the quant-research skill: turning an edge estimate into a bet size.

Phase 3 produced a pooled estimate of an edge. Phase 4 said how much of it was
manufactured by the search. This module answers the remaining question: how
much to bet, given that the edge is not known but *estimated*, with an error
bar that is usually large.

The short version, which the self-test measures rather than asserts: sizing off
a raw in-sample Sharpe using the Kelly formula is not aggressive, it is
ruinous. Kelly assumes the parameters are known. They are not, and the gap
between "known" and "estimated with SE 0.5" is the whole problem.

Usage as a library::

    from sizing import shrink_sharpe, kelly_leverage, sizing_report

    print(sizing_report(sharpe=0.64, se=0.32, vol=0.15, trials=250_000))

From the shell::

    python sizing.py --sharpe 0.64 --se 0.32 --vol 0.15 --trials 250000
    python sizing.py --selftest

Requires numpy.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Shrinking the estimate before it is used
# --------------------------------------------------------------------------


def shrink_sharpe(sharpe: float, se: float, prior_sd: float = 0.5) -> float:
    """Pull a Sharpe estimate toward zero in proportion to how noisy it is.

    Posterior mean under a normal prior centred on zero: strategies do not have
    large edges a priori, and an estimate with a wide error bar should barely
    move that belief.

    ``prior_sd`` encodes how good strategies plausibly are before measurement.
    0.5 is a defensible default: it says a true Sharpe above 1.0 is a two-sigma
    event. Raise it only with a reason that is not "my backtest said so".

    >>> round(shrink_sharpe(1.0, 0.5), 3)     # noisy estimate, halved
    0.5
    >>> round(shrink_sharpe(1.0, 0.1), 3)     # precise estimate, barely moved
    0.962
    """
    if se < 0 or prior_sd <= 0:
        raise ValueError("se must be >= 0 and prior_sd > 0")
    weight = prior_sd ** 2 / (prior_sd ** 2 + se ** 2)
    return sharpe * weight


def trial_penalty(sharpe: float, se: float, trials: int) -> float:
    """Subtract the Sharpe a pure-noise search would have produced anyway.

    The expected maximum of ``trials`` independent noise draws, using the
    standard extreme-value approximation.

    **This penalty applies to a SELECTED MAXIMUM, and only to one.** It asks
    "how high would the best of N noise draws have scored?", so it is the right
    correction for an argmax and the wrong correction for a pooled region,
    which is an average and not a maximum. Passing the full grid size here
    after pooling double-charges the study and will size it to zero.

    What to pass:

    - argmax of a grid of N cells: N, or the effective count if the cells are
      correlated (they are, so N is conservative)
    - a pooled region, selected once: the number of independent SELECTIONS
      made, which is usually 1 or a small number, not the grid size
    - a single pre-registered backtest: 1
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if trials == 1:
        return sharpe
    gamma = 0.5772156649
    from statistics import NormalDist
    nd = NormalDist()
    z1 = nd.inv_cdf(1.0 - 1.0 / trials)
    z2 = nd.inv_cdf(1.0 - 1.0 / (trials * math.e))
    expected_max = se * ((1.0 - gamma) * z1 + gamma * z2)
    return sharpe - expected_max


# --------------------------------------------------------------------------
# Kelly
# --------------------------------------------------------------------------


def kelly_leverage(sharpe: float, vol: float) -> float:
    """Growth-optimal leverage for a lognormal return stream: ``S / sigma``.

    This is the leverage that maximizes long-run log wealth IF the Sharpe and
    the volatility are known exactly. Both caveats matter, and the second one
    is usually forgotten: volatility is also estimated, and it moves.

    >>> round(kelly_leverage(0.5, 0.15), 2)
    3.33
    """
    if vol <= 0:
        raise ValueError("vol must be > 0")
    return sharpe / vol


def log_growth(leverage: float, true_sharpe: float, vol: float) -> float:
    """Expected log growth at a given leverage, under the TRUE parameters.

    ``g = L * S * sigma - (L * sigma)^2 / 2``

    The quadratic term is why over-betting is not merely suboptimal. Past
    twice the Kelly leverage, expected growth turns negative: the strategy has
    a real edge and still loses money, forever.
    """
    return leverage * true_sharpe * vol - (leverage * vol) ** 2 / 2.0


def growth_optimal_ceiling(true_sharpe: float, vol: float) -> float:
    """Leverage at which expected log growth crosses zero: twice Kelly."""
    return 2.0 * kelly_leverage(true_sharpe, vol)


# --------------------------------------------------------------------------
# Volatility targeting
# --------------------------------------------------------------------------


def vol_target_scale(
    realized_vol: Sequence[float],
    target_vol: float,
    max_leverage: float = 3.0,
) -> np.ndarray:
    """Scale factor per period to hold volatility near a target.

    The cap is not optional. Realized volatility in the denominator means the
    scale explodes exactly when markets are quietest, which is reliably just
    before they are not.
    """
    rv = np.asarray(realized_vol, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(rv > 0, target_vol / rv, 0.0)
    return np.clip(scale, 0.0, max_leverage)


def drawdown_distribution(leverage: float, sharpe: float, vol: float,
                          years: float, n_paths: int = 2000,
                          rng: Optional[np.random.Generator] = None,
                          periods: int = 252) -> np.ndarray:
    """Maximum drawdowns across many simulated lives at a fixed leverage.

    A realized maximum drawdown is one draw from this distribution. It is not a
    property of the strategy and not a ceiling on what comes next: unlike
    volatility, drawdown extremes carry no memory from one period to the next,
    so the worst decline seen so far predicts the next one poorly.

    Which is why sizing reads a PERCENTILE of this distribution rather than a
    single historical number. The realized figure typically sits well below the
    tail that matters.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = int(years * periods)
    mu_d = sharpe * vol / periods
    sd_d = vol / math.sqrt(periods)
    r = rng.normal(mu_d, sd_d, size=(n_paths, n)) * leverage
    equity = np.cumprod(1.0 + np.clip(r, -0.99, None), axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    return (1.0 - equity / peak).max(axis=1)


def drawdown_quantile(leverage: float, sharpe: float, vol: float, years: float,
                      percentile: float = 95.0, n_paths: int = 2000,
                      rng: Optional[np.random.Generator] = None) -> float:
    """The drawdown that only ``100 - percentile`` percent of lives exceed."""
    dd = drawdown_distribution(leverage, sharpe, vol, years, n_paths, rng)
    return float(np.percentile(dd, percentile))


def leverage_for_drawdown(sharpe: float, vol: float, years: float,
                          tolerance: float, percentile: float = 95.0,
                          n_paths: int = 2000, max_leverage: float = 10.0,
                          rng: Optional[np.random.Generator] = None) -> float:
    """Largest leverage whose drawdown at ``percentile`` stays within tolerance.

    Solved by bisection on the simulated distribution. Sizing on the p95 rather
    than the median is deliberate: the median year is not the one that ends the
    strategy.
    """
    if sharpe <= 0 or vol <= 0 or tolerance <= 0:
        return 0.0
    if rng is None:
        rng = np.random.default_rng(3)
    lo, hi = 0.0, max_leverage
    if drawdown_quantile(hi, sharpe, vol, years, percentile, n_paths, rng) <= tolerance:
        return hi
    for _ in range(14):
        mid = (lo + hi) / 2.0
        if drawdown_quantile(mid, sharpe, vol, years, percentile, n_paths, rng) <= tolerance:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class SizingAdvice:
    raw_sharpe: float
    se: float
    trials: int
    deflated: float
    shrunk: float
    vol: float
    kelly_raw: float
    kelly_used: float
    dd_capped: float
    recommended: float
    fraction: float
    binding: str


def recommend(
    sharpe: float,
    se: float,
    vol: float,
    trials: int = 1,
    kelly_fraction: float = 0.5,
    max_leverage: float = 3.0,
    max_drawdown: float = 0.35,
    years: float = 10.0,
) -> SizingAdvice:
    """Full chain: deflate for the search, shrink for the noise, then bet a
    fraction of Kelly on what survives.

    The two corrections do different jobs and stack in this order:

    - ``trials`` removes the bias from having SELECTED this result. See
      :func:`trial_penalty` for what to pass; for a pooled region it is small.
    - shrinking removes the over-confidence from having MEASURED it noisily.

    ``kelly_fraction`` then covers what neither addresses: the model is wrong
    in ways nobody estimated, and volatility is itself a moving estimate. Half
    is the defensible default once the estimate has been shrunk. A quarter is
    for a fragile estimate; anything above half needs an argument, and full
    Kelly on an estimated edge is not a bet, it is a coin flip on ruin.
    """
    deflated = trial_penalty(sharpe, se, trials)
    shrunk = shrink_sharpe(max(deflated, 0.0), se)
    kelly_raw = kelly_leverage(sharpe, vol)
    kelly_used = kelly_leverage(shrunk, vol)

    candidates = {
        "fractional Kelly": kelly_used * kelly_fraction,
        "drawdown tolerance": leverage_for_drawdown(shrunk, vol, years, max_drawdown),
        "hard leverage cap": max_leverage,
    }
    binding = min(candidates, key=candidates.get)
    recommended = max(min(candidates.values()), 0.0)

    return SizingAdvice(
        raw_sharpe=sharpe, se=se, trials=trials, deflated=deflated,
        shrunk=shrunk, vol=vol, kelly_raw=kelly_raw, kelly_used=kelly_used,
        dd_capped=candidates["drawdown tolerance"],
        recommended=recommended, fraction=kelly_fraction, binding=binding,
    )


def sizing_report(sharpe: float, se: float, vol: float, trials: int = 1,
                  kelly_fraction: float = 0.5, max_leverage: float = 3.0,
                  years: float = 10.0, max_drawdown: float = 0.35) -> str:
    a = recommend(sharpe, se, vol, trials, kelly_fraction, max_leverage,
                  max_drawdown, years)
    dd_med = (drawdown_quantile(a.recommended, a.shrunk, a.vol, years, 50.0)
              if a.recommended > 0 else 0.0)
    dd_p95 = (drawdown_quantile(a.recommended, a.shrunk, a.vol, years, 95.0)
              if a.recommended > 0 else 0.0)

    lines = [
        "Sizing",
        "-" * 58,
        f"  measured Sharpe           {a.raw_sharpe:+.3f}  (SE {a.se:.3f})",
        ("  after {:,} trial{}".format(a.trials, "" if a.trials == 1 else "s")
         ).ljust(28) + f"{a.deflated:+.3f}",
        f"  after shrinking           {a.shrunk:+.3f}",
        "",
        f"  annualized vol            {a.vol:.1%}",
        f"  Kelly on the raw estimate {a.kelly_raw:.2f}x   <- never bet this",
        f"  Kelly on what survives    {a.kelly_used:.2f}x",
        "",
        "  caps, smallest wins",
        f"    {a.fraction:g} Kelly".ljust(28) + f"{a.kelly_used * a.fraction:.2f}x",
        f"    {max_drawdown:.0%} drawdown at p95".ljust(28) + f"{a.dd_capped:.2f}x",
        f"    hard cap".ljust(28) + f"{max_leverage:.2f}x",
        "",
        f"  RECOMMENDED SIZE          {a.recommended:.2f}x"
        f"   (bound by {a.binding})",
        "",
        f"  drawdown over {years:g} years at that size",
        f"    median life".ljust(28) + f"{dd_med:.0%}",
        f"    bad life (p95)".ljust(28) + f"{dd_p95:.0%}   <- size against this one",
    ]
    if a.deflated <= 0:
        lines += ["", "  ! the edge does not survive the trial count: size is zero"]
    elif a.recommended <= 0.1:
        lines += ["", "  ! what survives is too small to trade at meaningful size"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _simulate(leverage: float, true_sharpe: float, vol: float,
              years: float, rng: np.random.Generator,
              periods: int = 252) -> Dict[str, float]:
    """One live path under a fixed leverage, returning growth and drawdown."""
    n = int(years * periods)
    mu_d = true_sharpe * vol / periods
    sd_d = vol / math.sqrt(periods)
    r = rng.normal(mu_d, sd_d, size=n) * leverage
    # log wealth, floored: a path that loses everything stays lost
    equity = np.cumprod(1.0 + np.clip(r, -0.99, None))
    peak = np.maximum.accumulate(equity)
    dd = float((1.0 - equity / peak).max())
    return {
        "cagr": float(equity[-1] ** (1.0 / years) - 1.0),
        "max_dd": dd,
        "ruined": bool(equity[-1] < 0.5),
    }


FULL_KELLY = "full Kelly on the estimate"
HALF_KELLY = "half Kelly on the estimate"
SHRUNK_KELLY = "half Kelly on the shrunk estimate"
FIXED = "fixed 1x, no edge estimate used"


def _selftest(replications: int = 400) -> int:
    """Four sizing policies, one edge, measured over many lives.

    The true Sharpe is 0.50. Each replication measures it over four years,
    which is what a researcher actually has, sizes on that measurement, then
    lives with the consequence for ten years.

    The comparison is deliberately NOT on median outcome. Every policy looks
    fine in the median; the differences live in the left tail, which is where
    accounts die. Read the 10th percentile column and the ruin column.
    """
    rng = np.random.default_rng(11)
    true_sharpe = 0.50
    vol = 0.15
    is_years = 4.0
    live_years = 10.0

    se = math.sqrt((1.0 + true_sharpe ** 2 / 2.0) / is_years)
    policies = [FULL_KELLY, HALF_KELLY, SHRUNK_KELLY, FIXED]
    out: Dict[str, List[Dict[str, float]]] = {p: [] for p in policies}
    leverages: Dict[str, List[float]] = {p: [] for p in policies}

    for _ in range(replications):
        # what the researcher measures in-sample, noise included
        measured = rng.normal(true_sharpe, se)
        raw_kelly = max(kelly_leverage(measured, vol), 0.0)

        levs = {
            FULL_KELLY: raw_kelly,
            HALF_KELLY: raw_kelly * 0.5,
            SHRUNK_KELLY: recommend(measured, se, vol, trials=1,
                                    kelly_fraction=0.5, max_leverage=1e9,
                                    max_drawdown=1e9).recommended,
            FIXED: 1.0,
        }
        for name, lev in levs.items():
            leverages[name].append(lev)
            out[name].append(_simulate(lev, true_sharpe, vol, live_years, rng))

    print(f"True Sharpe {true_sharpe:.2f}, vol {vol:.0%}, measured over "
          f"{is_years:g} years (SE {se:.2f}), then traded for {live_years:g}.")
    print(f"{replications} replications. Growth-optimal leverage is "
          f"{kelly_leverage(true_sharpe, vol):.2f}x, and expected growth turns "
          f"negative past {growth_optimal_ceiling(true_sharpe, vol):.2f}x.")
    print()
    header = (f"  {'policy':<36}{'lev':>7}{'CAGR p50':>10}{'CAGR p10':>10}"
              f"{'maxDD p90':>11}{'lost half':>11}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in policies:
        lev = float(np.median(leverages[name]))
        cagrs = np.array([r["cagr"] for r in out[name]])
        dds = np.array([r["max_dd"] for r in out[name]])
        ruin = float(np.mean([r["ruined"] for r in out[name]]))
        print(f"  {name:<36}{lev:>6.2f}x{np.median(cagrs):>9.1%}"
              f"{np.percentile(cagrs, 10):>10.1%}"
              f"{np.percentile(dds, 90):>11.0%}{ruin:>10.0%}")

    print()
    print("  Reading it: full Kelly on a four-year estimate loses half the")
    print("  account in a fifth of lives, for no gain in the median. The fixed")
    print("  1x row is not a policy that generalizes, it just happens to sit")
    print("  below Kelly at this Sharpe; at a lower true edge it over-bets.")

    def p10(name: str) -> float:
        return float(np.percentile([r["cagr"] for r in out[name]], 10))

    def ruin_rate(name: str) -> float:
        return float(np.mean([r["ruined"] for r in out[name]]))

    assert p10(SHRUNK_KELLY) > p10(FULL_KELLY), (
        "shrinking should protect the left tail against naive full Kelly"
    )
    assert ruin_rate(SHRUNK_KELLY) < ruin_rate(FULL_KELLY)
    assert abs(shrink_sharpe(1.0, 0.5) - 0.5) < 1e-9
    assert trial_penalty(2.0, 0.5, 1) == 2.0
    assert trial_penalty(2.0, 0.5, 10_000) < 2.0
    print("\nself-test OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sharpe", type=float, help="out-of-sample Sharpe of the pooled region")
    p.add_argument("--se", type=float, help="its standard error (see region_pool.py)")
    p.add_argument("--vol", type=float, default=0.15, help="annualized vol of the strategy at 1x")
    p.add_argument("--trials", type=int, default=1, help="total backtests looked at")
    p.add_argument("--kelly-fraction", type=float, default=0.5,
                   help="fraction of Kelly to bet (0.5 default, 0.25 if fragile)")
    p.add_argument("--max-leverage", type=float, default=3.0)
    p.add_argument("--years", type=float, default=10.0, help="horizon for the drawdown estimate")
    p.add_argument("--max-drawdown", type=float, default=0.35,
                   help="largest drawdown the capital can actually survive")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.sharpe is None or args.se is None:
        p.error("--sharpe and --se are required (or use --selftest)")

    print(sizing_report(args.sharpe, args.se, args.vol, args.trials,
                        args.kelly_fraction, args.max_leverage, args.years,
                        args.max_drawdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
