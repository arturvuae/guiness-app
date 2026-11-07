"""How many observations does a backtest actually contain?

The trade counter measures activity, not information. A strategy showing 2,000
trades can carry a few hundred independent observations, because trades that
overlap in time, or that sit in correlated instruments, are largely the same
bet recorded several times.

This matters because the t-statistic computed on trade returns divides by the
square root of the trade count. Inflate the count and every strategy looks
significant. The self-test below measures the effect on a strategy built to
have exactly zero alpha: the trade-based test rejects the null far more often
than its nominal 5%.

The practical conclusion is blunt: **the t-statistic on trades is not a
statistic to correct, it is a statistic to abandon.** Measure on time periods
instead, where the observation count cannot be inflated by trading more.

Usage::

    from effective_n import kish_effective_n, effective_n_from_autocorr

    n_eff = kish_effective_n(corr)             # from a correlation matrix
    n_eff = effective_n_from_autocorr(returns) # from a single series

    python effective_n.py --selftest

Requires numpy.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Effective sample size
# --------------------------------------------------------------------------


def kish_effective_n(corr: np.ndarray) -> float:
    """Effective number of independent observations, from a correlation matrix.

    ``N_eff = N^2 / sum_ij(rho_ij)``

    With all correlations zero this returns N, as it should. With everything
    perfectly correlated it returns 1: a thousand copies of the same bet are
    one bet.
    """
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("corr must be a square matrix")
    n = c.shape[0]
    total = float(c.sum())
    if total <= 0:
        return float(n)
    return n * n / total


def effective_n_from_autocorr(returns: Sequence[float],
                              max_lag: Optional[int] = None) -> float:
    """Effective sample size of one series, from its own autocorrelation.

    ``N_eff = N / (1 + 2 * sum_k (1 - k/N) * rho_k)``

    Positive autocorrelation, which is what overlapping positions produce,
    drives this below N. The Bartlett taper keeps the estimate stable at long
    lags.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 8:
        return float(n)
    if max_lag is None:
        max_lag = max(1, min(n // 4, int(4 * (n / 100.0) ** (2.0 / 9.0)) + 10))

    x = r - r.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return float(n)

    total = 0.0
    for k in range(1, max_lag + 1):
        rho = float((x[:-k] * x[k:]).sum()) / denom
        total += (1.0 - k / n) * rho
    factor = 1.0 + 2.0 * total
    if factor <= 0:
        return float(n)
    return max(1.0, n / factor)


def overlap_correlation(starts: Sequence[float], ends: Sequence[float]) -> np.ndarray:
    """Correlation proxy for trades, built from how much they overlap in time.

    Two positions held over the same days are exposed to the same shocks. The
    fraction of shared holding time, normalized by the geometric mean of the
    two holding periods, is a serviceable stand-in for their correlation when
    the individual return streams are not available.
    """
    s = np.asarray(starts, dtype=float)
    e = np.asarray(ends, dtype=float)
    if len(s) != len(e):
        raise ValueError("starts and ends must have the same length")
    length = np.maximum(e - s, 1e-9)
    lo = np.maximum(s[:, None], s[None, :])
    hi = np.minimum(e[:, None], e[None, :])
    shared = np.clip(hi - lo, 0.0, None)
    norm = np.sqrt(length[:, None] * length[None, :])
    corr = shared / norm
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, 0.0, 1.0)


# --------------------------------------------------------------------------
# The two t-statistics
# --------------------------------------------------------------------------


def t_on_trades(trade_returns: Sequence[float]) -> float:
    """The statistic to abandon. Kept here so the self-test can indict it."""
    r = np.asarray(trade_returns, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r))))


def t_on_trades_corrected(trade_returns: Sequence[float],
                          corr: np.ndarray) -> float:
    """Same statistic, rescaled to the effective count. Better, still not good.

    It fixes the arithmetic while leaving the deeper problem: trades are chosen
    by the strategy, so their number and their timing are outcomes of the thing
    being tested.
    """
    r = np.asarray(trade_returns, dtype=float)
    n_eff = kish_effective_n(corr)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / (r.std(ddof=1) / math.sqrt(n_eff)))


def t_on_periods(period_returns: Sequence[float]) -> float:
    """The statistic to use: computed on time periods, not on trades.

    Trading more cannot inflate this. The denominator counts days, and days
    arrive whatever the strategy does.
    """
    r = np.asarray(period_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r))))


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _zero_alpha_world(rng: np.random.Generator, n_assets: int = 16,
                      days: int = 2520, hold: int = 20,
                      asset_corr: float = 0.5):
    """A strategy with EXACTLY zero alpha, trading a lot.

    Correlated assets with no drift, and a book-wide direction that persists
    for ``hold`` days at a time: long everything, then short everything. That
    persistence is the point. It is what makes concurrent trades the same bet
    recorded sixteen times, and it is how real strategies behave, since a
    signal that flips per instrument independently is rare.
    """
    common = rng.normal(0, 1, size=(days, 1))
    idio = rng.normal(0, 1, size=(days, n_assets))
    shocks = math.sqrt(asset_corr) * common + math.sqrt(1 - asset_corr) * idio
    asset_returns = 0.01 * shocks                     # zero drift by construction

    n_blocks = days // hold
    directions = rng.choice([-1.0, 1.0], size=(n_blocks, 1))   # book-wide

    trade_returns: List[float] = []
    starts: List[float] = []
    ends: List[float] = []
    assets: List[int] = []
    position = np.zeros((days, n_assets))

    for b in range(n_blocks):
        lo, hi = b * hold, (b + 1) * hold
        position[lo:hi, :] = directions[b, 0]
        for a in range(n_assets):
            trade_returns.append(float((asset_returns[lo:hi, a] * directions[b, 0]).sum()))
            starts.append(lo)
            ends.append(hi)
            assets.append(a)

    daily = (position * asset_returns).mean(axis=1)   # equal-weight book
    return (trade_returns, np.array(starts, dtype=float),
            np.array(ends, dtype=float), np.array(assets), daily)


def _trade_correlation(starts, ends, assets, asset_corr: float) -> np.ndarray:
    """Overlap in time, discounted by how correlated the instruments are."""
    corr = overlap_correlation(starts, ends)
    same = assets[:, None] == assets[None, :]
    cross = np.where(same, 1.0, asset_corr)
    out = corr * cross
    np.fill_diagonal(out, 1.0)
    return out


def _selftest(replications: int = 200) -> int:
    rng = np.random.default_rng(5)
    n_trades = 0
    n_eff_kish: List[float] = []
    n_eff_auto: List[float] = []
    reject_trades = 0
    reject_corrected = 0
    reject_periods = 0

    for _ in range(replications):
        trades, starts, ends, assets, daily = _zero_alpha_world(rng)
        n_trades = len(trades)
        corr = _trade_correlation(starts, ends, assets, 0.5)

        n_eff_kish.append(kish_effective_n(corr))
        n_eff_auto.append(effective_n_from_autocorr(daily))

        if abs(t_on_trades(trades)) > 1.96:
            reject_trades += 1
        if abs(t_on_trades_corrected(trades, corr)) > 1.96:
            reject_corrected += 1
        if abs(t_on_periods(daily)) > 1.96:
            reject_periods += 1

    print("A strategy with EXACTLY zero alpha, by construction.")
    print(f"16 correlated assets, 10 years, direction held 20 days. "
          f"{replications} replications.")
    print()
    print(f"  trades per replication            {n_trades:,}")
    print(f"  effective observations (Kish)     {np.median(n_eff_kish):,.0f}")
    print(f"  effective observations (autocorr) {np.median(n_eff_auto):,.0f}")
    print(f"  inflation factor                  "
          f"{math.sqrt(n_trades / np.median(n_eff_kish)):.1f}x on the t-stat")
    print()
    print("  False positive rate at a nominal 5%:")
    print(f"    t on trades (naive)             {reject_trades / replications:.0%}")
    print(f"    t on trades (Kish-corrected)    {reject_corrected / replications:.0%}")
    print(f"    t on daily returns              {reject_periods / replications:.0%}")
    print()
    print("  The naive trade t-stat rejects a true null far more often than it")
    print("  should. Counting days instead of trades removes the inflation,")
    print("  because trading more cannot create more days.")

    assert np.median(n_eff_kish) < n_trades / 4, "expected heavy overlap"
    assert reject_trades > 3 * reject_periods, (
        "the naive trade t-stat should over-reject a true null badly"
    )
    assert reject_periods / replications < 0.12, (
        "the daily t-stat should sit near its nominal 5%"
    )
    print("\nself-test OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)
    if args.selftest:
        return _selftest()
    p.error("nothing to do; use --selftest or import the module")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
