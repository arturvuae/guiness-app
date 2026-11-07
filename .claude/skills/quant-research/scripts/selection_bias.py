"""Phase 4: how much of this result did the search manufacture?

Two tools that answer different halves of that question.

**Deflated Sharpe** asks: how high would the best of N noise draws have scored
anyway? Subtract that, and what remains is what the strategy contributed.

**Probability of backtest overfitting (PBO)** asks something sharper: does the
selection PROCEDURE generalize? It repeatedly splits the sample, applies the
procedure to one half, and checks where its choice lands on the other. A PBO
near 0.5 means the procedure has no predictive content, whatever the numbers
it produced.

The distinction matters. Deflating judges a result. PBO judges a method, which
is what you will reuse next quarter.

Usage::

    python selection_bias.py --sharpe 1.9 --se 0.53 --trials 10000
    python selection_bias.py --selftest

Requires numpy.
"""

from __future__ import annotations

import argparse
import itertools
import math
from statistics import NormalDist
from typing import List, Optional, Sequence

import numpy as np

_ND = NormalDist()
_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------
# Deflated Sharpe
# --------------------------------------------------------------------------


def expected_max_sharpe(se: float, trials: int) -> float:
    """Expected best Sharpe from ``trials`` pure-noise draws.

    Standard extreme-value approximation for the maximum of independent normal
    draws. This is the bar a strategy has to clear before it has demonstrated
    anything at all.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if trials == 1:
        return 0.0
    z1 = _ND.inv_cdf(1.0 - 1.0 / trials)
    z2 = _ND.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return se * ((1.0 - _GAMMA) * z1 + _GAMMA * z2)


def deflated_sharpe(sharpe: float, se: float, trials: int) -> float:
    """What survives after removing the height a search alone would produce.

    Pass the number of independent SELECTIONS. For an argmax over a grid that
    is the grid size (conservative, since neighbouring cells correlate). For a
    pooled region it is usually 1: an average is not a maximum, and charging it
    the full grid size sizes every study to zero.
    """
    return sharpe - expected_max_sharpe(se, trials)


def significance(sharpe: float, se: float, trials: int = 1) -> dict:
    """Raw and deflated figures side by side. Never report one without the other."""
    d = deflated_sharpe(sharpe, se, trials)
    return {
        "sharpe": sharpe,
        "se": se,
        "trials": trials,
        "noise_bar": expected_max_sharpe(se, trials),
        "deflated": d,
        "t_raw": sharpe / se if se > 0 else float("nan"),
        "t_deflated": d / se if se > 0 else float("nan"),
        "survives": d > 0 and (d / se if se > 0 else 0) > 1.96,
    }


# --------------------------------------------------------------------------
# Probability of backtest overfitting
# --------------------------------------------------------------------------


def _sharpe(x: np.ndarray) -> np.ndarray:
    """Annualization-free Sharpe per column; only the ranking matters here."""
    sd = x.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, x.mean(axis=0) / sd, -np.inf)


def pbo(returns: np.ndarray, n_splits: int = 10,
        max_partitions: int = 500) -> dict:
    """Probability that the in-sample winner underperforms out of sample.

    ``returns`` is (periods, configurations): one column per parameter
    combination, one row per period.

    The sample is cut into ``n_splits`` contiguous chunks and every balanced
    train/test partition is formed. On each, the best configuration in train is
    located and its rank in test recorded. PBO is the share of partitions where
    that choice lands below the test median.

    Readings: below ~0.2 is acceptable. Near 0.5 the selection carries no
    information, which is the situation pooling exists to handle rather than
    to hide.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns must be 2-D: (periods, configurations)")
    n_periods, n_configs = r.shape
    if n_configs < 2:
        raise ValueError("need at least 2 configurations to compare")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even so the halves are balanced")

    chunks = np.array_split(np.arange(n_periods), n_splits)
    half = n_splits // 2
    combos = list(itertools.combinations(range(n_splits), half))
    if len(combos) > max_partitions:
        step = len(combos) / max_partitions
        combos = [combos[int(i * step)] for i in range(max_partitions)]

    logits: List[float] = []
    below = 0
    for train_ids in combos:
        test_ids = [i for i in range(n_splits) if i not in train_ids]
        train = np.concatenate([chunks[i] for i in train_ids])
        test = np.concatenate([chunks[i] for i in test_ids])

        train_perf = _sharpe(r[train])
        test_perf = _sharpe(r[test])
        best = int(np.argmax(train_perf))

        # relative rank of that choice among test performances, in (0, 1)
        order = np.argsort(np.argsort(test_perf))
        rel = (order[best] + 1.0) / (n_configs + 1.0)
        if rel <= 0.5:
            below += 1
        logits.append(math.log(rel / (1.0 - rel)))

    return {
        "pbo": below / len(combos),
        "median_logit": float(np.median(logits)),
        "partitions": len(combos),
        "configurations": n_configs,
    }


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _selftest() -> int:
    rng = np.random.default_rng(4)
    n_periods, n_configs = 1000, 60

    print("Deflated Sharpe")
    print("-" * 58)
    se = 0.53
    for trials in (1, 100, 10_000, 250_000):
        bar = expected_max_sharpe(se, trials)
        print(f"  {trials:>8,} trials  ->  noise alone reaches {bar:.2f}")
    s = significance(1.9, se, 10_000)
    print(f"\n  a Sharpe of 1.90 after 10,000 trials deflates to "
          f"{s['deflated']:+.2f}  (survives: {s['survives']})")

    print("\nPBO on a world with NO edge anywhere")
    print("-" * 58)
    noise = rng.normal(0, 0.01, size=(n_periods, n_configs))
    res_null = pbo(noise, n_splits=10)
    print(f"  PBO {res_null['pbo']:.2f}   median logit "
          f"{res_null['median_logit']:+.2f}   ({res_null['partitions']} partitions)")
    print("  at or above 0.5 is the correct answer: nothing here generalizes.")
    print("  PBO on pure noise sits above 0.5, not exactly at it, because the")
    print("  in-sample winner is selected for luck that reverses out of sample.")

    print("\nPBO on a world where one configuration is genuinely better")
    print("-" * 58)
    real = rng.normal(0, 0.01, size=(n_periods, n_configs))
    real[:, 0] += 0.0015                      # a true, persistent edge
    res_real = pbo(real, n_splits=10)
    print(f"  PBO {res_real['pbo']:.2f}   median logit "
          f"{res_real['median_logit']:+.2f}   ({res_real['partitions']} partitions)")
    print("  low is the correct answer: the selection does generalize")

    assert expected_max_sharpe(se, 1) == 0.0
    assert deflated_sharpe(1.9, se, 10_000) < 1.9
    assert res_null["pbo"] > 0.3, "a null world should show near-chance PBO"
    assert res_real["pbo"] < res_null["pbo"], "a real edge should lower PBO"
    print("\nself-test OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sharpe", type=float)
    p.add_argument("--se", type=float)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.sharpe is None or args.se is None:
        p.error("--sharpe and --se are required (or use --selftest)")

    s = significance(args.sharpe, args.se, args.trials)
    print(f"  measured Sharpe    {s['sharpe']:+.3f}  (SE {s['se']:.3f}, "
          f"t {s['t_raw']:.2f})")
    print(f"  trials             {s['trials']:,}")
    print(f"  noise alone reaches{s['noise_bar']:+.3f}")
    print(f"  deflated Sharpe    {s['deflated']:+.3f}  (t {s['t_deflated']:.2f})")
    print(f"  survives at 5%     {'yes' if s['survives'] else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
