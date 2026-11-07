"""Phase 3 of the quant-research skill: surface -> region -> pooled estimate.

Engine-agnostic. The input is a table of sweep results: one row per parameter
combination, one column per swept parameter, one metric column. Anything that
can produce that table works, ManifoldBT's ``sweep.to_df()`` included.

What it does, in order:

1. Reshape the rows into an N-dimensional grid over the swept parameters.
2. Smooth each cell over its 3x3(x3...) neighbourhood, so isolated spikes stop
   winning and robustness is rewarded.
3. Compute the standard error of the Sharpe ratio for the sample length.
4. Keep every cell within one standard error of the smoothed best. That set,
   not its maximum, is the output of the optimization.
5. Report the region's size, connectivity and position, and the correlation
   between in-sample and out-of-sample results inside it (usually ~0, which is
   the whole reason for pooling).
6. Pool the region equal-weight.

Usage as a library::

    from region_pool import select_region, region_summary, pool_series

    reg = select_region(df, param_cols=["param_fast", "param_slow"],
                        metric="sharpe", years=4.0)
    print(region_summary(reg))

Usage from the shell::

    python region_pool.py sweep.csv --params param_fast param_slow \\
                                    --metric sharpe --years 4
    python region_pool.py --selftest

Requires numpy. pandas is used if present, but plain dicts of lists work too.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Standard error
# --------------------------------------------------------------------------


def sharpe_se(sharpe: float, periods: float) -> float:
    """Standard error of a Sharpe ratio estimate.

    ``SE = sqrt((1 + S^2 / 2) / T)``

    ``sharpe`` and ``periods`` must be expressed at the SAME frequency. Sharpe
    ratios are normally quoted annualized, so ``periods`` is then the number of
    YEARS in the sample. Passing a bar count for an annualized Sharpe
    understates the error by a large factor and makes every difference look
    significant, which is the mistake this whole module exists to prevent.

    >>> round(sharpe_se(1.0, 4.0), 3)
    0.612
    >>> round(sharpe_se(1.0, 10.0), 3)
    0.387
    """
    if periods <= 0:
        raise ValueError("periods must be > 0")
    return math.sqrt((1.0 + (sharpe ** 2) / 2.0) / periods)


def t_statistic(sharpe: float, periods: float) -> float:
    """Sharpe divided by its standard error. |t| < 1.96 is not significant."""
    return sharpe / sharpe_se(sharpe, periods)


# --------------------------------------------------------------------------
# Grid construction and smoothing
# --------------------------------------------------------------------------


@dataclass
class Grid:
    """A sweep's metric values arranged on the parameter lattice."""

    values: np.ndarray                    # N-d array of the metric, NaN where missing
    axes: List[np.ndarray]                # sorted unique value per parameter
    param_cols: List[str]
    index: np.ndarray                     # N-d array of row indices, -1 where missing


def _column(table: Any, name: str) -> np.ndarray:
    """Read one column from a DataFrame or a dict of sequences."""
    if hasattr(table, "columns"):
        if name not in table.columns:
            raise KeyError(
                f"column {name!r} not found; available: {list(table.columns)}"
            )
        return np.asarray(table[name].to_numpy())
    if name not in table:
        raise KeyError(f"key {name!r} not found; available: {list(table)}")
    return np.asarray(table[name])


def build_grid(table: Any, param_cols: Sequence[str], metric: str) -> Grid:
    """Reshape a flat results table into an N-dimensional grid.

    The axes are built from the sorted unique values of each parameter column,
    and every row is placed by looking its coordinates up. This is deliberate:
    reshaping by row order instead is how a surface gets silently transposed,
    because engines do not all enumerate a grid in the order the parameter dict
    was written in.
    """
    param_cols = list(param_cols)
    if not param_cols:
        raise ValueError("at least one parameter column is required")

    cols = [_column(table, c) for c in param_cols]
    metric_values = np.asarray(_column(table, metric), dtype=float)
    n_rows = len(metric_values)

    axes = [np.unique(c) for c in cols]
    shape = tuple(len(a) for a in axes)

    values = np.full(shape, np.nan, dtype=float)
    index = np.full(shape, -1, dtype=np.int64)

    positions = [
        {v: i for i, v in enumerate(axis)} for axis in axes
    ]
    for row in range(n_rows):
        coord = tuple(positions[d][cols[d][row]] for d in range(len(param_cols)))
        values[coord] = metric_values[row]
        index[coord] = row

    filled = int(np.count_nonzero(~np.isnan(values)))
    if filled < n_rows:
        # duplicate coordinates: the table has more rows than lattice cells
        raise ValueError(
            f"{n_rows} rows collapsed into {filled} cells; the parameter "
            "columns do not uniquely identify a row"
        )

    return Grid(values=values, axes=axes, param_cols=param_cols, index=index)


def smooth(values: np.ndarray, radius: int = 1) -> np.ndarray:
    """Mean of each cell's neighbourhood, ignoring NaN and clipping at edges.

    With ``radius=1`` this is the 3x3 mean in two dimensions, the 3x3x3 mean in
    three, and so on. Smoothing is what separates a plateau from a spike: a
    peak that does not survive it was never a peak.
    """
    if radius < 1:
        return values.astype(float, copy=True)

    finite = np.nan_to_num(values, nan=0.0)
    mask = (~np.isnan(values)).astype(float)

    total = np.zeros_like(finite, dtype=float)
    count = np.zeros_like(finite, dtype=float)

    offsets = itertools.product(range(-radius, radius + 1), repeat=values.ndim)
    for offset in offsets:
        total += _shift_clamped(finite, offset)
        count += _shift_clamped(mask, offset)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / count, np.nan)
    # a cell with no data stays empty even if its neighbours have some
    out[np.isnan(values)] = np.nan
    return out


def _shift_clamped(arr: np.ndarray, offset: Sequence[int]) -> np.ndarray:
    """Shift by ``offset``, repeating the edge rather than wrapping."""
    out = arr
    for axis, delta in enumerate(offset):
        if delta == 0:
            continue
        idx = np.clip(np.arange(arr.shape[axis]) - delta, 0, arr.shape[axis] - 1)
        out = np.take(out, idx, axis=axis)
    return out


# --------------------------------------------------------------------------
# Region selection
# --------------------------------------------------------------------------


@dataclass
class Region:
    """The set of parameter combinations that survive the 1-SE rule."""

    grid: Grid
    smoothed: np.ndarray
    mask: np.ndarray                      # boolean, same shape as the grid
    threshold: float
    best_smoothed: float
    standard_error: float
    n_sigma: float
    metric: str
    years: float
    rows: List[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return int(self.mask.sum())

    @property
    def coverage(self) -> float:
        finite = int(np.count_nonzero(~np.isnan(self.grid.values)))
        return self.size / finite if finite else 0.0

    def params(self) -> List[Dict[str, Any]]:
        """Parameter values of every member, as a list of dicts."""
        out = []
        for coord in zip(*np.nonzero(self.mask)):
            out.append(
                {
                    name: self.grid.axes[d][coord[d]]
                    for d, name in enumerate(self.grid.param_cols)
                }
            )
        return out

    def raw_metric_values(self) -> np.ndarray:
        return self.grid.values[self.mask]

    def touches_edge(self) -> bool:
        """True if the region reaches a grid boundary, meaning the grid was too narrow."""
        for axis in range(self.mask.ndim):
            lo = np.take(self.mask, 0, axis=axis)
            hi = np.take(self.mask, self.mask.shape[axis] - 1, axis=axis)
            if lo.any() or hi.any():
                return True
        return False

    def n_components(self) -> int:
        """Number of connected blobs (face-adjacency flood fill).

        One large component is what a real edge looks like. Several scattered
        islands are a warning: real edges occupy neighbourhoods.
        """
        seen = np.zeros_like(self.mask, dtype=bool)
        components = 0
        for start in zip(*np.nonzero(self.mask)):
            if seen[start]:
                continue
            components += 1
            stack = [start]
            seen[start] = True
            while stack:
                cell = stack.pop()
                for axis in range(self.mask.ndim):
                    for delta in (-1, 1):
                        nxt = list(cell)
                        nxt[axis] += delta
                        if not (0 <= nxt[axis] < self.mask.shape[axis]):
                            continue
                        nxt_t = tuple(nxt)
                        if self.mask[nxt_t] and not seen[nxt_t]:
                            seen[nxt_t] = True
                            stack.append(nxt_t)
        return components


def select_region(
    table: Any,
    param_cols: Sequence[str],
    metric: str = "sharpe",
    years: float = 4.0,
    n_sigma: float = 1.0,
    radius: int = 1,
) -> Region:
    """Bound the stable region around the smoothed best cell.

    Keeps every combination whose SMOOTHED metric lies within ``n_sigma``
    standard errors of the smoothed best. The result is a set, typically with
    hundreds of members. That set is the output of the optimization; its
    maximum is not.
    """
    grid = build_grid(table, param_cols, metric)
    smoothed = smooth(grid.values, radius=radius)

    if np.all(np.isnan(smoothed)):
        raise ValueError("every cell is NaN; nothing to select")

    best = float(np.nanmax(smoothed))
    se = sharpe_se(best, years)
    threshold = best - n_sigma * se

    mask = ~np.isnan(smoothed) & (smoothed >= threshold)
    rows = [int(grid.index[c]) for c in zip(*np.nonzero(mask))]

    return Region(
        grid=grid,
        smoothed=smoothed,
        mask=mask,
        threshold=threshold,
        best_smoothed=best,
        standard_error=se,
        n_sigma=n_sigma,
        metric=metric,
        years=years,
        rows=rows,
    )


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation, no scipy required. NaN pairs are dropped."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    keep = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom else float("nan")


def _rank(v: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(len(v), dtype=float)
    # average tied ranks
    sorted_v = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.arange(i, j + 1).mean()
        i = j + 1
    return ranks


def is_oos_correlation(
    region: Region,
    table: Any,
    oos_metric: str,
) -> float:
    """Rank correlation between in-sample and out-of-sample metrics, inside the region.

    Expect something near zero. That is not a disappointing result, it is the
    justification for pooling: if the in-sample ranking carried information,
    selecting on it would be rational. It does not, so it is not.

    A materially positive value is unusual and worth investigating before
    exploiting, since a leak explains it more often than a discovery does.
    """
    is_vals = np.asarray(_column(table, region.metric), dtype=float)[region.rows]
    oos_vals = np.asarray(_column(table, oos_metric), dtype=float)[region.rows]
    return spearman(is_vals, oos_vals)


def pool_series(series: Sequence[Sequence[float]]) -> np.ndarray:
    """Equal-weight average of the region members' series (equity, returns, positions).

    Equal weight, deliberately. Weighting by in-sample performance would
    reintroduce exactly the selection that pooling exists to remove.
    """
    arr = np.asarray([np.asarray(s, dtype=float) for s in series])
    if arr.ndim != 2:
        raise ValueError("all series must have the same length")
    return np.nanmean(arr, axis=0)


def region_summary(region: Region, table: Any = None, oos_metric: Optional[str] = None) -> str:
    """Human-readable verdict on the region. Print this into the report."""
    raw = region.raw_metric_values()
    mean_raw = float(np.nanmean(raw))
    best_raw = float(np.nanmax(region.grid.values))

    lines = [
        f"Region on '{region.metric}'  ({region.years:g} years of sample)",
        "-" * 58,
        f"  grid cells                {int(np.count_nonzero(~np.isnan(region.grid.values)))}",
        f"  standard error            {region.standard_error:.3f}",
        f"  smoothed best             {region.best_smoothed:.3f}",
        f"  threshold ({region.n_sigma:g} SE)          {region.threshold:.3f}",
        f"  region size               {region.size}  ({region.coverage:.1%} of grid)",
        f"  connected components      {region.n_components()}",
        f"  touches grid edge         {'YES - widen the grid' if region.touches_edge() else 'no'}",
        "",
        f"  region mean (raw)         {mean_raw:.3f}   <- report this",
        f"  in-sample argmax (raw)    {best_raw:.3f}   <- do not report this",
        f"  t-statistic of the mean   {t_statistic(mean_raw, region.years):.2f}",
    ]

    if table is not None and oos_metric is not None:
        r = is_oos_correlation(region, table, oos_metric)
        lines += [
            "",
            f"  IS/OOS rank correlation   {r:+.3f}",
            "    near zero => in-sample ranking carries no information;"
            " pooling is the correct response",
        ]

    warnings = []
    if region.size < 5:
        warnings.append("region has fewer than 5 members: no plateau, no edge to bound")
    if region.n_components() > 3:
        warnings.append("scattered region: looks like an artifact, not a plateau")
    if region.touches_edge():
        warnings.append("region reaches a grid boundary: the true region extends past the grid")
    if region.coverage > 0.9:
        warnings.append("region covers >90% of the grid: the grid is too narrow, or the parameter does not matter")
    if warnings:
        lines += [""] + [f"  ! {w}" for w in warnings]

    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(returns.mean() / sd * math.sqrt(periods_per_year))


def _synthetic_world(rng: np.random.Generator, n_days: int = 1008):
    """A grid of strategies with a known true edge, over IS and OOS periods.

    Two modelling choices carry the whole demonstration, and both are the
    claim being illustrated rather than incidental detail:

    1. **The true edge is a plateau, not a peak.** Inside a region of the grid
       every parameterization expresses the same edge and earns the same true
       Sharpe; outside, nothing. This is what "one edge, many
       parameterizations" means. If the truth were a single sharp peak, the
       argmax would be the right estimator and this method would be wrong.

    2. **Each cell is a return stream, not a Sharpe number.** The gain from
       pooling comes from averaging imperfectly correlated STREAMS, which
       lowers the variance of the combined curve. Averaging Sharpe ratios
       instead misses the effect entirely.

    Estimation noise is large relative to the edge, as it is in practice: with
    800 cells, the luckiest pure-noise cell routinely outscores every genuine
    one in-sample.
    """
    fast = np.arange(5, 105, 5)          # 20 values
    slow = np.arange(20, 420, 10)        # 40 values
    ff, ss = np.meshgrid(fast, slow, indexing="ij")

    # a plateau with soft edges: same true Sharpe everywhere inside it
    dist = np.sqrt(((ff - 40) / 22.0) ** 2 + ((ss - 200) / 90.0) ** 2)
    true_sharpe = (0.90 / (1.0 + np.exp((dist - 1.0) * 12.0))).ravel()
    n_cells = true_sharpe.size

    sigma_d = 0.01                        # daily vol
    rho = 0.50                            # correlation across parameterizations

    def period(n: int) -> np.ndarray:
        common = rng.normal(0, 1, size=(n, 1))
        idio = rng.normal(0, 1, size=(n, n_cells))
        shocks = math.sqrt(rho) * common + math.sqrt(1 - rho) * idio
        drift = true_sharpe * sigma_d / math.sqrt(252)
        return drift + sigma_d * shocks   # (days, cells)

    return fast, slow, true_sharpe, period(n_days), period(n_days)


def _selftest(replications: int = 25) -> int:
    """Does the method beat the argmax? Measured, not asserted.

    A single replication proves nothing: the champion can get lucky twice.
    The comparison is run many times and the medians reported.
    """
    rng = np.random.default_rng(7)
    champ_scores: List[float] = []
    pool_scores: List[float] = []
    sizes: List[int] = []
    correlations: List[float] = []
    printed = False
    true_plateau = 0.0

    for rep in range(replications):
        fast, slow, true_sharpe, is_ret, oos_ret = _synthetic_world(rng)
        true_plateau = float(true_sharpe.max())

        is_sharpe = np.array([_annualized_sharpe(is_ret[:, i]) for i in range(is_ret.shape[1])])
        oos_sharpe = np.array([_annualized_sharpe(oos_ret[:, i]) for i in range(oos_ret.shape[1])])

        rows: Dict[str, Any] = {
            "param_fast": np.repeat(fast, len(slow)).astype(float),
            "param_slow": np.tile(slow, len(fast)).astype(float),
            "sharpe": is_sharpe,
            "sharpe_oos": oos_sharpe,
        }

        region = select_region(rows, ["param_fast", "param_slow"], "sharpe", years=4.0)
        sizes.append(region.size)
        correlations.append(is_oos_correlation(region, rows, "sharpe_oos"))

        champion = int(np.argmax(is_sharpe))
        champ_scores.append(oos_sharpe[champion])

        # pool the return STREAMS of the region, then measure the pooled curve
        pooled_oos = pool_series([oos_ret[:, i] for i in region.rows])
        pool_scores.append(_annualized_sharpe(pooled_oos))

        if not printed:
            print(region_summary(region, rows, "sharpe_oos"))
            print()
            printed = True

    champ = np.array(champ_scores)
    pooled = np.array(pool_scores)
    wins = float((pooled > champ).mean())

    print(f"Out-of-sample comparison over {replications} replications")
    print("-" * 58)
    print(f"  true Sharpe on the plateau          {true_plateau:.2f}")
    print(f"  in-sample champion, OOS   median    {np.median(champ):.3f}"
          f"   (sd {champ.std(ddof=1):.3f})")
    print(f"  pooled region,     OOS    median    {np.median(pooled):.3f}"
          f"   (sd {pooled.std(ddof=1):.3f})")
    print(f"  pooling beats the champion          {wins:.0%} of the time")
    print(f"  median region size                  {np.median(sizes):.0f} cells")
    print(f"  median IS/OOS rank corr in region   {np.median(correlations):+.3f}")

    assert abs(sharpe_se(1.0, 4.0) - 0.6124) < 1e-3
    assert np.median(pooled) > np.median(champ), (
        "pooling failed to beat the argmax on synthetic data where it should"
    )
    print("\nself-test OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv", nargs="?", help="sweep results, one row per combination")
    parser.add_argument("--params", nargs="+", help="parameter column names")
    parser.add_argument("--metric", default="sharpe", help="in-sample metric column")
    parser.add_argument("--oos-metric", default=None, help="out-of-sample metric column, if present")
    parser.add_argument("--years", type=float, default=4.0, help="sample length in years")
    parser.add_argument("--n-sigma", type=float, default=1.0, help="region width in standard errors")
    parser.add_argument("--radius", type=int, default=1, help="smoothing radius (1 = 3x3)")
    parser.add_argument("--selftest", action="store_true", help="run on a synthetic surface")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.csv or not args.params:
        parser.error("a CSV and --params are required (or use --selftest)")

    try:
        import pandas as pd
    except ImportError:
        print("reading a CSV requires pandas: pip install pandas", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv)
    region = select_region(df, args.params, args.metric, years=args.years,
                           n_sigma=args.n_sigma, radius=args.radius)
    print(region_summary(region, df, args.oos_metric))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
