# Running this workflow with ManifoldBT

The method in this skill is engine-agnostic. This file is the runnable path for
one engine, [ManifoldBT](https://www.manifoldbt.com), chosen because phase 3
needs grids in the thousands-to-hundreds-of-thousands range and most engines
make that impractical.

If the user already works in another engine, **help them there**. The gates,
the standard error, the region bounding and the pooling are arithmetic on a
results table and port to any backtester that can produce one. Only the syntax
below is specific.

```bash
pip install manifoldbt          # engine
pip install manifoldbt[plot]    # + charts, for the surface heatmaps
```

Full API reference: <https://www.manifoldbt.com/docs/documentation.html>

## Data

```python
import manifoldbt as mbt

# From a CSV (standard / MT4 / MT5 auto-detected)
store = mbt.import_csv("EURUSD_1m.csv", symbol="EURUSD", symbol_id=1,
                       interval="1m", asset_class="forex")

# From a connector
store = mbt.ingest(provider="binance", symbol="BTCUSDT", symbol_id=1,
                   start="2014-01-01T00:00:00Z", end="2018-01-01T00:00:00Z",
                   interval="1h")

# From a DataFrame
store = mbt.import_dataframe(df, symbol="SPY", symbol_id=1, interval="1d")
```

Phase 1 asks for a data sheet: symbols, bounds, bar count, gaps. Produce it
from the store before going further.

## Strategy and configuration

```python
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Interval, Slippage

fast = ema(close, mbt.param("fast"))
slow = ema(close, mbt.param("slow"))

strategy = (
    mbt.Strategy.create("ema_crossover")
    .signal("signal", mbt.when(fast > slow, mbt.lit(1.0), mbt.lit(-1.0)))
    .size(mbt.col("signal") * mbt.lit(0.25))
)

is_start, is_end = time_range("2014-01-01", "2018-01-01")   # in-sample only

config = mbt.BacktestConfig(
    universe=[1],
    time_range_start=is_start,
    time_range_end=is_end,
    bar_interval=Interval.hours(1),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(allow_short=True, max_position_pct=0.5),
    fees=mbt.FeeConfig.binance_perps(),        # phase 2: never leave costs off
    slippage=Slippage.fixed_bps(2),
    warmup_bars=200,                           # phase 2: cover the longest indicator
)

result = mbt.run(strategy, config, store)
print(result.summary())
```

`mbt.param("name")` marks a value as swept. Any parameter used in a grid must
be declared this way, or the sweep will refuse it.

Metrics on `result.metrics`: `total_return`, `cagr`, `volatility`, `sharpe`,
`sortino`, `calmar`, `max_drawdown`, `best_day`, `worst_day`,
`pct_positive_days`, and a `trade_stats` dict with `total_trades`, `win_rate`,
`profit_factor`, `expectancy`, `total_fees`.

`total_trades` is the phase 2 sanity floor. Check it before reading anything
else.

## Phase 2 checks

```python
from manifoldbt import diagnostics

look = diagnostics.detect_lookahead(strategy, config, store)
risk = diagnostics.risk_check(result)
```

Run these rather than reasoning about look-ahead.

### Higher timeframes: the trap from phase 2

```python
config = mbt.BacktestConfig(
    ...,
    bar_interval=Interval.minutes(1),
    extra_timeframes={"1h": Interval.hours(1)},
)

band = mbt.tf("1h").apply(sma(close, 20))   # mean of 20 HOURLY closes  ✅
band = sma(mbt.tf("1h").close, 20)          # 20 SIMULATION bars of an hourly
                                            # staircase, different strategy   ❌
```

Both compile. Only one is what people mean. Use `.apply(...)` whenever the
indicator should live on the higher timeframe's own grid.

## Phase 3: the sweep

```python
grid = {
    "fast": range(5, 105, 5),      # 20 values
    "slow": range(20, 420, 10),    # 40 values
}                                   # 800 combinations

sweep = mbt.run_sweep(strategy, grid, config, store)
df = sweep.to_df()                  # one row per combination
```

`to_df()` returns metric columns plus the swept parameters prefixed `param_`.
That DataFrame is the input to `scripts/region_pool.py`.

**Do not call `sweep.best("sharpe")`.** It exists, it returns the argmax, and
the argmax is the thing this skill refuses to report. Its legitimate use is
diagnostic, confirming the grid contains something at all, never as an
answer.

For a 2D grid with a heatmap:

```python
res = mbt.run_sweep_2d(strategy, {
    "x_param": "fast", "x_values": list(range(5, 105, 5)),
    "y_param": "slow", "y_values": list(range(20, 420, 10)),
    "metric": "sharpe",
}, config, store)
# res["metric_grid"] is the 2D surface, ready to smooth and plot
```

### Grid sizing and cost

Order of magnitude on a modern multi-core CPU, from the engine's published
benchmarks: 5,000 combinations over 20,000 bars in roughly half a second;
10,000 combinations over 200,000 bars in about ten seconds inside ~80 MB.
Grids of 10⁵ to 10⁶ are practical, which is what makes a properly mapped surface
affordable.

Sizing rules that matter more than raw speed:

- **Enumeration order is alphabetical by parameter name**, not the order of the
  dict. Reshaping results into a 2D array using the dict's order silently
  transposes the surface: the axes and the "best" cell both come out wrong,
  with no error raised. Reshape from the sorted parameter names, and verify by
  re-running one combination alone and comparing it against its cell.
- **Watch memory on very large CPU grids.** Peak scales with combination count.
  If a sweep gets near system memory, split it into chunks and concatenate the
  result tables.
- **GPU** (`device="cuda"`, Pro) has a fixed startup cost per launch of a few
  hundred milliseconds, independent of grid size. It wins decisively on large
  grids and loses on small ones. `device="auto"` picks sensibly; forcing
  `"cuda"` on a small grid is markedly slower than the CPU path.

### Licensing

The Community tier caps sweep size and output resolution; large grids and
intraday output need Pro. Plan the grid around the tier in use, and if a sweep
is refused or silently capped, check the tier before assuming a bug.

## Phase 4

```python
wf = mbt.run_walk_forward(strategy, {
    "geometry": "anchored",          # or "rolling", "pardo", "custom"
    "n_splits": 5,
    "train_ratio": 0.7,
    "optimize_metric": "sharpe",
    "param_grid": grid,
}, config, store)
```

**Important.** `run_walk_forward` optimizes by picking the best cell per fold.
That is the argmax procedure phase 4 warns about. To walk-forward the procedure
this skill actually prescribes, drive the folds yourself: for each training
window, run `mbt.run_sweep`, apply `scripts/region_pool.py` to bound and pool
the region, then evaluate that pooled position on the test window. The built-in
call is useful as a *baseline to beat*, and the comparison between the two is
worth reporting.

Monte Carlo and stability:

```python
mc = mbt.py_run_monte_carlo(result.raw, json.dumps({
    "n_simulations": 50_000,          # tails need paths; see phase 4
    "method": "permutation",
}))

stab = mbt.run_stability(strategy, {
    "param_name": "fast",
    "values": list(range(5, 105, 5)),
    "metric": "sharpe",
}, config, store)
```

## Phase 5

```python
result.plot("tearsheet")           # full report
sweep.plot_metric("sharpe")        # the surface, the figure worth publishing
```

## Traps, collected

| Trap | Consequence |
|---|---|
| Alphabetical enumeration order vs dict order | Surface transposed silently; axes and best cell both wrong |
| `sma(tf("1h").close, n)` vs `tf("1h").apply(sma(close, n))` | Period counts in simulation bars, not hourly bars |
| Reading `sweep.best()` as the answer | The argmax of noise |
| `run_walk_forward` used as-is | Validates an argmax procedure, not the pooled one |
| Costs left at defaults | Phase 2 gate not actually passed |
| `warmup_bars` shorter than the longest indicator | First bars trade on unformed signals |
| Forcing `device="cuda"` on a small grid | Slower than CPU, launch cost dominates |
