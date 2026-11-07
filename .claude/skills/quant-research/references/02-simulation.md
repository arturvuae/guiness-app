# Phase 2: honest simulation

The goal of this phase is a measurement you can believe. Not a good result: a
believable one.

## 1. Costs

A backtest without costs is a drawing of a strategy, not a test of one. Four
components, all of which have killed real edges:

**Fees.** Maker/taker, per venue, per instrument. Round-trip, not one way.

**Slippage.** The gap between the price the signal saw and the price actually
paid. Model it as a spread crossed plus an impact term. A fixed basis-point
assumption is fine, but state it and sweep it (phase 4).

**Funding.** Perpetual futures pay or receive funding every interval. Over a
year, funding often dominates the strategy's gross PnL. Ignoring it on a
persistently long crypto strategy overstates returns badly.

**Borrow.** Short positions cost carry, and the cost is highest exactly on the
names most worth shorting.

Sizing matters here: a strategy trading 5% of a bar's volume gets a different
fill than one trading 0.05%. If the sweep's best cells are the highest-turnover
ones, suspect that costs are under-modelled.

**Calibration test.** Take the cost assumption to a practitioner or to the
venue's own fee schedule. If nobody can defend the number, it is decoration.

## 2. Fill semantics

Decide explicitly, and write it in the report:

- **Which bar.** A signal computed on bar `t` fills on `t+1`, unless there is a
  documented reason otherwise. Filling on `t` at the close that produced the
  signal is look-ahead wearing a disguise.
- **Which price.** Open, close, VWAP, a computed level. "Next open" is the
  conservative default; a limit at a computed level requires modelling whether
  it would have been reached *and* filled.
- **Inside the bar.** When a stop and a target both sit inside one bar's range,
  which triggers first is unknowable from OHLC. Assume the adverse one. If the
  result depends on this assumption, the result is about bar resolution, not
  about the strategy.
- **Partial fills and capacity.** At size, the entire fill does not happen at
  one price.

## 3. Bar semantics and look-ahead

Look-ahead is rarely a blunt `shift(-1)`. It arrives through:

**Readability timing.** A bar's value is knowable only after that bar closes.
Any indicator that reads the current bar's close and trades at that same close
has used information it did not have.

**Higher timeframes.** The classic trap: computing an indicator on a
step-held higher-timeframe series versus computing it on that timeframe's own
grid. A 20-period average of hourly closes held across minute bars is *not* a
20-hour average. On a 1-minute simulation it is a 20-*minute* smoothing of an
hourly staircase. These are different strategies. Know which one the code
expresses.

**Normalization across the full sample.** Z-scoring, min-max scaling, or
standardizing using statistics computed over the whole period leaks the future
into every bar. Use expanding or rolling windows.

**Post-hoc universe.** Selecting the symbols that ended up liquid, or listed,
or surviving.

**Revised data.** Macro releases and fundamentals as they read *today*, not as
they read then.

Run the automated look-ahead detector rather than reasoning about it. Reasoning
about look-ahead is how look-ahead survives.

## 4. Sanity floors

Before any metric is quoted, check the run is not degenerate:

| Check | Floor | Why |
|---|---|---|
| **Effective** observations | ~30 minimum | The real floor. See below, it is not the trade count |
| Exposure | Not ~0%, not ~100% | Always-in is a benchmark, never-in is a bug |
| Equity variation | Non-flat | Flat equity means the strategy never traded |
| Warmup | Indicators fully formed | The first N bars produce garbage signals |
| Longest flat period | Reported | A strategy idle for 3 of 4 years was tested on 1 |

**The floor is on effective observations, not on trades.** A trade count is a
measure of activity, and the strategy controls it: trading twice as often
doubles the count without adding information. Trades that overlap in time or
sit in correlated instruments are the same bet recorded several times.

The gap is not small. On a strategy built to have exactly zero alpha, 2,016
trades carried 237 effective observations, and the t-statistic computed on the
raw count declared significance in 47% of runs against a nominal 5%. Compute
the effective count, and never quote a t-statistic built on trades.

→ `references/07-effective-sample.md`, with `scripts/effective_n.py`

## 5. Reading the first result

A single backtest answers one question: *is this measurement plausible?*

- Sharpe > 3 on a single set: treat as a bug report. Look for look-ahead,
  missing costs, or a data artifact, in that order.
- Perfectly smooth equity: almost always a leak.
- All profit from one week: not a strategy, an event.
- Best trades clustered at data boundaries: stitching artifact.

If it survives all of that, it still is not a result. It is one cell of a
surface, and phase 3 is where its meaning gets established.

## 6. Gate

- [ ] Fees, slippage, funding, borrow applied and justified
- [ ] Fill rule stated (bar, price, intra-bar tie-break)
- [ ] Look-ahead detector run and clean
- [ ] Trade count, exposure, warmup, flat periods all pass
- [ ] Any implausibly good metric investigated before being reported
