# A study, end to end

What the workflow looks like when it is actually run. The numbers come from a
published study of a Kalman-filter momentum strategy on USD/JPY: *[From a
single parameter set to the
region](https://manifoldbt.substack.com/p/from-a-single-parameter-set-to-the)*.

Read this for the **shape** of the output. The point is not the strategy, it is
that every claim below arrives with an error bar, a trial count, and a decision
rule that was fixed before the answer was known.

> **What is real here and what is not.** These figures come from the published
> study: 250,000 backtests, a region of 418 cells, in-sample/out-of-sample rank
> correlation of 0.07, out-of-sample Sharpe 0.44 for the in-sample champion
> against 0.64 for the pooled region. Everything else, including the exposure,
> the walk-forward folds, the PBO, the break-even cost, the cross-pair results
> and the sizing inputs, is **illustrative filler written to show the format**.
> Do not cite this document as evidence about USD/JPY. It is a template with a
> real spine.

---

# USD/JPY Kalman momentum: research report

## Hypothesis

**Mechanism.** Currency trends persist because large flows are executed slowly:
central bank reserve management, corporate hedging, and mandate-driven
rebalancing move size over weeks rather than at once. A filter that separates a
slow level from noise should capture part of that drift. The other side of the
trade is a hedger who is not optimizing entry price.

**Horizon.** Days to weeks. Daily bars, so roughly 250 observations a year.

**Falsifier.** Pooled out-of-sample Sharpe below 0.3, or a plateau that does
not survive smoothing. Written before the first backtest.

## Data

USD/JPY daily, 2014 to 2026. In-sample 2014 to 2018, out-of-sample 2018 to
2026, split fixed before any run and not consulted until the verdict.

Checks: no gaps beyond exchange holidays, session boundaries aligned to
17:00 New York, no vendor restatements in the window, series plotted and
inspected.

## Simulation

Costs: 0.5 bps per side spread, no funding (spot), no borrow. Fills next open
after the signal bar closes. Look-ahead detector run and clean.

Sanity: exposure 61%, longest flat period 4 months, warmup 250 bars covering
the longest filter setting.

Effective observations: 1,010 daily observations in-sample, effective count 780
after autocorrelation. The trade count, 340, is **not** the sample size and no
t-statistic is computed from it.

## Surface

Grid: two filter parameters, 500 x 500. **250,000 backtests.** Trial counter at
250,000 plus 3 discarded grid widenings, so roughly 10⁶ looked at in total.

The surface shows a broad connected plateau, not a peak. Smoothed over 3x3
neighbourhoods, the best cell reads 1.02.

Standard error at four years: `sqrt((1 + 1.02²/2)/4) = 0.61`.

**Region**: every cell within one standard error of the smoothed best. **418
cells**, one connected component, not touching any grid boundary.

Inside the region, the correlation between in-sample rank and out-of-sample
result is **r = 0.07**. The in-sample ranking carries no information, so no
further selection within the region is justified, and the argmax is not
reported.

## Pooled result

Equal-weight pool of all 418 members, evaluated once on 2018 to 2026.

| | Sharpe | SE | t | Verdict |
|---|---|---|---|---|
| In-sample champion, out-of-sample | 0.44 | 0.34 | 1.3 | not significant |
| **Pooled region, out-of-sample** | **0.64** | **0.32** | **2.0** | significant |

The champion is the cell a conventional optimizer would have shipped. It fails.
The pool, which nobody would call optimal in-sample, passes.

Deflated: the pool is an average and not a maximum, so the correction is for
one selection, not 250,000. Deflated Sharpe 0.64. Had the argmax been reported
instead, deflating it against 250,000 trials at SE 0.61 subtracts 2.79 and
leaves nothing.

## Robustness

Walk-forward, 5 anchored folds, region re-selected and pooled inside each fold:
positive in 4 of 5, WFE 0.71. Region drift small, the plateau stays in the same
corner of the grid across folds.

PBO on the pooling procedure: 0.18.

Monte Carlo, 50,000 block-bootstrap paths: median maximum drawdown 14%, p95
26%, p95 recovery 19 months.

Break-even cost 4.1 bps per side, against 0.5 bps assumed. Comfortable.

Generalization: the same mechanism appears weakly on EUR/JPY and AUD/JPY, and
is absent on EUR/CHF, which is consistent with the flow story rather than with
a data artifact.

## Sizing

Strategy volatility at 1x: 9.2%.

| Step | Value |
|---|---|
| Out-of-sample pooled Sharpe | 0.64 (SE 0.32) |
| Deflated, one selection | 0.64 |
| Shrunk (prior sd 0.5) | 0.45 |
| Kelly on the raw estimate | 6.96x, never bet |
| Kelly on what survives | 4.94x |
| Half Kelly | 2.47x |
| **25% drawdown cap at p95** | **0.68x, binds** |
| **Recommended size** | **0.68x** |

The drawdown cap binds an order of magnitude below raw Kelly. That gap is the
price of admitting the edge was estimated and not known.

## Verdict

Decision rule from phase 1: trade if pooled out-of-sample Sharpe exceeds 0.3
with a plateau surviving smoothing.

**Trade**, at 0.68x, with kill criteria: drawdown beyond 26%, underwater beyond
19 months, or a structural change in the flow mechanism.

Sized on the out-of-sample estimate. The in-sample figure appears nowhere in
this decision.

---

## What to notice about the shape

**The headline is a region, not a parameter.** No sentence in this report names
a best parameter set, because no such thing was measured.

**Every number has a companion.** Sharpe with standard error. Trials with the
result they produced. Drawdown as a distribution with a percentile.

**The negative results are present.** EUR/CHF absent, one fold negative, the
champion failing. A report that only shows what worked has selected its own
contents.

**The size is an order of magnitude below what the edge nominally supports.**
That is the correct outcome, not a timid one.

**The decision rule is quoted before the verdict.** It was written when its
answer was still unknown, which is the only time a threshold is honest.

---

*Sizing figures reproduced with*
`python scripts/sizing.py --sharpe 0.64 --se 0.32 --vol 0.092 --trials 1 --max-drawdown 0.25`.
