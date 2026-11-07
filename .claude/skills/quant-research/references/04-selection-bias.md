# Phase 4: correcting for the search

Phase 3 searched thousands of cells. That search manufactured performance.
This phase measures how much and subtracts it.

## 1. Deflated Sharpe

Under a pure-noise null, the expected **maximum** Sharpe across `N` independent
trials grows with `N`. Approximately:

```
E[max SR] ≈ SE × ( (1-γ)·Z(1 - 1/N) + γ·Z(1 - 1/(N·e)) )
```

with `γ ≈ 0.5772` (Euler-Mascheroni), `Z` the inverse normal CDF, and `SE` the
standard error from phase 3.

The practical shape of it, computed at `SE = 0.53`, which is a four-year sample
(`scripts/selection_bias.py --selftest`):

| Trials | Expected max Sharpe from noise alone |
|---|---|
| 1 | 0.00 |
| 100 | 1.34 |
| 10,000 | 2.05 |
| 250,000 | 2.43 |

A strategy scoring 1.9 in-sample after a 10,000-cell sweep deflates to -0.15:
it underperformed what pure noise delivers at that trial count. This is not a
technicality, it is the single most common reason strategies that backtest well
fail live.

Note what this does NOT apply to. The penalty is the expected maximum of N
draws, so it corrects a selected argmax. A pooled region is an average, not a
maximum, and charging it the full grid size double-counts the correction and
sizes every study to zero. Pass the number of independent selections made,
which after pooling is usually one.

Report the deflated figure **next to** the raw one, with `N` stated. Never the
raw one alone.

Note that trials are not independent, since neighbouring cells are highly
correlated, so the effective `N` is smaller than the count. Estimate it from
the number of *distinct plateaus* or via the correlation structure of the
grid's returns. Using the raw count is conservative, which is the right
direction to err.

## 2. Probability of backtest overfitting (PBO)

Split the sample into `S` chunks (typically 16), form all balanced train/test
partitions, select the best cell on each train half, and record its rank on the
matching test half. PBO is the fraction of partitions where the in-sample
winner lands below median out of sample.

Readings: PBO under ~0.2 is acceptable; near 0.5 means in-sample selection is
worthless, which is precisely the situation the pooling in phase 3 addresses.

Run PBO on the **selection procedure**, not on a single cell. If the procedure
is "bound a region and pool it", test that procedure. Testing an argmax
procedure you do not intend to use gives an answer about the wrong thing.

Measured on synthetic worlds by `scripts/selection_bias.py --selftest`: a world
with no edge anywhere returns PBO 0.69, and a world containing one genuinely
better configuration returns 0.10. Note that the null case sits above 0.5 and
not at it, because the in-sample winner is actively selected for luck that then
reverses.

## 3. Walk-forward

Split into successive train/test folds and re-run the *entire* selection inside
each training fold.

**Geometry.** Anchored (training window grows from a fixed start) or rolling
(fixed-length window slides). Rolling adapts to regime change; anchored uses
more data. Report which, and the window lengths.

**The critical detail.** Inside each fold, re-run phase 3: smooth, bound the
region, pool. Carry the *pooled* result into the test fold. A walk-forward that
re-picks an argmax per fold validates a procedure nobody should run, and it
will report a better number than the procedure you would actually trade.

**Warmup.** Test segments need indicator warmup from data preceding the segment,
or the first bars of every fold trade on unformed indicators.

**Non-overlapping tests.** Test windows should tile end to end. Overlapping
test windows reuse observations and inflate the apparent sample.

**Walk-forward efficiency**: `WFE = OOS performance / IS performance`. Below
~0.5 means the fitting is not transferring. Note that WFE is itself a noisy
ratio of two noisy numbers; read its trend across folds, not its exact value.

**Region drift.** Track where the region sits fold to fold. A region that stays
put is evidence of a stable mechanism. A region that jumps around is evidence
of fitting, even when each fold's numbers look acceptable.

## 4. Monte Carlo

Point estimates hide the distribution that matters.

**Trade-order permutation.** Shuffle trade sequence, rebuild equity. Tests
whether the result depends on a lucky ordering. Breaks autocorrelation, so it
is a lower bound on realistic path risk.

**Block bootstrap.** Resample contiguous blocks of returns, preserving local
autocorrelation. Better for drawdown questions.

**Parameter perturbation.** Jitter the parameters within the region and observe
the spread. This should already look tame after pooling.

**Path count.** Match it to the rarity of the event. A median is stable at
1,000 paths. A tail is not: `P(drawdown > 50%)` reads exactly 0.000 at 1,000
paths and only becomes measurable past ~30,000. Reporting "0% chance of a 50%
drawdown" from 1,000 paths is a measurement failure presented as a safety
guarantee.

Report intervals, not means: 5th percentile equity, expected maximum drawdown,
P(drawdown > X), time to recovery.

## 5. Cost sensitivity

Sweep fees and slippage as an explicit axis and plot performance against cost.
What you want to know is the **break-even cost**: the level at which the edge
reaches zero.

- Break-even far above realistic costs: robust.
- Break-even near realistic costs: it is dead; the backtest just has not
  noticed yet.
- Edge that only exists at zero cost: not an edge, an accounting identity.

High-turnover strategies fail this test most often, and they are also the ones
that look best when costs are under-modelled. That is not a coincidence.

## 6. Generalization

- **Sub-periods.** Split the in-sample into halves or thirds. An edge present
  in one third only is a description of that third.
- **Other assets.** A mechanism that is real usually shows up, weakly, on
  related instruments. Complete absence elsewhere is informative.
- **Other regimes.** High and low volatility, trending and ranging, pre- and
  post-2020.
- **Perturbed data.** Add small noise to prices. A strategy whose result
  collapses under 1bp of noise is fitted to specific ticks.

Each of these adds to the trial counter. Count them.

## 7. Gate

- [ ] Deflated Sharpe reported beside the raw figure, with `N`
- [ ] PBO run on the actual selection procedure
- [ ] Walk-forward with region re-selection and pooling inside each fold
- [ ] Monte Carlo with a path count matched to the rarest reported event
- [ ] Break-even cost level identified
- [ ] Sub-period and cross-asset generalization tested
