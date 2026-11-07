# Phase 5b: sizing the bet

Phases 3 and 4 answered whether an edge exists and how much of it the search
manufactured. This one answers the question that actually moves money: how much
to bet.

It gets its own reference because it is where the largest single-decision
losses happen. A strategy sized wrong destroys capital faster than a strategy
chosen wrong, and it does so while being *right about the edge*.

## 1. The core problem

Kelly's formula gives the growth-optimal leverage for a return stream:

```
L* = S / sigma
```

with `S` the Sharpe ratio and `sigma` the annualized volatility. It maximizes
long-run log wealth, and it is correct, under one assumption that never holds:
**that `S` and `sigma` are known.**

They are estimated. From phase 3, a Sharpe measured over four years carries a
standard error around 0.5. Feeding a point estimate into a formula that assumes
exactness is not aggressive investing, it is a category error.

The consequence is asymmetric, which is what makes it dangerous. Expected log
growth at leverage `L` under the *true* parameters is:

```
g(L) = L·S·sigma - (L·sigma)² / 2
```

Under-betting costs a little growth, linearly. Over-betting costs
quadratically, and past **twice** the Kelly leverage expected growth turns
**negative**: a strategy with a genuine edge that loses money forever.

Measured on synthetic data where the true Sharpe is 0.50 and known
(`scripts/sizing.py --selftest`, 400 replications, four-year estimate then ten
years of trading):

| Policy | Leverage | CAGR p50 | CAGR p10 | maxDD p90 | Lost half |
|---|---|---|---|---|---|
| Full Kelly on the estimate | 3.03x | 2.2% | -24.3% | 100% | 19% |
| Half Kelly on the estimate | 1.51x | 4.3% | -0.3% | 84% | 6% |
| Half Kelly on the shrunk estimate | 0.71x | 2.9% | 0.0% | 54% | 1% |

Full Kelly on a four-year estimate loses half the account in a fifth of lives,
and earns *less* in the median than betting a fifth of the size. Read the p10
and ruin columns, not the median: every policy looks acceptable in the middle.

## 2. The three corrections, in order

Each addresses a different failure and they stack. `scripts/sizing.py`
implements the chain.

**Deflate for the selection.** If the number reported is the best of many, part
of its height is the search. Subtract the expected maximum of that many noise
draws (phase 4).

The critical detail, easy to get backwards: this penalty applies to a
**selected maximum only**. A pooled region is an average, not a maximum, so
passing the full grid size after pooling double-charges the study and sizes it
to zero. Pass the number of independent *selections* made, which after pooling
is usually one.

**Shrink for the noise.** A noisy estimate should barely move a prior belief.
With a prior centred on zero and standard deviation `tau`:

```
S_posterior = S_measured · tau² / (tau² + SE²)
```

At `tau = 0.5` and `SE = 0.5`, a measured Sharpe of 1.0 becomes 0.5. That is
not pessimism, it is what the measurement actually supports. Raise `tau` only
for a reason that is not "my backtest said so".

**Bet a fraction of what survives.** Half Kelly on the shrunk estimate is the
defensible default. It covers what neither correction addresses: the model is
wrong in ways nobody estimated, volatility is itself a moving estimate, and
regimes change. A quarter is for a fragile estimate. Above half needs an
argument.

## 3. The cap that usually binds

Kelly maximizes growth for an investor who can sit through any drawdown. Nobody
can. Capital gets withdrawn, mandates get pulled, and the person running the
strategy stops trusting it at exactly the wrong moment. A size implying an 80%
drawdown is not a size, it is a plan to abandon the strategy near the bottom.

So the recommended size is the **smallest** of three caps:

1. Fractional Kelly on the shrunk estimate
2. The leverage whose expected maximum drawdown stays within what the capital
   can actually survive
3. A hard leverage limit from the mandate, the broker, or the venue

In practice the drawdown cap binds most often. Worked example, a pooled OOS
Sharpe of 0.64 with SE 0.32 at 15% volatility:

```
$ python sizing.py --sharpe 0.64 --se 0.32 --vol 0.15 --trials 1

  after shrinking           +0.454
  Kelly on the raw estimate  4.27x   <- never bet this
  Kelly on what survives     3.03x

  caps, smallest wins
    0.5 Kelly                1.51x
    35% drawdown tolerance   0.62x
    hard cap                 3.00x

  RECOMMENDED SIZE           0.62x   (bound by drawdown tolerance)
```

From 4.27x to 0.62x, a factor of seven, with no change to the measured edge.
That gap is the price of admitting the edge was estimated rather than known.

## 4. Volatility targeting

Sizing to a constant volatility rather than a constant notional keeps risk
stable as market conditions change:

```
scale_t = target_vol / realized_vol_t
```

Three things to get right:

- **Cap the scale.** Realized volatility sits in the denominator, so the
  position explodes exactly when markets are quietest, which is reliably just
  before they are not.
- **The forecast lags.** Realized volatility is backward-looking, so the
  scaling arrives late to every regime change, which is when it matters.
- **It is not free.** Rescaling generates turnover, and turnover pays costs.
  Sweep the rebalancing threshold as a parameter rather than rebalancing daily
  out of habit.

## 5. At portfolio level

Individual sizes do not add up to a portfolio.

**Size by risk contribution, not by capital.** Equal capital across correlated
strategies is a concentrated bet wearing a diversified label.

**Correlations rise in stress.** Size for the stressed correlation matrix, not
the sample one. The historical estimate is the optimistic case by construction.

**Effective breadth.** `N` strategies at average pairwise correlation `rho`
behave like about `N / (1 + (N-1)·rho)` independent bets. Ten strategies at
`rho = 0.6` are roughly 1.5 bets, and sizing them as ten is how a book ends up
with one position under many names.

**Capacity.** At what size does the edge disappear into its own market impact?
A strategy whose backtest assumed 0.05% of volume and is deployed at 5% is not
the strategy that was tested.

## 6. What never to do

| Never | Because |
|---|---|
| Size on the in-sample estimate | It is inflated by exactly the search that produced it |
| Full Kelly on any estimated edge | Measured above: 19% chance of losing half, for no median gain |
| Size up after a good run | The edge did not change; the sample did |
| Size up to hit a return target | The target has no bearing on what the edge supports |
| Ignore the drawdown cap because "I can handle it" | The cap protects against the version of you that is 40% down |
| Reuse a size after the strategy changed | A different strategy needs a different estimate |

## 7. Gate

- [ ] Size derived from the **out-of-sample** pooled estimate, never in-sample
- [ ] Deflated for the number of independent selections, not the grid size
- [ ] Shrunk in proportion to its standard error
- [ ] At most half Kelly on what survives
- [ ] Drawdown cap applied and stated, with the tolerance justified
- [ ] Capacity checked against the volume the backtest assumed
- [ ] At portfolio level: risk contribution, stressed correlations, effective breadth
