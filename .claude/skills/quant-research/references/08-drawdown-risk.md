# Drawdown is a draw, not a property

The maximum drawdown in a backtest gets treated as the strategy's worst case,
and used as the risk budget. It is neither. It is one sample from a
distribution, and usually a mild one.

## 1. Why the realized figure is nearly useless

**Volatility has memory. Drawdown does not.** Daily variance carries forward
almost entirely into the next day, which is the entire justification for
volatility models. Maximum drawdown carries essentially nothing: measured
across successive windows it wanders, and the extreme observed in one period
does not bound the next.

A maximum drawdown is an extremum of a single path. It depends on where the bad
days happened to cluster, which is exactly the part of history least likely to
repeat.

The practical consequence, measured on a leveraged multi-asset book:

| Quantity | Value |
|---|---|
| Historical max drawdown | -33.7% |
| Median expected, ordinary year | ~11% |
| p95, bad year | ~25% |
| Prediction error, simulation-based | 6 points |
| Prediction error, using the past extreme | 18 points |

The historical figure was three times the median expectation and still a poor
predictor. Using it as a risk budget is doubly wrong: too pessimistic about the
ordinary case, and unanchored to the tail that matters.

## 2. What to use instead

**Simulate the distribution, then read a percentile.**

1. Fit a volatility model that captures clustering on a rolling window.
2. Simulate many forward paths from it.
3. Extract the maximum drawdown of each path.
4. Read the distribution, not a single number.

Report a median and a p95, and size against the p95. The median year is not the
one that ends the strategy.

`scripts/sizing.py` does the simulation version of this directly:
`drawdown_distribution` returns the maximum drawdowns across many simulated
lives at a given leverage, `drawdown_quantile` reads a percentile, and
`leverage_for_drawdown` inverts it to answer the question actually being asked:
how large can the position be before the bad case exceeds what the capital can
survive.

## 3. Sizing on the tail

This is where the correction pays. Setting a volatility target by looking at a
historical drawdown produces a number with no relationship to future risk.
Setting it from the simulated p95 produces one that does.

The direction of the correction is usually the same: **down.** On the book
above, holding a 20% drawdown tolerance requires cutting the volatility target
from 15% to roughly 9%. A third of the risk budget disappears once the tail is
estimated rather than remembered.

Worked through the sizing tool, a pooled out-of-sample Sharpe of 0.64 with a
standard error of 0.32 at 15% volatility, with a 35% drawdown tolerance:

```
  caps, smallest wins
    0.5 Kelly                1.51x
    35% drawdown at p95      0.63x     <- binds
    hard cap                 3.00x

  drawdown over 10 years at that size
    median life              20%
    bad life (p95)           35%
```

The drawdown cap binds well before Kelly does. That is the normal case, not an
edge case.

## 4. Reporting rules

**Never present a realized max drawdown as a worst case.** Present it as one
draw, next to the simulated distribution it came from.

**Never compare two strategies on realized max drawdown.** The comparison is
mostly comparing which one had a luckier path. Compare the distributions.

**State the horizon.** Maximum drawdown grows with observation time: a ten-year
figure and a two-year figure are not the same quantity, and a strategy will
eventually exceed any drawdown it has shown so far, simply by running longer.

**Pair it with recovery time.** A 30% drawdown recovered in four months and one
recovered in four years are different risks that the single number hides.

## 5. Gate

- [ ] Drawdown reported as a distribution, with median and p95
- [ ] The realized figure labelled as one draw, never as a ceiling
- [ ] Position size derived from the simulated tail, not from history
- [ ] Horizon stated alongside any drawdown figure
- [ ] Recovery time reported next to depth
