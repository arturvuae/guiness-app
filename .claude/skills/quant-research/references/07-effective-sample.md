# How many observations does the backtest actually contain?

Phase 2 asks whether the measurement is honest. This reference answers the
sub-question that decides it: **how much information is in the sample.**

Not how many trades. Trades are activity. Information is something else, and
conflating the two is the most reliable way to be confident about nothing.

## 1. The trade counter lies

A backtest reporting 2,000 trades may carry a few hundred independent
observations. Trades that overlap in time, or that sit in correlated
instruments, are largely the same bet recorded many times.

This would be a curiosity if the trade count did not sit in a denominator:

```
t = mean(r) / (stdev(r) / sqrt(N))
```

Inflate `N` and everything becomes significant. Worse, `N` is under the
strategy's control: trade more often and the count rises while the information
does not.

Measured on a strategy constructed to have **exactly zero alpha**, 16
correlated instruments, ten years, book-wide direction held 20 days at a time
(`scripts/effective_n.py --selftest`, 200 replications):

| Statistic | False positive rate at a nominal 5% |
|---|---|
| t on trades, naive | **47%** |
| t on trades, corrected to the effective count | 4% |
| t on daily returns | 4% |

2,016 trades, 237 effective observations, a 2.9x inflation of the t-statistic.
Nearly half of these zero-alpha strategies pass a 5% significance test.

## 2. Effective sample size

Two estimators, both in `scripts/effective_n.py`.

**Kish, from a correlation matrix.** When the pairwise correlation structure of
the observations is available:

```
N_eff = N² / sum_ij(rho_ij)
```

All correlations zero returns `N`. Everything perfectly correlated returns 1,
which is right: a thousand copies of one bet are one bet.

For trades, build the correlation from how much they overlap in holding time,
discounted by how correlated their instruments are. Two positions held over the
same days in correlated instruments are exposed to the same shocks.

**Autocorrelation, from a single series.** When only the return stream exists:

```
N_eff = N / (1 + 2·sum_k (1 - k/N)·rho_k)
```

Positive autocorrelation, which overlapping positions produce mechanically,
drives this below `N`.

## 3. What to do instead

Correcting the trade t-statistic fixes its arithmetic and leaves its deeper
problem: the number of trades and their timing are **outcomes of the strategy
being tested**, so they are not a neutral denominator.

**Measure on time periods.** Daily returns, or whatever the natural period is.
Trading more cannot manufacture more days, so the denominator cannot be gamed.
This is why the table above shows the daily statistic holding its nominal rate
while the trade statistic collapses.

**Regress against the benchmark.** An alpha from a regression of strategy
returns on market returns counts periods, isolates the part of the return not
explained by exposure, and holds its false positive rate regardless of trade
frequency.

**Report the effective count next to the raw one.** "2,016 trades, 237
effective observations" is honest. "2,016 trades" alone invites the reader to
compute the wrong standard error.

## 4. Consequences for the rest of the workflow

**The sanity floor is not a trade count.** A floor of "30 trades" is the same
counting error in miniature. The floor is on *effective observations*, and 30
highly overlapping trades in correlated names can be worth three.

**Sharpe standard errors use `T` in the right unit.** Phase 3 computes
`SE = sqrt((1 + S²/2)/T)` with `T` in years for an annualized Sharpe. If the
returns are heavily autocorrelated, even that `T` is optimistic, and the
effective year count is lower.

**Long holding periods are the dangerous case.** The inflation grows with the
overlap, so slow strategies with many concurrent positions are the ones whose
significance is most overstated, which is the opposite of the usual intuition
that high-frequency strategies are the ones with a sample-size problem.

## 5. Gate

- [ ] Effective observation count computed, not just the trade count
- [ ] Significance measured on periods or on a benchmark regression, never on
      the raw trade t-statistic
- [ ] Both counts reported
- [ ] Sanity floor applied to effective observations
