# After the verdict: monitoring and killing

Research does not end when a strategy is deployed. It changes question, from
*does this edge exist* to *does it still*.

This phase is where most of the discipline built in phases 1 to 5 gets quietly
abandoned, because live results arrive one day at a time and each day invites a
decision that the framework was designed to prevent.

## 1. Pre-register the kill criteria

Write them **before** the first live trade, alongside the phase 1 falsifier,
and store them where they cannot be edited comfortably.

A usable set:

- **Drawdown**: exceeding the p95 of the simulated distribution
  (`references/08-drawdown-risk.md`). Not the historical drawdown, the
  simulated tail.
- **Duration**: underwater longer than the p95 recovery time from the same
  simulation.
- **Deviation**: live Sharpe below the out-of-sample estimate by more than two
  standard errors, over a window long enough to mean something.
- **Mechanism**: the structural reason the edge existed has demonstrably
  changed. A venue changed its fee schedule, a counterparty left, a rule
  changed. This one triggers immediately and does not wait for the statistics.

The first three are statistical and slow. The fourth is the one that saves
money, and it requires knowing why the strategy worked, which is why phase 1
insisted on a mechanism.

## 2. Live results are a tiny sample

The trap: six months of live data feels more real than ten years of backtest,
so it gets over-weighted in both directions.

Run the numbers. Six months at a true Sharpe of 0.8 has a standard error of
about `sqrt((1 + 0.32)/0.5) ≈ 1.6`. A live Sharpe anywhere between -2.4 and 4.0
is consistent with the strategy working exactly as designed. Concluding
anything from six months is not possible, and the framework has to say so in
both directions:

- A good first six months is **not** confirmation, and not a reason to size up.
- A bad first six months is **not** falsification, and not a reason to
  intervene, unless a pre-registered criterion fired.

This is why the criteria are pre-registered. In the moment, every deviation
finds an explanation.

## 3. What to actually track

**Live against simulated, not against the backtest.** The backtest is a point.
The Monte Carlo distribution from phase 4 is the reference: the honest question
is which percentile of it the live path sits in. Sitting at the 30th percentile
is unremarkable. Sitting below the 5th is a signal.

**Costs realized against costs assumed.** The most common source of live
underperformance is not the edge disappearing, it is slippage and fees running
above what the backtest assumed. Track them separately from the edge, because
the fix is different: one is a strategy problem, the other an execution one.

**Fill quality.** Actual fill prices against the prices the signal saw. A
widening gap means capacity has been reached, or the market has learned.

**Exposure and turnover against expectation.** A strategy trading twice as much
as it did in the backtest is not the same strategy.

**Decay in the effective count.** Track live observations in effective terms,
not in trades, for the reasons in `references/07-effective-sample.md`.

## 4. What not to do

| Never | Because |
|---|---|
| Re-optimize on live data | The live period is the only genuinely out-of-sample data left. Spending it on fitting destroys the only honest evidence available. |
| Size up after a good run | The edge did not change, the sample did. |
| Turn it off during a drawdown inside expectations | The drawdown was simulated in advance; stopping there realizes the loss and forfeits the recovery. |
| Add a filter to avoid the recent bad period | This is in-sample fitting performed on live data, with real money already committed. |
| Compare live to the in-sample backtest | The comparison is against the out-of-sample estimate, which is lower. |

## 5. When to retire

Retire on a pre-registered criterion, or on a mechanism change. Not on a
feeling, and not on a peer's opinion.

When retiring, **write the post-mortem**: what the edge was, what changed, how
long it took to detect, and whether any monitoring signal fired early enough to
matter. The value of that document is that it makes the next study cheaper,
and it is the only part of this work that compounds.

## 6. Gate

- [ ] Kill criteria written and stored before the first live trade
- [ ] Live path tracked against the simulated distribution, not the backtest
- [ ] Realized costs tracked separately from the edge
- [ ] No re-optimization on live data
- [ ] Post-mortem written at retirement
