# Phase 1: framing, data, and the trial budget

Everything expensive that goes wrong later was decided here, cheaply.

## 1. The hypothesis

Write three lines before writing any code.

**Mechanism.** What structural feature of the market pays this? Candidates that
qualify: a risk premium someone is paid to bear, a flow that is forced
(rebalancing, expiry, liquidation), a constraint on a class of participants, a
slow diffusion of information. Candidates that do not qualify: "the 14-period
RSI works on this pair", "the backtest looks good".

Ask who is on the other side and why they keep taking it. If there is no
plausible answer, the study is a search over noise. That is allowed, but say
so, and expect the workflow to confirm it.

**Horizon.** Holding period and trade frequency, stated up front. It determines
the data resolution, the cost model, and how many independent observations the
sample actually contains. A daily strategy over four years has ~1,000
observations; a monthly one has 48, which is not enough to conclude anything.

**Falsifier.** What outcome drops this idea? Written now, while it costs
nothing to be honest. "Pooled OOS Sharpe below 0.3" is a falsifier. "It doesn't
look good" is not.

## 2. Data hygiene

Bad data produces confident wrong answers, and the confidence is the problem.

| Check | What goes wrong if skipped |
|---|---|
| Provenance | Vendor-adjusted series silently restate history |
| Resolution | Daily bars hide intraday stops that would have triggered |
| Gaps | Missing bars become fake continuity; a gap over a crash flatters everything |
| Timezone / session | Bar boundaries misaligned by hours, signals fire on the wrong bar |
| Corporate actions | Unadjusted splits create fake ±50% returns |
| Survivorship | Only-survivors universe is the single largest source of fake edge |
| Multi-asset alignment | Different calendars, forward-filled to a common grid, invent co-movement |
| Point-in-time | Fundamentals or macro that were revised later are look-ahead by construction |

Produce a short data sheet: source, symbols, date bounds, bar interval, number
of bars, gaps found, checks passed. Put it in the report. A study whose data
provenance is not written down cannot be reproduced, including by its author in
six months.

**Adversarial check.** Plot the raw series before running anything. Human eyes
catch flat segments, spikes and stitched vendors faster than any assertion.

## 3. The out-of-sample contract

Split the period *now*, before looking at anything.

- **In-sample**: everything the research touches. Sweeps, plateau reading,
  region selection, cost modelling, debugging.
- **Out-of-sample**: touched exactly once, at the verdict.

Rules of thumb: enough OOS to contain several market regimes, and enough
in-sample to support the grid size intended. A 250,000-combination sweep over
two years of daily data is fitting 250,000 parameters to 500 observations.

**Consumption.** Once OOS is read, it is in-sample forever. A study that reads
OOS, dislikes it, adjusts, and re-reads has no out-of-sample left. It has a
second in-sample and a false sense of validation. If this happens, say so in
the report; the honest move is to declare the study in-sample-only and hold out
new data (a later period, another asset) before claiming anything.

Ways the contract breaks quietly:

- Choosing the universe after seeing which symbols did well
- Fixing a "bug" that only became visible because OOS was bad
- Reading OOS "just to check" before the region is bounded
- Sharing OOS across several strategies in the same research programme

## 4. The trial budget

Start a counter and keep it.

```
trials = combinations × assets × variants × restarts
```

Everything counts, including runs that were discarded, grids that were widened,
and indicators that were swapped out. It is not the number of results kept, it
is the number of results *looked at*. Phase 4 divides by this; it cannot be
reconstructed after the fact, and reconstructing it optimistically is the most
common way a deflated Sharpe comes out wrong.

Order of magnitude matters more than exactness. Under 100 trials, selection
bias is modest. At 10,000, the expected maximum Sharpe of pure noise is around
1.0 for typical sample lengths, so a strategy scoring 1.2 in-sample after a
10,000-cell sweep has demonstrated almost nothing.

## 5. Gate

Do not proceed until all four hold:

- [ ] Mechanism, horizon and falsifier written down
- [ ] Data sheet produced, checks passed, series plotted
- [ ] IS/OOS split fixed and recorded
- [ ] Trial counter started
