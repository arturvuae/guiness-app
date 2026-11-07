# Phase 5: portfolio, report, verdict

## 1. Portfolio level

A strategy is rarely traded alone, and its standalone metrics can mislead in
both directions.

**Correlation.** Compute the correlation of the candidate's returns against
what is already traded. A Sharpe 0.6 strategy uncorrelated with the book can be
worth more than a Sharpe 1.2 strategy that duplicates it.

**Effective breadth.** `N` strategies with average pairwise correlation `ρ`
behave like roughly `N / (1 + (N-1)ρ)` independent bets. Ten strategies at
`ρ = 0.6` are about 1.5 independent bets, not ten. Counting strategies instead
of independent bets is how books end up with one position under many names.

**Joint drawdown.** Correlations rise in stress. Estimate the drawdown of the
combined book, not the sum of individual drawdowns, which understates it.

**Capacity.** At what size does the edge disappear into its own impact? Answer
this before allocating, not after.

## 2. Report structure

Use this template. It is short on purpose: a report nobody reads protects
nobody.

```markdown
# [Strategy name]: research report

## Hypothesis
Mechanism, horizon, falsifier. Written before the work started.

## Data
Source, symbols, period, resolution, checks passed, known gaps.
IS/OOS split with exact dates.

## Simulation
Cost assumptions with justification. Fill rule. Look-ahead check result.
Trade count and exposure.

## Surface
Grid definition and total trial count.
Heatmap or surface plot.
Plateau description: size, connectivity, position.
Region: selection rule, number of members, IS/OOS correlation inside it.

## Pooled result
Pooled equity curve, in-sample and out-of-sample.
Sharpe ± SE, deflated Sharpe with N, t-statistic.
Comparison against the in-sample champion's OOS result.

## Robustness
Walk-forward table by fold, WFE, region drift.
Monte Carlo intervals with path count.
Break-even cost level.
Sub-period and cross-asset results.

## Verdict
The decision rule as fixed in phase 1.
The result against it.
Trade / reject / retest, and the sizing basis if trading.
```

## 3. Reporting rules

**Every Sharpe carries an error bar and a trial count.** No exceptions. A bare
Sharpe is uninterpretable, and stating it invites the reader to over-conclude.

**Report the region, not a champion.** If a single line is required for a
dashboard, it is the region's pooled result, labelled as such.

**Publish the negative results.** The sub-period where it failed, the asset
where it did not appear, the cost level that kills it. A report that shows only
what worked has selected its own contents, which is the same error the whole
workflow exists to avoid.

**Show the surface.** One heatmap communicates more about robustness than any
table of metrics. If there is one figure, make it that one.

**Separate measurement from interpretation.** "Pooled OOS Sharpe 0.64 ± 0.32,
t ≈ 2.0" is a measurement. "This is a good strategy" is an interpretation, and
it belongs in a clearly marked sentence.

## 4. The verdict

Apply the rule from phase 1, unmodified. If the rule now feels wrong, that is
information about the rule, and the honest move is to say so explicitly rather
than to quietly move the threshold.

**Trade.** The pooled OOS result clears the criterion, walk-forward transfers,
break-even cost sits comfortably above realistic costs, the mechanism still
makes sense. Size on the **out-of-sample** estimate, discounted further for the
trial count. Never size on in-sample.

**Reject.** The criterion is not met. Write it up anyway. A documented negative
result stops the same idea from being re-tested next quarter, which is worth
real money.

**Retest.** Genuinely inconclusive, and there is *new* data or a *new* market
to test on. Not "run it again on the same data with a tweak": that is fitting
under another name, and it consumes what is left of the sample.

## 5. Knowing when to stop

Signals that the study is over:

- The plateau does not survive smoothing.
- The deflated Sharpe is at or below zero.
- The edge dies at realistic costs.
- The region is a handful of scattered cells.
- The mechanism was never articulated, and the search has not suggested one.

At that point say **"no significant edge"** plainly. It is a finding. It is the
most common correct finding in quantitative research, and a workflow incapable
of producing it is not measuring anything.

The strongest tell that a study has gone wrong is a researcher who has already
decided the answer and is iterating until the data agrees. When the user shows
that pattern, repeated tweaks after each disappointing result, name it once,
plainly, and point at what the iterations have cost: the out-of-sample period,
and the trial count that now has to be deflated against.
