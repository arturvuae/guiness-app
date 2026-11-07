# Phase 3: map the surface, bound the region, pool

The core of the method. Everything else supports this.

An optimizer's `argmax` answers "which cell scored highest?". Nobody wants the
answer to that question. What people want is "which parameterization will pay
next year?", and the highest-scoring cell is a poor estimator of it: often no
better than its neighbours, and sometimes worse, because part of what made it
highest was luck that does not repeat.

## 1. Why the argmax fails

Each cell's in-sample metric is a **noisy estimate** of an unobservable true
performance. Write it as:

```
observed = true + noise
```

Selecting the maximum of `observed` selects jointly for high `true` **and**
high `noise`. Over thousands of cells, the noise term dominates the selection:
the winner is mostly the luckiest, not the best. This is why in-sample rank
predicts out-of-sample performance so poorly. Inside a plateau, the measured
correlation between the two is routinely around **0.05 to 0.10**, i.e. nothing.

Read that number carefully, because it settles the question. If in-sample rank
carried real information about out-of-sample results, the correlation would be
positive and material, and picking the top cell would be rational. It is not,
so it is not.

## 2. Designing the grid

**Axes carry the hypothesis.** Sweep parameters the mechanism actually depends
on. Sweeping seven axes because they exist multiplies the trial count (which
phase 4 charges for) without adding information.

**Bounds must contain the edge.** The plateau's *boundary* is the informative
part, because it tells you the range over which the edge exists. A grid entirely
inside the plateau looks wonderful and teaches nothing; a grid entirely outside
looks dead. Widen until both edges are visible.

**Resolution.** Fine enough that neighbouring cells are genuinely neighbours
(so smoothing means something), coarse enough to keep the grid affordable. If
adjacent cells give wildly different results, the step is too large or the
strategy is unstable. Both are findings.

**Sweep choices, not only numbers.** Which timeframe, which exogenous series,
which indicator variant: these are axes too, and treating them as axes rather
than as separate studies keeps the trial count honest.

**Size.** Thousands of combinations, not dozens. A 20-point sweep cannot show a
plateau's shape. This is also where engine speed stops being a convenience:
250,000 backtests is a routine grid for this method, and an engine that needs
minutes per thousand combinations makes the method impossible rather than slow.

## 3. Smoothing

Replace each cell with the mean of its local neighbourhood (3×3 in two
dimensions, the analogous cube in more). Two reasons:

1. It attenuates per-cell noise, so the smoothed argmax lands nearer the true
   optimum than the raw argmax does.
2. It rewards *robustness*. A cell surrounded by good cells is worth more than
   an equally good isolated cell, because parameter estimates drift and a
   strategy will not sit exactly on its cell forever.

A peak that disappears under smoothing was never a peak. That single test
removes most fitting artifacts.

## 4. The error bar

For a Sharpe ratio `S` estimated from `T` observations, where `S` and `T` are
expressed at the **same frequency**:

```
SE(S) ≈ sqrt((1 + S²/2) / T)
```

Sharpe ratios are normally quoted annualized, so use `T` = **number of years**.
The unit matters more than the formula: using `T` = 1,000 daily bars for an
annualized Sharpe understates the error by a factor of ~16 and makes every
difference look significant.

| Annualized Sharpe | Sample | SE | Interpretation |
|---|---|---|---|
| 1.0 | 4 years | 0.61 | t ≈ 1.6, not significant |
| 1.0 | 10 years | 0.39 | t ≈ 2.6, significant |
| 0.5 | 4 years | 0.53 | t ≈ 0.9, indistinguishable from zero |
| 0.5 | 20 years | 0.24 | t ≈ 2.1, marginal |

Two consequences, both uncomfortable:

**Differences of a few tenths are not differences.** Two cells at 1.31 and 1.14
are the same cell measured twice. Ranking them and deploying the winner is a
ritual, not an inference.

**Most published backtests are not significant.** A Sharpe of 1.0 over four
years has `t ≈ 1.6`, below the conventional 1.96 threshold, and that is before
any correction for the number of trials. Phase 4 makes this worse.

Every Sharpe quoted anywhere in the report carries its standard error. A bare
Sharpe is uninterpretable and invites exactly the wrong conclusion.

## 5. Bounding the region

The selection rule:

> Keep every combination whose **smoothed** metric is within **one standard
> error** of the **smoothed best**.

That set is the output of the optimization. It typically contains hundreds of
members, which is not a problem to be reduced. It is the finding: one edge,
many parameterizations, and no way to tell them apart from in-sample data.

Then interrogate the region:

- **Size.** How much of the grid does it cover? A large connected region means
  a robust edge. A region of three scattered cells means an artifact.
- **Connectivity.** Contiguous, or scattered islands? Scattered is a warning.
  Real edges occupy neighbourhoods.
- **Position.** Against a grid boundary? Then the grid was too narrow and the
  true region extends past it. Widen and re-run.
- **Internal correlation IS vs OOS.** Compute it. Near zero confirms that no
  further selection inside the region is justified. If it is materially
  positive, that is interesting and unusual, so investigate before exploiting it,
  since it more often indicates a leak than a discovery.

## 6. Pooling

Treat the region as an **equal-weight portfolio** of its members. Each member
is a correlated, noisy estimator of the same underlying signal; averaging them
cancels part of the noise while preserving the signal.

Effects to expect:

- Pooled Sharpe usually lands **above** the median member and **below** the
  in-sample champion, and, this being the point of the exercise, above what
  that champion delivers out of sample.
- The pooled equity curve is smoother, because idiosyncratic parameter noise
  partially cancels.
- The result is reproducible by someone else, because it does not depend on
  which cell happened to win.

Equal weight, not metric-weighted. Weighting by in-sample performance
reintroduces the selection the pooling exists to remove.

Practically, pooling means holding the average position of the region's
members: a position sized at the mean of what each parameterization would hold.
Verify that the pooled position is actually tradable: its turnover, its
maximum size, and its capacity, since averaging can produce a position that
moves more often in smaller increments.

`scripts/region_pool.py` implements smoothing, SE, region selection, and
pooling on a sweep DataFrame.

## 7. Common failure modes

| Symptom | Reading |
|---|---|
| Sharp isolated spike | Fitting artifact. Not tradable. |
| Region touching grid edge | Grid too narrow. Widen and re-run. |
| Region of 1 to 3 cells | No plateau. There is no edge to bound. |
| Whole grid inside region | Grid too narrow the other way, or the parameter does not matter. Both are findings worth stating. |
| Two separate plateaus | Possibly two different regimes or two different mechanisms. Investigate before pooling across them, since pooling unrelated edges is not the same operation. |
| Plateau vanishes under smoothing | Noise. Stop here. |

## 8. Gate

- [ ] Grid wide enough that the plateau's edges are visible
- [ ] Metric smoothed over neighbourhoods
- [ ] Standard error computed, convention stated
- [ ] Region bounded at 1 SE of the smoothed best
- [ ] Region size, connectivity and position inspected
- [ ] IS/OOS correlation inside the region measured and reported
- [ ] Pooled equity curve built, equal weight, tradability checked
