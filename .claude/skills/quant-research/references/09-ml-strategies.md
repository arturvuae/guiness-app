# When the strategy is a model, not a rule

A gradient boosting model over 40 features is not a parameter grid, so phase 3
does not map onto it directly. Everything else does, and the failure modes get
worse rather than different: a model with thousands of effective degrees of
freedom overfits faster than a two-parameter crossover, and it does so in ways
that are harder to see.

Read this when the strategy has learned parameters rather than chosen ones.

## 1. What carries over unchanged

- The OOS contract. One out-of-sample period, consumed once.
- Costs, fill semantics, look-ahead detection.
- Effective sample size. Labels built from overlapping windows are the same
  bet counted many times (`references/07-effective-sample.md`).
- Deflation for the search. Every hyperparameter configuration tried is a
  trial, and hyperparameter search is a parameter sweep with a different name.
- Sizing from the tail.

## 2. Cross-validation has to be purged

Standard k-fold cross-validation is invalid on financial data, and the reason
is not "time series need ordering". It is that **labels overlap**.

If the label at time `t` is the return over the next 20 days, then the label at
`t` and the label at `t+5` share 15 days of the same future. Put one in train
and the other in test and the model has seen its test answer. Ordinary k-fold
does this thousands of times, and the resulting scores can be spectacular and
entirely fictional.

Two corrections, applied together:

**Purging.** Drop from the training set every observation whose label window
overlaps any test observation's label window.

**Embargo.** Drop an additional buffer of observations immediately after each
test block, because serial correlation leaks information forward past the
purge boundary. An embargo of about 1% of the sample is a common starting
point.

Walk-forward with purge and embargo is the safest arrangement: it respects
ordering *and* removes the overlap.

## 3. Where the leakage actually comes from

Ranked by how often each one silently ruins a study:

**Feature scaling over the whole sample.** Fitting the scaler, the PCA, or the
imputation on all data before splitting. The test set contributed to the
transform. Fit inside the training fold, apply to the test fold.

**Feature selection before splitting.** Choosing the top 20 features by
correlation with the target, on the full sample, then cross-validating those
20. The selection already used the test data, and the CV score is meaningless.

**Target built with future information.** Triple-barrier labels, forward
returns, "did it hit the target before the stop": all legitimate, all trivially
leaky if the barrier window extends past the split.

**Point-in-time failures.** Fundamentals restated later, index membership as it
reads today, macro series revised after the fact.

**Duplicated rows across the split.** Resampling, augmentation, or overlapping
windows placing near-identical rows on both sides.

## 4. Sizing the search

Hyperparameter tuning is a sweep. The same discipline applies:

- Count every configuration tried. The trial counter from phase 1 covers model
  architectures, feature sets, label definitions, and random seeds.
- Prefer a small hyperparameter grid with a clear rationale to a large random
  search. The large search buys a better in-sample number and a worse
  out-of-sample one.
- Seeds are trials. Running 20 seeds and reporting the best is selection on
  noise, in its purest form. Report the distribution across seeds, and use its
  mean as the estimate.

**The pooling idea transfers directly.** Rather than deploying the best
configuration, average the predictions of the configurations that are
statistically indistinguishable. This is what ensembling does, and it works for
the same reason regional pooling works: averaging correlated noisy estimators
of one signal converges, and taking their maximum does not.

## 5. Interpreting a model you cannot read

**Feature importance is not a mechanism.** It says which inputs the model
leaned on, not why they should pay. A study whose only justification is a
feature importance chart has not identified an edge.

**Permutation importance on the test set, not the training set.** Training-set
importance measures memorization.

**Check stability across folds.** Features that matter in one fold and vanish
in the next are fitting noise. A stable ranking is weak evidence of something
real; an unstable one is strong evidence of nothing.

**Simpler baseline first, always.** Run the two-parameter rule before the
40-feature model. If the model does not beat it out of sample by a margin that
clears its own standard error, the model is complexity without return.

## 6. Gate

- [ ] Cross-validation purged and embargoed, or walk-forward with both
- [ ] Every transform fitted inside the training fold only
- [ ] Feature selection performed inside the fold, never before splitting
- [ ] Label construction checked for windows crossing the split
- [ ] Seeds and configurations counted as trials
- [ ] Predictions ensembled across indistinguishable configurations
- [ ] Beaten a simple baseline by more than its standard error
