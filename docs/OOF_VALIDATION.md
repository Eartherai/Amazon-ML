# Phase4 entity OOF protocol

Baseline BASELINE-P4-001 is frozen at 9e08d01d7bba8035acf858f3240258861106f466. Its 0.91056 score is a repeatedly observed 505-entity development result, not expected leaderboard performance.

Natural-prevalence samples contain 5,000 / 20,000 / 50,000 / 100,000 S1 entities, chosen by deterministic SHA256 rank from original folds1–3. They are nested, not independently sampled. Diagnostic strata are separate and must never supply the headline score. Fold4 remains CLOSED. Sampling labels describe strata but do not balance the natural samples. The original fold1 training pilot may overlap these development samples; OOF excludes each scored entity from its fitted model, but does not erase prior architectural exposure. Fold4 remains the eventual reality check.

## Fitting boundary

The Phase3 character IDF fit used fold1–3-owned targets. Reusing it would undermine a training-only OOF claim. Phase4 changes this one required component: fit vocabulary/IDF on a deterministic approximately1/64 sample of fold0-owned and unowned training targets. No fold1–3 or fold4-owned target text fits the vocabulary. Full training target text is transformed and searched, as it will be at deployment. Target ownership is used solely to enforce fitting exclusions. No test text or labels fit this configuration. This fixed external fitting partition must be retained when reproducing these results; it is not per-fold IDF fitting.

The generic Mac Foundation transliteration map is a deterministic library transformation, not learned from labels or frequencies. Native forms remain available. No learned replacement dictionaries are active.

## Nested decisions

For each outer fold1/2/3, fit the model on the other two folds. Training candidate negatives whose owner is outside the fit folds are excluded, including fold0/4 owners. Unowned targets remain eligible. Every S1's pairs share its fold.

Select that outer fold's threshold using two inner predictions: train on one of its training folds and score the other, then reverse. Outer labels never select the threshold. Finally fit on both outer training folds, score the held-out fold and apply the inner-selected threshold. Also report the frozen0.58 threshold. Inner models have smaller training sets, so threshold transfer must be examined. Pooled outer threshold curves, if added later, are descriptive development curves and cannot retrospectively replace a held-out decision.

Metrics include exact macro F0.5, micro precision/recall (explicitly named), singleton/non-singleton, country and source-specific set metrics. Source metrics retain all S1s and score an empty source-specific truth correctly. Unretrieved true links remain false negatives. Bootstrap resamples S1 entities, never pair rows; its interval conditions on this training procedure and is not model-refit uncertainty.

## Execution status

Nested samples and validation code implemented; 5k full-pool retrieval in progress. No Phase4 OOF score available yet. Fold counts and measured results will be appended after successful execution. No leaderboard submission and no EC2/GPU launch.

## Completed measured run

See [Phase4 checkpoint](PHASE4_CHECKPOINT.md) for per-fold natural prevalence, macro/P/R, source/country scores and bootstrap interval.
