# Phase4 checkpoint — validation preparation

Current action: 5,000 natural-prevalence S1 full-pool name/address char3 retrieval running locally. Nested entity OOF implementation and safeguards ready; execution follows retrieval. Fold4 CLOSED. No leaderboard submission.

## Measured evidence

- Frozen reference: configs/baselines/BASELINE-P4-001.yaml, Git9e08d01d7bba8035acf858f3240258861106f466; GBDT-004 multiview,45 features, threshold0.58.
- Old development check: macro0.910560, precision0.953547, recall0.862578, singleton0.80;505 repeatedly inspected S1. Not OOF and not expected leaderboard performance.
- Old two-char3-route candidate recall96.3468%,197.318 candidates/S1. The96.7817% number additionally includes token_union.
- Broad pilot union:97.6225% recall,93.0526% complete-positive-entity recall,0.991159 oracle,952.446 candidates/S1. Not a model score.
- Nested natural samples built:5k,20k,50k,100k; singleton rates5.32%,5.465%,5.598%,5.601%. Separate diagnostic sample available. Samples do not open fold4.
-54 tests pass. Paid AWS compute $0; S3 storage accruing, estimated $0.092684/month plus request charges; actual billed total/credit balance unverified.

## OOF metrics

OOF entities/folds scored:0/0. OOF macro, standard deviation,95% CI, precision, recall, singleton, non-singleton, India, US, S2 and S3: pending successful run. Do not substitute pilot results.

## Validation decisions

Original IDF used folds1–3 and cannot be reused as training-only OOF there. Phase4 fits IDF only on fixed fold0-owned/unowned training targets. Full training target pool remains searchable. All held-out-owned targets are excluded from model-training negatives. Threshold selection uses two inner folds wholly inside each outer training partition. Three outer folds1–3; no pair split. See docs/OOF_VALIDATION.md.

## Error evidence

Broad pilot misses82 links: weak address76, weak name46, both weak43, non-ASCII target name41, missing address12, numeric disagreement8. Overlapping descriptive flags, not proven causes. Expanded tokens add493,006 candidates for7 final marginal links. Large-scale false merges and best rescue/singleton strategy remain pending OOF.

## Next10 experiments

1. Complete5k two-route retrieval and measure oracle/coverage.
2. Run3-fold nested OOF with frozen multiview GBDT.
3. Audit every outer OOF false merge and unretrieved positive.
4. Compare global threshold with joint pair/empty threshold.
5. Compare nested Platt/isotonic calibration and reliability.
6. Measure feature-group ablations on identical folds/candidates.
7. Run20k full-pool retrieval with a recorded local runtime estimate.
8. Expand OOF and quantify paired entity bootstrap uncertainty.
9. Test observable-query rescue triggers against universal expansion.
10. Only then compare model families/category-aware negatives and gate dense benchmarks.

Current blockers: none for5k; larger retrieval/index scaling requires measured cost/runtime. No final calibrated model, submission artifact or fold4 score exists.
