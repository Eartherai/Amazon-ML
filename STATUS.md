# Phase4 checkpoint — first nested OOF measured

Current action:20k natural-prevalence full-pool retrieval running locally (P4-B-002), plus eight feature ablations on the completed5k OOF. Fold4 CLOSED. No submission or paid AWS compute.

## Validation result

- OOF entities:5,000; outer folds:3; two inner folds select each outer threshold.
- OOF macro F0.5:0.923407; fold std:0.002393; fold min/max:0.920961/0.925744.
- Entity-bootstrap95% CI:0.918577–0.927935 (conditional on fitted procedure).
- Micro precision:0.980893; micro recall:0.834650.
- Singleton F0.5:0.879699; non-singleton:0.925862.
- India:0.902014; US:0.938733; source-specific S2:0.894321; S3:0.890603.
- Selected reference remains45-feature multiview LightGBM. Nested raw thresholds:0.83/0.82/0.81. Baseline-P4-001 at0.58 remains immutable for comparison.
- Isotonic0.923497 versus raw0.923407 is insufficient evidence to add complexity. Platt0.922785. Pair+empty0.923113 raises singleton score to0.906015 but slightly lowers macro. Retain raw nested procedure.

## Candidate configuration

P4-A-001: name_char3+address_char3, top100 each, same-country full10,320,219 training targets. Fixed vocabulary/IDF fit only fold0-owned/unowned text; OOF1–3 text excluded from fitting.

- Link recall:0.966503; complete-positive-entity recall:0.903464; oracle:0.989149.
- Candidates:985,945; average:197.189; p95/p99:200/200.
- Most expensive route: address_char3,476.99s versus261.01s for name.
- Best targeted rescue: not yet validated at scale. Broad Phase3 union remains only a pilot diagnostic; do not attach its97.62% recall to this model.
- P4-B-001 was stopped before completing a route to avoid repeated target-matrix transposes; preserved. P4-B-002 moves transpose outside query blocks and uses fused sparse topK with deterministic near-tie fallback. Nested5k parity is required before20k OOF.

## Errors

282 outer OOF false merges;162 raw scores>=0.90. Leading overlapping observed patterns: different-known-owner113, name-token containment91, near-identical name51, missing address26, exact name/different address23. No chain/franchise cause inferred from these flags.

581 unretrieved positives: weak address473, weak name272, non-ASCII target name257, both weak228, missing address137. India393/US188. Causes like cap truncation remain unproven without deeper rankings.

## Compute and reproducibility

5k retrieval738.51s/1.93GiB peak; nested OOF81.58s/2.79GiB peak.57 tests pass. AWS compute$0; S3 storage accruing at estimated$0.092684/month plus requests, actual bill/remaining credits unverified. No new cloud resource launched. Local free-space reserve remains8GiB.

Frozen reference: configs/baselines/BASELINE-P4-001.yaml. Results: docs/PHASE4_CHECKPOINT.md. Candidate/feature/model artifacts are immutable under outputs/candidates/P4-A-001 and outputs/oof/P4-A-001. Experiments006–008 recorded. These are development OOF results, not final leaderboard forecasts.

## Next10 experiments

1. Finish feature ablations and compare paired entity-bootstrap deltas.
2. Finish20k retrieval and verify nested5k candidate parity.
3. Run20k nested OOF with the retained reference model.
4. Compare score, country gap and singleton stability across sample sizes.
5. Test transliteration/name and address rescue on observable query triggers.
6. Measure marginal complete-entity coverage and candidate cost.
7. Test nested S1-level singleton/meta-model if error evidence supports it.
8. Test source-specific thresholds and category-aware negatives with nested selection.
9. Compare GBDT families on fixed candidates/folds; dense only after lexical marginal benchmark.
10. Freeze final procedure before one-time fold4 evaluation, then deterministic test generation/validator and submission gate.

## Important new risk — country transfer

Completed label-transfer stress test: India→US macro0.886261; US→India0.767327, singleton0.516129. This materially weakens any claim of France robustness. Fixed IDF contains both known countries' external-fold text, so this is not strict held-country-text evaluation. See docs/COUNTRY_TRANSFER.md. Next add sample-size-matched controls and require cross-country evidence before promoting more complex models. No threshold chosen on evaluation-country labels.

All eight feature ablations finished; see docs/PHASE4_ABLATIONS.md. Numeric, transliteration, retrieval and token/character evidence matter; route-count-only effect is inconclusive. A full-country persistent name index matched reference candidates but did not show a speed win; retain chunked20k retrieval. Exact-name address rescue recovered5 development links with830 additional candidates; not yet part of the trained matcher.

## Larger20k confirmation

Baseline OOF macro0.928759, CI[0.9264584347104737, 0.9310251784781759]; singleton0.920403; precision0.980463; recall0.848629. Full-pool candidate recall0.966626, oracle0.988531. Both routes exactly match all nested5k reference pairs/scores. Numeric6-feature20k model is running; new15k subset comparison follows. One-config CatBoost/XGBoost probes did not clearly beat LightGBM.62 tests pass.
