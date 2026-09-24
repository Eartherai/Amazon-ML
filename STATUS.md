# Latest Phase5 status — 2026-09-24T22:52:42.142603+00:00

Full-training retrieval is RUNNING on EC2: P5-FULL-RETRIEVAL-001, instancei-0609d158c38e96160, frozen code5ac0845. All2,206,821S1queries,10,320,219targets,64SHA256shards per country/route. Projected17.76hours plus overhead,24hshutdown,$18planningallowance; actualcostunknown. Automatic S3checkpoint verification and termination are installed. Do not launch a duplicate. Collect with scripts/aws/collect_run.py.

Best remains20k NUMERIC-V2 macro0.9318965293, precision0.98158282, recall0.85593807, singleton0.92131747. Country-balanced EXP-026did not improve mixed OOF:0.93171205; India improves,US declines. Full-data retrieval/model scores pending. Fold4 CLOSED; testinference/validator/submission/package NOTREADY. Active AWSprofile amamzon_01_a1_0, same account ending6318 asdefault. CPU64/G8/SageMakerG5quota requests awaiting AWSapproval. Creditsunverified. Next: collect checkpoints/full metrics, build larger OOF feature store, test targeted lexical rescue, reassess countryrobustness, then GPUrescue if quota/evidence permit.

Prior status entries follow.

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

## Current best —20k numeric confirmation

51-feature LightGBM with preserved raw/original numeric features plus6 Unicode-digit/leading-zero comparisons: macro0.931897, precision0.981583, recall0.855938, singleton0.921317. Nested3outer/2inner folds,20k natural entities. New15k-only delta+0.003299,95% CI[+0.001840,+0.004701]. See docs/NUMERIC_CONFIRMATION.md. Current best artifact outputs/experiments/P4-NUMERIC-B-001; original reference unchanged. All launched local jobs completed. Next priorities: targeted India retrieval rescue and nested singleton/meta-model, then larger validation. Fold4 CLOSED; no submission; country robustness unresolved.62 tests pass.

## Phase5 started — current action
Best remains20k numeric-v2 macro0.9318965293; no new full-data score. Active AWS profile amamzon_01_a1_0 verified same account ending6318. CPU8vCPU; GPU/SageMaker training0. Increases pending. Persistent full-target index benchmark EXP-024 prepared with90min cap; not yet measured. Fold4 CLOSED. Full OOF, test inference, validator and submission remain pending.

### Phase5 benchmark completed
All full-target persistent indexes built,184S3objects verified; benchmark instance terminated. Current best remains20k0.9318965293. Full2.2M retrieval is prepared next (17.76h projection,24hcap,$18planning ceiling). Active profile amamzon_01_a1_0. CPU64/G8/SageMakerG5quota requests CASE_OPENED, not approved. Fold4 remains closed. See docs/PHASE5_CHECKPOINT.md.
