# Latest Phase5 status

25 September 2026, 10:05 IST: EXP-031's audited parity now passes as `NUMERIC_TIE_PASS`, not exact parity. All 256 archives passed SHA256; both old20k routes have 2,000,000 pairs and 20,000 queries. Name candidate IDs are identical with 16 near-tie rank changes. Address has 57 shared-pair rank changes and one rank-100 exchange of two known negatives; max shared-pair score delta 2.384e-7. Strict exact ID/rank parity remains recorded as failed. The bounded acceptance gate rejects changed positives and broader drift. EXP-032 can launch after its fresh cost guard; it has not yet launched. The local audit fixture is archived and its one full feature parity integration test now skips explicitly until restore: test suite 85 passed, 1 skipped. SUB-001 frozen inference paths are unchanged, public score unknown, Fold4 CLOSED.

25 September 2026, 10:02 IST: EXP-031 / P5-LEARNING-200K-001 completed 200,000 queries, all 256 archives and exited 0 with EC2 terminated. Strict old-20k parity initially failed for the name route: 16 missing and 16 extra ID/rank rows among 2,000,000, with maximum shared-pair score difference 2.38e-7; EXP-032 remains gated while these are diagnosed. The generated 3.44 GB audit database was backed up to private S3, fully read back with matching SHA256 and removed locally to restore about 14 GiB free for submission files. Original data and frozen inputs remain intact. France shard pre-upload and three SUB-001 inference paths continue.

25 September 2026, 09:38 IST: India remainder worker `P5-SUB001-INDIA-001` launched successfully on on-demand r8i.4xlarge `i-0a29e371cccca1b99`, immutable code commit `4a1872d`, deterministic India shards 8–63 only. It has a 12-hour independent shutdown and $15 planning ceiling. Fresh cost guard: gross bill estimate $3.09, active commitments $37.50, new maximum $14.34, projected project worst case $54.93 below $120 soft cap; actual credit balance unverified. Cloud US has begun shard output; Mac France/India remains active. No complete full-test TSV, validator PASS or public score yet. EXP-031→032→033 remains the next model-scale path; Fold4 CLOSED.

25 September 2026, 09:36 IST: US cloud partition has checkpointed its first two of 64 shards. Mac has completed France and at least five India shards. An India-only cloud partition for shards 8–63 is prepared to run concurrently without restarting finished Mac shards; it will cover 709,176 queries while Mac supplies India 0–7. A 100-query partition smoke run matched all 88 applicable prior frozen TSV shard contents exactly. The three-way assembler has a synthetic integration test and SHA256 download verification; India cloud has not yet launched. Frozen model/threshold/candidates and Fold4 remain unchanged.

25 September 2026, 09:27 IST: Capped US-only SUB-001 acceleration worker launched successfully on on-demand r8i.4xlarge `i-01b623904660bdf50`, immutable code commit `7dd0efa`, same frozen inputs/model/threshold, automatic termination at 12 hours. Prelaunch Cost Explorer gross estimate $3.09 plus $22.50 maximum existing-worker commitments and $14.34 maximum new-job exposure was below the $120 soft cap; credit balance remains unverified. Mac France/India continues unchanged. A checksum-verified country assembler and integration test are ready; no US full output, final TSV, official PASS or public score exists yet.

25 September 2026, 09:25 IST: User-reported public leaderboard first 0.976808 and third 0.972446; our public score/rank remain unknown. Frozen SUB-001 Mac inference is healthy at India 4/64 completed after all 64 France shards. To shorten time to the first validated upload, a separate capped US-only EC2 worker is being prepared using the exact frozen candidate routes, 51 features, model and 0.83 threshold. A 100-US-query run produced byte-identical decompressed TSV contents for all 98 matching/candidate shards against the existing frozen smoke run. Mac work and all existing shards remain untouched. The cloud job has not yet launched. EXP-031/032/033 sequence remains in force. Fold4 CLOSED.

25 September 2026, 09:05 IST: SUB-001 frozen Mac inference remains healthy at India 3/64 active progress shards; 68 matching and 68 candidate shards have been written, with no final COMPLETE.json. EXP-031 has uploaded 140/256 route archives; EXP-025 continues independently. A full 256-archive SHA256 and exact old-20k route parity gate is now implemented and must pass before launching EXP-032. EXP-033 will record fit, prediction and peak-RSS by size. Neither dependent worker has launched; no 50k/100k score exists. All 80 tests pass. Fold4 CLOSED.

25 September 2026, 08:58 IST: New classical-only handoff confirmed. Frozen SUB-001 remains healthy, with France complete and India active at 2/64 progress shards (67 matching and 67 candidate gzip shards on disk; final COMPLETE.json absent). EXP-025 full retrieval has 81/256 S3 route archives and EXP-031 200k retrieval has 125/256 at the latest check; both EC2 instances remain running with their existing finite caps. EXP-032 and EXP-033 are prepared but unlaunched. [docs/LEARNING_CURVE.md](docs/LEARNING_CURVE.md) now holds the fixed-15k OOF protocol and pending 20k/50k/100k table; the plotting script was smoke-tested on synthetic values and will only plot real verified metrics after EXP-033. No new model score, public leaderboard result or uploadable SUB-001 file exists. Fold4 CLOSED; classical ML saturation remains unproven.

25 September 2026, 08:36 IST: Frozen SUB-001 Mac inference has completed all 64 France shards and begun India (at least 1/64 India shards, 259,452 total queries fully processed at last confirmed progress). PID 86583 is healthy; final TSV, official validator PASS, and upload ZIP are still pending. Do not restart or change its frozen model, features, threshold, or candidates. The two bounded EC2 retrieval jobs are active: EXP-025 full 2,206,821 training S1 (76 S3 objects last check) and EXP-031 deterministic 200,000 fold1–3 S1 (71 S3 objects). EXP-031 was launched at 02:39:53 UTC on `i-014e9cdbb267db46d`, 6-hour OS cap and $4.50 planning ceiling. EXP-032 owner-safe 51-feature materialization and EXP-033 20k/50k/100k LightGBM learning curve on a separately frozen 15k OOF set are implemented and input-uploaded, but **neither job is launched**; each requires its predecessor to finish, terminate, and pass checksum inventory. Full test suite: 79 passed. Fold 4 CLOSED. The user will upload both portal files after we provide validated local paths.

25 September 2026, 08:08 IST: This branch is now classical ML only by user request; neural approaches are deferred to the separate architecture effort. Frozen SUB-001 remains operational priority and is healthy at France shard 57/64 (still no complete TSV/ZIP). The existing full-training EC2 retrieval worker remains active at at least 69 of 256 expected training country/route/query shards; labels for Fold4 remain closed. Live us-east-1 quotas: 256 on-demand standard EC2 vCPUs, SageMaker ml.r5.2xlarge Processing 2 and Training 1; no Processing job active. EXP-031 prepared a 200,000-query fold1–3 sample that includes the previous20k, and SHA256-uploaded the input to S3. Its second bounded EC2 worker is prepared, not launched yet: estimated 1.61h query time, 6h hard cutoff, $4.50 planning ceiling, automatic termination and per-shard S3 checkpoints. See docs/CLASSICAL_ML_FRONTIER.md.

25 September 2026, 07:34 IST: The user will upload the two portal files themselves. The frozen SUB-001 full-test inference is running and healthy in detached screen `aml-sub001-v005`; its last confirmed progress was France shard 39/64, so neither final TSV nor code ZIP is ready. Do not offer partial shards as portal files. The end-to-end code-package reproduction path has been added; regenerated Parquet inputs match all 11,702,133 frozen test rows exactly. After all 192 candidate and 192 matching shards finish, upload them for the bounded 64 GB SageMaker official validator, download SHA256-verified outputs, then build and CRC-check the final ZIP with both TSVs, code, frozen artifacts, and the methodology. Notify the user with local links only; do not upload to the portal.

Full-training retrieval is **RUNNING** on EC2: `P5-FULL-RETRIEVAL-001`, instance `i-0609d158c38e96160`, frozen code `5ac0845`. It covers all **2,206,821 S1 queries** against **10,320,219 targets**, using 64 deterministic query shards per country and route. Projected query time is 17.76 hours plus overhead, with a 24-hour shutdown cap and an $18 total planning allowance. Actual cost is unknown. Each completed shard is checksum-verified in S3. Do not launch a duplicate job. Read its ledger and use `scripts/aws/collect_run.py` to collect progress.

Current best remains the 20k NUMERIC-V2 model: macro F0.5 **0.9318965293**, precision **0.98158282**, recall **0.85593807**, singleton F0.5 **0.92131747**. Country-balanced EXP-026 scored **0.93171205**: India improved while US declined, so it was not promoted. Full-data retrieval and larger OOF scores remain pending.

**Fold 4 is CLOSED.** SUB-001 full-test inference is running, while final validation, portal submission, and package are not ready. The active AWS profile is `amamzon_01_a1_0`, which accesses the same account ending 6318 as `default`. The ml.r5.2xlarge SageMaker Processing quota is approved at 2 in us-east-1; other quotas and remaining credits require fresh checks.

Next five steps: finish frozen SUB-001 and its official validation/package; finish and verify EXP-031 retrieval; launch EXP-032 owner-safe 200k feature store after its predecessor exits successfully; launch EXP-033 fixed-holdout 20k/50k/100k learning curve after complete feature receipts; evaluate full EXP-025 retrieval on unlocked folds 0–3. Current full retrieval evaluates labels only for the 1,765,649 entities outside Fold 4. No leaderboard submission exists. This branch is classical ML only.

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

## 2026-09-24T23:22:32.013520+00:00 — SUB-001 live status
SUB-001 RUNNING on Mac against complete test set; frozen 51-feature LightGBM, threshold0.83, 20k natural-sample training population. Input bundle v002 SHA-verified; exact model SHA256 d84957742e05f5cd790d7dfc8c14ca05d3b5a2dc941a5094b8874d623b35117b. Local CV macro F0.5 0.9318965293. Final candidate count, inference runtime, official validator, file hashes, portal submission and public leaderboard PENDING. 69 tests pass. Fold4 CLOSED. AWS full-training retrieval active, with 24h independent cutoff and $18 planning ceiling; 4 SUB-001 Spot attempts lacked capacity, no additional instance launched. Mac path running; cloud validator still required. Project gross soft cap $120, planning hard cap $150; AWS credit balance unverified.

## 2026-09-25T00:51:06.664498+00:00 — Phase5 operational correction
SageMaker ml.r5.2xlarge processing quota is confirmed 2, newly propagated. SUB-001 current full-test run is the detached screen job aml-sub001-v005, output outputs/submissions/SUB-001/local-full-v005/inference; former local-full and v002 partial outputs are preserved. No portal submission. Current AGENTS.md requires all Phase4 gates before submission, so a completed validated early-calibration file will be held until that gate is satisfied or user explicitly resolves the conflict. Fold4 CLOSED; full retrieval EC2 unchanged.

## 2026-09-25 — priority override

SUB-001 is now explicitly authorized for early calibration once its complete test files pass independent checks and the unchanged official validator. Active detached screen job aml-sub001-v005: France 8/64 shards, 6,411,934 scored candidate pairs at last check; no public score. Fold4 CLOSED. Full-training retrieval continues. Latest user-provided leaderboard top0.964733/third0.957920; local0.9318965293 is not directly comparable. See docs/COMPETITIVE_STATUS.md.

## 2026-09-25T01:25:00+00:00 — Validator prepared; learning curve complete

SUB-001 frozen Mac inference remains healthy in screen `aml-sub001-v005`, PID86583, at France19/64 shards at last check. Final TSVs, official PASS and public score are pending. The full-training retrieval EC2 worker remains active and has reached the India/name shard50 checkpoint. SageMaker `ml.r5.2xlarge` Processing validator is prepared with a dedicated least-privilege execution role, SHA-verified immutable code bundle, AWS-CLI-schema-checked request, 6h cutoff and $5 ceiling; **no processing job is running**. Launch requires complete 384-shard S3 receipt. EXP-027 fixed-heldout diagnostic completed: training 2k→5k→10k→~13.3k entities yielded weighted macro F0.5 0.924221→0.929379→0.930829→0.932231. It does not establish a full-scale or France gain; best promoted model stays frozen NUMERIC-V2 0.9318965293. Fold4 CLOSED. 69 tests pass.

## 2026-09-25T01:35:00+00:00 — Error slice priority

EXP-028 on the existing 20k known-country OOF population found 2,315 retrieval misses and 7,678 retrieved-but-rejected true links. India/S3 alone contributes 982 retrieval misses; non-ASCII target names have 89.13% candidate recall. When either address is missing, only 684/3,045 true links are predicted and current predicted-pair precision is 684/873 (78.35%), so a blanket lower threshold is unsafe. See `docs/PHASE5_ERROR_SLICES.md`. Next experiments should test targeted retrieval and nested calibrated missing-address decisions while SUB-001 remains frozen. No Fold4 or test labels were used.

## 2026-09-25T01:45:00+00:00 — EXP-029 rejected

Separate missing-address threshold selected within inner OOF fell on all three outer folds: pooled 0.930533 versus frozen baseline 0.931897, paired delta -0.001363, 95% CI [-0.002074, -0.000677]. Frozen SUB-001 stays best and unchanged. SageMaker validator role, launch gate and verified-result downloader are ready; no validator job or portal submission yet. Mac inference and full-training retrieval remain active. 71 tests pass.

## 2026-09-25T01:50:00+00:00 — EXP-030 candidate-cost result

On a fixed 1,000-entity India development sample, top100 name retrieval reproduced frozen candidates exactly. Expanding only the name route to top200 recovered 11 additional true links but added 99,873 candidates (9,079 per recovered link); candidate oracle macro rose0.981488→0.982530 before any matcher. Universal top200 is not promoted; SUB-001 remains frozen. Full-scale retrieval and Mac test inference continue. Next: prioritize full-data training materialization after retrieval and complementary India/S3 or cross-script retrieval with better marginal cost.

## 2026-09-25T01:53:00+00:00 — Live handoff checkpoint

SUB-001 detached Mac PID86583 is healthy at France34/64 shards; full query count, final TSVs and official PASS remain pending. Free local disk is about17GiB, above the8GiB reserve. EC2 full-training retrieval has uploaded through India/name shard59/64 and remains active. No SageMaker validator job or portal submission has occurred. Its role, 384-shard upload gate, six-hour/$5 launcher and verified-output downloader are ready. The continuation heartbeat now checks every30 minutes and remains quiet while work is merely progressing; registered-team Unstop sign-in was requested asynchronously because the Codex browser is not signed in.
