# Append-only timeline

All timestamps include timezone. Never replace historical entries.

## 2026-09-25T00:46:46.247511+05:30 | Codex | Initialization
- Action: inspected complete workspace; read source documents and supplied code.
- Commands: rg --files; find student_resource; pypdf extraction; sysctl; df.
- Config: M5, 10 CPU cores, 24 GiB unified RAM, 24 GiB available disk.
- Result: seven TSVs, two PDFs, README, template, validator, original ZIP. No existing Git repo.
- Interpretation: research-first local processing feasible; all-pairs comparison infeasible; storage constrained.
- Next: install isolated tools, audit every source, document discrepancies.

## 2026-09-25T01:01:23.120469+05:30 | Codex | Phase-one evidence
- Action: full six-source/ground-truth audit, full positive-pair comparison, near-neighbor diagnostics, entity folds, metric/TSV tests; official Colab CLI installed and login verified.
- Commands: src.audit_data; src.audit_pairs; src.build_validation; pytest; colab usage; colab sessions.
- Config: 4 CPU workers, DuckDB 6 GB memory limit, folds seed 20260925.
- Result: 24,229,173 source records, 7,638,365 labeled links, zero shared target owners or cross-country positive links; 22 tests pass. Colab 0 units, no sessions. AWS saved session expired.
- Interpretation: large candidate pool; multi-match output and cross-script recall are essential. No model training or cloud spend.
- Repairs: audit attempts 001/002 hit reserved SQL alias names and were retained as failed logs; 003 completed. Unicode audit normalization was then corrected to preserve combining marks; authoritative audit is AUDIT-004/PAIR-002, older reports preserved. Colab 0.7.1 unavailable; PyPI 0.7.2 installed.
- Next: finalize report, transliteration probe and EXP-001 diagnostic.

## 2026-09-25T01:08:47.520574+05:30 | Codex | Research phase complete
- Action: measured exact baseline, generic transliteration sample and independent per-file memory; completed research, compute plan, roadmap and handoff prompt.
- Commands: src.exact_baseline --config configs/experiments/EXP-001.json; src.audit_supplement; src.audit_memory; colab usage; colab sessions.
- Config: dev fold 0, seed 20260925, full training target pool; non-ASCII positive probe n=10000; no fitting.
- Result: macro F0.5=0.08327203, precision=1.000000, recall=0.01314345, pairs=20075; Colab authenticated with 0 units and no sessions.
- Interpretation: exact matching is only a diagnostic; union retrieval and cross-script rescue have highest priority. Generic transliteration helps but does not solve all India noise.
- Next: EXP-002 lexical retrieval; AWS remains unused, learned training not started.

## 2026-09-25T01:12:50.224359+05:30 | Codex | Final research verification
- Action: integrated candidate-generation test and completed artifact consistency review.
- Commands: pytest; compileall; git diff --check; JSON/CSV and inventory assertions.
- Result: 23 tests passed; six independent parser row counts agree; split covers all 2,206,821 S1; label-integrity and forbidden-target checks pass; no leaderboard submissions; current data/pair/memory evidence preserved in Git.
- Interpretation: phase-one infrastructure and measured diagnostics are ready for the retrieval work. One-command research wrapper is supplied; its individual stages were verified, not redundantly rerun as a full wrapper.
- Next: EXP-002.

## 2026-09-25T01:33:04.179803+05:30 | Codex | Phase 2 start
- Action: read Phase 2 instructions and existing audit/research/validation/normalization; preserved all prior artifacts and splits.
- Commands: git rev-parse HEAD; pytest; read project records.
- Config: starting commit 5d50551204f7a677c982d3c2f18ced7019173302; M5/24 GiB; local-only bounded independent analyses.
- Result: clean initial tree, 23 existing tests passed, ~20 GiB disk free.
- Interpretation: preprocessing evidence and visualization precede model fitting. User reports AWS CLI configured in us-east-1 with ~$200; verify authentication read-only.
- Next: full profiles/vocabulary, positive/hard-negative features, composable preprocessing, normalization ablations and charts.

## 2026-09-25T01:54:17.719946+05:30 | Codex | Phase 2 evidence and first retrieval pilot
- Action: full-source profiling, 34 scientific figures, train/test shift, tested normalization, train-only mapping proposals, hard-negative diagnostics, transliteration and full-target lexical retrieval.
- Commands: src.analysis.profile_dataset; src.preprocessing mine-benchmark; src.analysis.pair_morphology; src.analysis.transliteration_diagnostics; src.blocking.token_candidates; scripts/render_preprocessing_report.py; scripts/analyze_candidate_pilot.py; pytest.
- Config: PROFILE-001; PREP-001; MORPH-002; TOKEN-001/run-002; folds 1–3 fit / 0 development / 4 closed; bounded local CPU jobs.
- Results: 675 profile integrity checks PASS; 44 unit/integration tests PASS; 77.33% full-target candidate recall on 1,000 queries, 179,298 pairs. No learned score replaces EXP-001. AWS read-only authentication succeeded; no resources launched; task spend $0.
- Repairs: MORPH-001 metadata failed due to variable shadowing, preserved and rerun as MORPH-002. TOKEN-001 first correlated-UNNEST plan hit scratch cap; preserved failure/source and rerun successfully using streamed partitions. Candidate analysis TEMP VIEW parameter binding changed to TEMP TABLE before successful run.
- Interpretation: raw/Unicode-safe parallel views are essential; exact token routes leave substantial recall gaps. Aggressive normalization is not universally better; ambiguous learned maps remain disabled.
- Next action: character-ngram and transliteration candidate rescue, larger stratified retrieval benchmark, no model training until coverage improves.

## 2026-09-25T02:15:13.801446+05:30 | Codex | Phase 3 begins
- Starting commit a716ccc57e620026c3e3c50a80ebbe70e5ee74fe; all 44 tests passed; 20GiB free disk.
- AWS authentication and configured us-east-1 region verified read-only; no secrets logged.
- User adopts Phase 3 as adaptable guidance: public methodology research, candidate recall, S3 foundation, models only after retrieval evidence.
- Exact metric count form retained: 1.25TP/(TP+FP+0.25FN); both-empty=1. No claim that beta=0.5 means only a 2x FP penalty.
- Next: bounded parallel sparse retrieval/research/Parquet preparation; immutable raw upload with verified cost and access controls.

## 2026-09-24T21:09:35.946477+00:00 | Codex | Phase 3 measured checkpoint
- Completed six full-target char3/4/5 routes, transliteration, expanded tokens, union ablations, two lexical GBDTs and multiview/hard-negative refinement.
- Best calibration-selected pilot check macro 0.910560; broad union link recall 0.976225. Not final holdout/OOF. Hard-negative weighting rejected on calibration.
- S3 raw/processed SHA256 verified; 35 Parquet shards preserve all raw records. No rented compute.
- Failures preserved: TRANS-001 temp-view connection scope, GBDT-001 reserved label alias, GBDT-003 reserved owner alias. Fresh runs succeeded.
- Next: larger entity-split training, production retrieval throughput, singleton/OOF calibration and residual error rescue.

## 2026-09-25 — Codex Phase4 preparation
- Request: freeze Phase3, prove validation credibility; preserve fold4 and submission gates.
- Actions: froze BASELINE-P4-001 with exact hashes; generated nested natural5k/20k/50k/100k and separate stratified diagnostics; added disjoint-IDF retrieval, nested3-fold OOF and owner exclusion tests.
- Commands: scripts/phase4_prepare.py; src.blocking.phase4_retrieval --sample A --output outputs/candidates/P4-A-001; pytest.
- Results: samples complete;54 tests pass;5k retrieval running. Initial DuckDB COPY parameter-order failure preserved in phase4-v001-failed-copy-parameters; repaired integer-bound COPY. Pilot diagnostics initially required unavailable Arrow; changed to bounded ordinary row transfer and preserved failed directory.
- Interpretation: old0.91056 remains development-only. Char3-only recall96.35%;96.78% also includes tokens. Expanded token marginal cost493,006 pairs/7links.
- Next: complete retrieval, run nested OOF, report real scores with uncertainty. No paid compute or fold4 evaluation.

## 2026-09-24T21:35:48.340483+00:00 — Codex measured Phase4 checkpoint
- Run outputs/oof/P4-A-001; pooled nested OOF macro 0.923407; CI [0.9185769101763813, 0.9279348540650086]; entities 5000; folds3.
- Retrieval {'query_count': 5000, 'true_links': 17345, 'retrieved_links': 16764, 'candidate_pairs': 985945, 'target_pool_count': 10320219, 'link_recall': 0.9665033150763909, 'oracle_macro_f0_5': 0.9891492764644434, 'oracle_non_singleton_f0_5': 0.9885395822395896, 'oracle_singleton_f0_5': 1.0, 'positive_entity_any_coverage': 0.9974651457541192, 'positive_entity_all_coverage': 0.9034643008027038, 'singleton_count': 266, 'singleton_rate': 0.0532, 'singleton_candidate_free_rate': 0.0, 'average_candidates_per_query': 197.189, 'candidate_quantiles': {'p50': 197.0, 'p95': 200.0, 'p99': 200.0}, 'max_candidates_per_query': 200, 'candidate_reduction_ratio_vs_unrestricted_pool': 0.9999808929442292, 'by_country_source': [{'country': 'India', 'target_source': 'S2', 'true_links': 3450, 'retrieved_links': 3304, 'link_recall': 0.9576811594202899}, {'country': 'India', 'target_source': 'S3', 'true_links': 3721, 'retrieved_links': 3474, 'link_recall': 0.9336199946251008}, {'country': 'US', 'target_source': 'S2', 'true_links': 4920, 'retrieved_links': 4836, 'link_recall': 0.9829268292682927}, {'country': 'US', 'target_source': 'S3', 'true_links': 5254, 'retrieved_links': 5150, 'link_recall': 0.9802055576703465}], 'by_country': [{'country': 'India', 'queries': 2087, 'link_recall': 0.9451959280435086, 'oracle_macro_f0_5': 0.9806798982423558, 'singleton_count': 124}, {'country': 'US', 'queries': 2913, 'link_recall': 0.9815215254570474, 'oracle_macro_f0_5': 0.9952171076863784, 'singleton_count': 142}]}.
- Decision comparisons recorded in docs/PHASE4_CHECKPOINT.md. Fold4 CLOSED, no submission or paid compute.
- Next: larger natural sample, error-driven ablations and rescue; retain all outputs.

## 2026-09-25 — Phase4 ablations and retrieval diagnostics
- Eight nested feature ablations completed on the same5k candidates/folds. Removing numeric/transliteration/retrieval/token/character features lowers macro with paired intervals excluding zero; removing all name/address evidence causes large losses. Route-count effect is inconclusive.
- Exact-name collision rescue adds830 candidates for5 links/5 complete entities in2.36s;126 queries triggered from observable fanout>100. Not promoted to matcher yet.
- Full-country name index probe preserved:40.32s build,55.60s store,81.98s/1000 queries search, perfect reference candidate parity. No demonstrated search speed win; retain chunked retrieval.
- P4-B-001 stopped before completed route to remove repeated transpose; P4-B-002 continues with transpose once per target chunk. No fold4 or leaderboard activity.

## 2026-09-24T21:57:24.552608+00:00 — verified Phase4 S3 checkpoint and kernel parity
- Private existing-bucket conditional uploads:2objects/205762384bytes, service SHA256/size verified.20k ongoing files excluded.
- Full-pool name retrieval parity: all500,000 pairs and scores identical for nested5k queries between reference and fused20k run. Address parity pending completion.
- Query-context features did not improve OOF (0.922214 vs0.923407); not promoted. Alternative canonical numeric view now under nested validation; preserves raw digits.

## 2026-09-24T22:08:59.720444+00:00 — Codex measured Phase4 checkpoint
- Run outputs/oof/P4-B-001; pooled nested OOF macro 0.928759; CI [0.9264584347104737, 0.9310251784781759]; entities 20000; folds3.
- Retrieval {'query_count': 20000, 'true_links': 69366, 'retrieved_links': 67051, 'candidate_pairs': 3943627, 'target_pool_count': 10320219, 'link_recall': 0.9666263010696883, 'oracle_macro_f0_5': 0.9885314605341274, 'oracle_non_singleton_f0_5': 0.9878684725595047, 'oracle_singleton_f0_5': 1.0, 'positive_entity_any_coverage': 0.996985243560586, 'positive_entity_all_coverage': 0.9030517797641086, 'singleton_count': 1093, 'singleton_rate': 0.05465, 'singleton_candidate_free_rate': 0.0, 'average_candidates_per_query': 197.18135, 'candidate_quantiles': {'p50': 197.0, 'p95': 200.0, 'p99': 200.0}, 'max_candidates_per_query': 200, 'candidate_reduction_ratio_vs_unrestricted_pool': 0.9999808936854925, 'by_country_source': [{'country': 'India', 'target_source': 'S2', 'true_links': 13576, 'retrieved_links': 12926, 'link_recall': 0.952121390689452}, {'country': 'India', 'target_source': 'S3', 'true_links': 14525, 'retrieved_links': 13543, 'link_recall': 0.9323924268502581}, {'country': 'US', 'target_source': 'S2', 'true_links': 19959, 'retrieved_links': 19647, 'link_recall': 0.984367954306328}, {'country': 'US', 'target_source': 'S3', 'true_links': 21306, 'retrieved_links': 20935, 'link_recall': 0.9825870646766169}], 'by_country': [{'country': 'India', 'queries': 8116, 'link_recall': 0.941923774954628, 'oracle_macro_f0_5': 0.9789356903597326, 'singleton_count': 441}, {'country': 'US', 'queries': 11884, 'link_recall': 0.9834484429904278, 'oracle_macro_f0_5': 0.9950847482096066, 'singleton_count': 652}]}.
- Decision comparisons recorded in docs/PHASE4_CHECKPOINT.md. Fold4 CLOSED, no submission or paid compute.
- Next: larger natural sample, error-driven ablations and rescue; retain all outputs.

## 2026-09-24T22:13:50.753312+00:00 —20k numeric confirmation complete
- Macro0.931896529 versus baseline0.928758931; paired delta+0.003137599. New15k-only delta+0.003299244 with CI[0.001839712,0.004700534].
- Interpretation: retain canonical numeric comparison features as current development best; no automatic number-equivalence merge rule. Country transfer remains weak.
- Next: targeted lexical rescue, singleton modeling, larger validation; no fold4/submission.

## 2026-09-24T22:34:01.376480+00:00 — Codex — Phase5 cloud foundation
Read latest user directive; inventoried configured profiles, EC2/SageMaker quotas and instance offerings. Named profile refreshed by user; same account as default. Requested CPU64/G8/SageMakerG5=1 quotas (pending). Created scoped EC2 role and no-ingress group, uploaded checksummed unlabeled10,320,219-target benchmark inputs. Persistent source/country index merger test passed, including boundary ties/unseen Unicode country. Preparing EXP-02490minR8i benchmark. Fold4 closed; no leaderboard submission.

## 2026-09-24T22:47:14.680824+00:00 — Codex — cloud benchmark completed
P5-INDEX-001 / EXP-024 built6.078GB full-target indexes; all4country/route candidate parity tests passed at2/4/8threads. S3uploads verified; exit0; EC2terminated.63tests passed before retrieval evaluator; its2new tests additionally verify oracle and Fold4rejection.20k best checkpoint backed up. FullS1text2,206,821exported with no labels; separate evaluation export excludesFold4 (1,765,649queries). Full-run17.76h extrapolation,24hcap,$18planningceiling. Next launch EXP-025 with checkpointed shards.

## 2026-09-24T22:52:42.142603+00:00 — Codex — full training retrieval running and country trial
EXP-025 / P5-FULL-RETRIEVAL-001 launched i-0609d158c38e96160, code5ac0845, all2,206,821trainingS1,24hshutdown. Inputs/indexes checksummed; output shards versioned toS3. NoFold4labels supplied. EXP-026country balancing20k completed:0.93171205 vs0.93189653; India +0.00196164,US -0.00165014. Not promoted. Best staysNUMERIC-V2. Researched/pinned candidate embedding licenses; none selected or executed.

## 2026-09-24T22:54:18.545341+00:00 — Codex — full-run first checkpoint verified
EC2 console reports India/name shard0 completed:1,380,300routepairs in182.8521seconds. S3contains first shard archive10,864,640bytes; worker upload uses conditional creation and SHA256head verification before reporting completion. This is one name-route shard, not full union metrics. Fullrun remains active.66tests pass. NoFold4evaluation, finalvalidator or submission.

## 2026-09-24T23:22:32.013520+00:00 — Codex — SUB-001 frozen and local full inference started
- Frozen 51-feature LightGBM NUMERIC-V2 from 20,000 natural-sample entities; threshold 0.83; Fold4 CLOSED. Model SHA256 d84957742e05f5cd790d7dfc8c14ca05d3b5a2dc941a5094b8874d623b35117b. Code tag SUB001-BASELINE; immutable v002 test inputs checksum verified in S3.
- Test: 1,732,544 S1 and 9,969,589 S2/S3 targets, including France. Six-query and 300-query all-country smoke runs completed; 69 tests pass. Full Mac run outputs/submissions/SUB-001/local-full/inference is active. Final pairs, runtime, hashes, official validator and leaderboard are pending.
- Gross budget guard measured $0.0099310264 estimated month-to-date billing at prelaunch; other active full-retrieval worker maximum commitment $18. Proposed bounded Spot max $11.80; project worst-case $29.81 < $120 soft/$150 hard. $25/$50/$75/$100/$125 gross budget alerts configured. Credit balance unverified.
- Four Spot attempts across R8i/R6i/R7a/R7i failed on EC2 capacity before instance creation; no SUB-001 EC2 running. Training full-retrieval EC2 i-0609d158c38e96160 remains running with 24h OS cutoff.

## 2026-09-25T00:51:06.664498+00:00 — Codex — quota propagated and detached inference recovery
- Verified SageMaker ml.r5.2xlarge processing quota now 2 in us-east-1; no processing job launched. On-demand EC2 full retrieval remains active; new SageMaker capacity is optional and has its own capped-job gate.
- The prior foreground Mac inference ended at 5 completed France shards when its tool session ended. Preserved that output unchanged. A nohup retry and launchctl retries also ended; preserved their versioned directories. Relaunched the same frozen input/model in a detached screen session aml-sub001-v005 under caffeinate; separate process check showed Python PID86583 active. New output is outputs/submissions/SUB-001/local-full-v005/inference. No final candidate count, validator, hashes or submission yet.
- Updated continuation automation to inspect v005 and obey the latest AGENTS.md Phase4 submission gates. Fold4 CLOSED.

## 2026-09-25T01:05:07.486323+00:00 — Codex — SUB-001 prioritized by explicit user override
- Read latest user request and official output-format excerpt. User explicitly authorized early SUB-001 submission after official PASS without waiting for full retrieval/Fold4; Fold4 remains CLOSED. Frozen model/threshold/candidate routes unchanged.
- Detached screen aml-sub001-v005 healthy at France shard8/64,6,411,934 candidate pairs scored. 1,732,544 test S1 total. No final output, hashes, validator or public score yet.
- Created docs/COMPETITIVE_STATUS.md with user-reported leaderboard references and pending public gap. Updated automation and documentation to obey this single early-calibration override.

## 2026-09-25T01:25:00+00:00 — Codex — validator role ready and EXP-027 complete
- Mac SUB-001 screen job remained healthy through France19/64 shards; no model/config change, no final files or submission. Full-training EC2 retrieval remained active, reaching India/name shard50.
- Created dedicated SageMaker Processing role `aml2026-phase5-sagemaker-validation` in account ending6318, trust restricted to exact processing-job ARN/account, S3 read/write prefixes scoped to frozen inputs/results, ECR read to pinned official sklearn repository, and processing log group permissions. Source request passed AWS CLI schema validation. No paid SageMaker job launched; `scripts/aws/launch_sagemaker_validator.py` requires 384 SHA-verified uploaded shards, checks other active processing jobs, EC2 commitments and $120/$150 project caps, then launches only with explicit `--launch`; 6h hard cutoff and $5 planned ceiling.
- EXP-027 (`outputs/experiments/EXP-027-v002/metrics.json`) finished on fixed held-out folds1–3; weighted macro F0.5 at fit sizes2k/5k/10k/~13.3k was0.924221/0.929379/0.930829/0.932231. Diagnostic fixed thresholds limit causal interpretation and do not predict France. Logged EXPERIMENTS.csv. Tests:69passed. Next: finish Mac shards, upload/checksum, execute official validation, submit SUB-001 via team portal and record observed score.

## 2026-09-25T01:35:00+00:00 — Codex — EXP-028 frozen-model error slices
- Reproducible read-only diagnostic on 20k natural folds1–3, frozen candidate union and NUMERIC-V2 predictions. Two initial output directories were created by SQL binding/parser failures; preserved, empty. Corrected `scripts/analysis/phase5_numeric_error_slices.py` and produced versioned `outputs/analysis/P5-NUMERIC-ERRORS-004/report.json`; no model or submission change.
- Of69,366 true links,2,315 were absent from candidates and7,678 were candidates rejected by the matcher. India/S3 had982 retrieval misses; missing-address links had501 retrieval misses and1,860 matcher misses, with current predicted precision684/873. See docs/PHASE5_ERROR_SLICES.md. Interpretation: investigate targeted retrieval and nested calibrated decisions, not blanket acceptance. Fold4 CLOSED.

## 2026-09-25T01:45:00+00:00 — Codex — EXP-029 threshold split rejected
- Pre-registered a missing-address threshold grid and trained six one-fold inner models on the existing 20k features. Ownership-safe masks and frozen full outer models reproduced the published baseline fold scores exactly. No Fold4/test labels. Inner OOF selected 0.90 on each outer fold; pooled outer macro F0.5 0.930533 versus0.931897 baseline, paired delta-0.001363 and 95% CI[-0.002074,-0.000677]. Reject; do not alter SUB-001.
- Prepared `scripts/submissions/download_validated.py` to require completed SageMaker job, exact file inventory, per-object SHA256, raw full-TSV SHA256, official PASS and zero strict warnings before writing READY.json. Full suite71passed. Mac test inference and full-training EC2 retrieval remain active; no validator processing job or portal submission.

## 2026-09-25T01:50:00+00:00 — Codex — EXP-030 India name top200 probe
- Fixed first1,000 India S1 by hash within natural20k folds1–3; full4,133,346-target local index, byte-identical Phase4-B IDF. Top100 candidate parity0 mismatches. Top200 added99,873 candidates and recovered11 true links; link recall0.945977→0.949172, oracle macro0.981488→0.982530, runtime105.63s and peak RSS1.94GiB. Universal expansion rejected on candidate cost; no model or SUB-001 modification. Labels were read only for evaluation after retrieval. Fold4 CLOSED.

## 2026-09-25T01:53:00+00:00 — Codex — live job and continuation checkpoint
- Verified detached Mac SUB-001 PID86583 healthy at France34/64 shards, with about17GiB free disk. Full-training EC2 retrieval uploaded India/name shard59/64. Neither job has completed; no SageMaker validator or portal submission has launched.
- Updated existing `amazon-ml-phase-5-continuation` heartbeat to check every30 minutes, perform checksum upload, budget-gated strict validation and portal submission once ready, and stay quiet while unchanged. Requested registered-team Unstop sign-in asynchronously; current Codex portal tab is not authenticated.

## 2026-09-25T02:06:00+00:00 — Codex — prepare user-upload artifacts while SUB-001 runs
- User confirmed they will upload both portal files themselves. Updated the existing continuation heartbeat accordingly; it will finish full inference, SHA256-upload shards, run the bounded SageMaker official validator, build the ZIP, and provide local file links, without portal access.
- Added `src/submission/prepare_inputs.py`, `merge_outputs.py`, and `reproduce.py`; copied frozen `run_sub001.py` byte-for-byte into the package source. Full raw-TSV preprocessing reproduced all 1,732,544 queries and 9,969,589 target rows in exactly the same sorted logical representation as the frozen inference inputs (zero row differences). Removed the temporary parity output after verification.
- Added `scripts/submissions/build_final_package.py` to require READY.json, both official PASS results, strict zero warnings and exact matching/candidate SHA256s before assembling the official ZIP. It includes both TSVs, runnable source, frozen model/IDF/transliteration artifacts, requirements, validator and filled methodology. A temporary synthetic package passed ZIP CRC and file-inventory smoke checks; the synthetic package was deleted.
- Updated code README, methodology and runbook; verified installed LightGBM 4.7.0 declares MIT. Full test inference remained healthy at France 39/64 shards; neither real portal artifact is ready. No submission or paid validator job launched. Full test suite: 67 passed. Fold 4 CLOSED.

## 2026-09-25T03:06:19+00:00 — Codex — classical 200k learning-curve pipeline prepared

- Launched EXP-031 / P5-LEARNING-200K-001 at 02:39:53Z, instance `i-014e9cdbb267db46d`, frozen code `5245c87`; 200,000 fold1–3 S1, no labels or Fold4 queries. Six-hour OS shutdown, $4.50 planning ceiling, checkpointed 256 retrieval archives. At 03:04:05Z running with 71 S3 objects/74,096,640 bytes. A completed route archive reproduced 104 old P4-B-002 query routes and ranks (10,400 pairs; score difference at most 1.788e-7).
- Full EXP-025 retrieval also running at 03:04:41Z with 76 S3 objects/890,306,560 bytes. No duplicate launch. Frozen SUB-001 Mac PID 86583 completed France 64/64 and began India; 259,452 queries fully processed, at least one India shard written. Model and threshold untouched. No final TSV, official validation or portal upload.
- Prepared EXP-032 feature worker and EXP-033 20k/50k/100k fixed-holdout LightGBM worker at code commit `75cad642659bd873a5b27c271bdc9e30cabbd7cd`. Feature inputs: 200k labeled fold1–3 queries, 10,320,219 full targets plus owner folds, 691,582 truth pairs; separate 15k new OOF queries outside old20k. Inputs and manifests SHA256-uploaded. Both jobs remain unlaunched pending predecessor completion/termination, full receipt inventory and fresh cost guard. Shell syntax, Python compilation, tiny feature-store integration and full 79-test suite passed. Fold4 CLOSED.

## 2026-09-25T03:12:43+00:00 — Codex — dependent-job prelaunch guard verified

- Moved EXP-032/033 predecessor checks ahead of run-directory creation, code archive and S3 upload at commit `b85c982127f5f6eab5f87a98a90de47587288173`. A deliberate early EXP-032 command stopped at the expected incomplete-EXP-031 guard and left no EXP-032 run directory; no compute or S3 artifact was created. Full suite 79 passed.
- SUB-001 Mac run remained healthy during the check; France candidate shards total 258.9 MiB compressed. At least two India candidate shards had been written. Local free disk about 15 GiB; maintain the 8 GiB reserve while raw TSV and ZIP materialize. No final portal files yet.
- Corrected the EXP-033 prelaunch S3 inventory to follow every continuation token: the 200k feature store can exceed one 1,000-object page. Commit `56ca0867f4f4fa34c262b294b71c6a52fc4a8be6`; AWS CLI two-key page smoke check returned a continuation token, Python compilation and the full 79-test suite passed. No new cloud job launched.

## 2026-09-25T03:28:29+00:00 — Codex — new classical handoff and learning-curve report prepared

- Read the latest user handoff. Kept frozen SUB-001 unchanged, Fold4 CLOSED and both existing retrieval workers running. At this check SUB-001 had 67 matching and 67 candidate gzip shards, India active 2/64, no COMPLETE.json. EXP-025 had 81/256 route archives and EXP-031 had 125/256. EXP-032/033 remained unlaunched.
- Added `docs/LEARNING_CURVE.md` with a pending-only 20k/50k/100k table, exact fixed-15k OOF scope, prior-threshold caveat and a decision gate for larger training sizes. Added `scripts/analysis/plot_learning_curve.py` for overall/India macro F0.5; a synthetic three-size report generated a valid PNG, then the temporary file was removed. No synthetic score entered the project report. Commit `446ca38`.

## 2026-09-25T03:35:36+00:00 — Codex — EXP-031 parity gate and EXP-033 timing prepared

- EXP-031 continued to 140/256 S3 route archives; SUB-001 Mac inference remained healthy at India 3/64 progress shards, 68 matching and 68 candidate shard files, no COMPLETE.json. No new job launched.
- Added `scripts/aws/verify_sample_retrieval_parity.py` at commit `f516e1e21145769aa8b175bf6c6dd5e1a325c7af`: after EXP-031 completion it validates all 256 archive SHA256 receipts and exact IDs/ranks for every one of the frozen old-20k query routes against P4-B-002, with score tolerance 1e-5. Runbook commands download the archived output and require a PASS before EXP-032. Synthetic archive corruption and candidate mismatch tests passed; full suite 80 passed. Actual 200k result parity remains pending.
- EXP-033 worker now records fit time, prediction time and peak RSS by fold/size (commit `875e459a1376e864bcf1caed36b710b0f0ecdfdd`). This is instrumentation only; no 20k/50k/100k scores yet.

## 2026-09-25T03:38:53+00:00 — Codex — full retrieval report scaffold and AWS poll backoff

- Added `docs/FULL_RETRIEVAL_ANALYSIS.md` with pending-only full-population metrics, explicit positive-pair versus Source-1-entity slice denominators and Fold4 exclusion. No full retrieval metric was claimed before EXP-025 completion.
- Two concurrent read-only AWS CLI polls caused one transient HTTP 429 on `CreateOAuth2Token`; a subsequent single sequential S3 poll succeeded and found EXP-031 at 147/256 route archives. This is an auth-service throttle, not a worker failure. Runbook now calls for sequential polling and backoff. Frozen SUB-001 Mac process remained healthy at India 3/64; no new compute launched.

## 2026-09-25T03:54:28+00:00 — Codex — urgent leaderboard response and US-only acceleration prepared

- User reported public first 0.976808 and third 0.972446; our public score/rank remain unknown and are not compared directly with the local 0.9318965293 OOF. Prioritized SUB-001 files and official validation ahead of additional research.
- Mac frozen SUB-001 remained healthy: France 64/64 complete, India 4/64 complete at the latest check. Measured France 259,452 queries in 7,096 seconds and India four shards in 3,520 seconds including index build. The US has 663,106 queries, so serial Mac US inference would add material time after India.
- Prepared independent US-only execution with the exact same frozen input bundle, candidate routes, 51 features, model hash and threshold 0.83. The only new runner option limits the country; default full-run behavior is unchanged. All 98 matching/candidate gzip shards from the 100-US-query smoke run had byte-identical decompressed TSV contents versus the earlier frozen smoke run. Full suite: 80 passed; shell syntax and Python compilation passed.
- Proposed on-demand r8i.4xlarge job `P5-SUB001-US-001`: 12-hour independent shutdown, hourly planning rate $1.11136, $15 all-in ceiling, per-shard SHA256 S3 checkpoint, terminate-on-shutdown. Expected benefit is wall-clock time to the first upload, not a change in model score. Cloud launch and cost guard remain pending at this entry; do not stop or overwrite the Mac run.

## 2026-09-25T03:57:11+00:00 — Codex — US acceleration launched; hybrid assembly prepared

- Cost guard read gross billed estimate $3.0928398446, active worst-case commitments $22.50, proposed 12-hour maximum $14.33632 and projected project worst case $39.9291598446, below the $120 soft cap. Credit balance remains unverified.
- Launched `P5-SUB001-US-001` on on-demand r8i.4xlarge `i-01b623904660bdf50`, code commit `7dd0efae5b001d2ec48c643f8de5929fcffefcdc`; independent 12-hour OS shutdown and terminate-on-shutdown EBS. Mac process remains active and unchanged. Full US results and public score are pending.
- Added a country assembler that SHA256-checks every cloud US shard receipt, verifies 64 shards for each of France, India and US, checks unique deterministic S1 assignment and creates a full 1,732,544-row completion manifest without modifying TSV bytes. Synthetic integration test passed. Official validator and upload package remain blocked on the full results.

## 2026-09-25T04:07:00+00:00 — Codex — India remainder acceleration prepared

- Cloud US `P5-SUB001-US-001` produced its first two of 64 verified shard pairs; first pair appeared about five minutes after launch including bootstrap and index build. Mac was healthy with at least India shards 0–4 complete. Its next three India shards remain assigned to Mac.
- Prepared independent cloud India shards 8–63, exactly 709,176 queries, with the same frozen input, model, char3 candidate routes, 51 features and threshold 0.83. The 100-query India 8–63 smoke run matched all 88 corresponding prior frozen shard TSVs byte-for-byte after decompression. No previously completed Mac shard is assigned to the cloud partition.
- Prepared SHA256 service-checksum downloader for both cloud partitions and a three-way assembler using Mac France/India 0–7, cloud India 8–63, cloud US 0–63. Synthetic four-query integration test verified query uniqueness, counts and partition provenance. The India worker has not launched at this entry; official validation and public score remain pending.

## 2026-09-25T04:07:36+00:00 — Codex — India remainder worker launched

- Fresh cost guard found estimated gross month-to-date $3.0928398446, active worst-case commitments $37.50 and proposed maximum $14.33632, total worst case $54.9291598446 below the $120 soft cap. Balance remains unverified.
- Launched `P5-SUB001-INDIA-001` on `i-0a29e371cccca1b99`, r8i.4xlarge on-demand, code commit `4a1872ddcf9549286df75d3009ca11fc30af3c9e`. It processes only India shards 8–63, uses SHA256 S3 checkpoints, and terminates automatically at completion or 12-hour OS cap. No already completed Mac shard is restarted.
- Three-way merge, official validation, ZIP packaging and public-score classification are pending both cloud partitions and Mac India shards 0–7. The Mac remains a fallback; Fold4 CLOSED.

## 2026-09-25T04:32:00+00:00 — Codex — EXP-031 completion, parity discrepancy, and verified disk recovery

- EXP-031 completed its 200,000-query retrieval and 256 route archives, exited 0, and its EC2 instance terminated. All archives and receipts were downloaded for the mandatory parity gate.
- Initial name-route comparison against frozen old20k P4-B-002 found 16 missing and 16 extra ID/rank rows among 2,000,000 pairs, while shared-pair scores differed by at most 2.384e-7. This is not a passing exact parity result; EXP-032 remains unlaunched pending pair-level diagnosis and a defensible resolution. Address route has not yet been assessed by the failing script.
- The generated `artifacts/audit.duckdb` (3,444,322,304 bytes) was uploaded to the private project S3 bucket. The S3 object was streamed back and independently SHA256-hashed; it matched local SHA256 `b2e9aa42b806243890c40d86f28b4af05d0a8f17e6d875fef23b4b78c8a3d700`, and object size, metadata, encryption and version ID were checked. The idle local derived database was then removed; originals were preserved and free disk rose to about 14 GiB. Receipt and restore command are at `artifacts/cloud/phase5/audit-backup-v001.json`.

## 2026-09-25T04:35:00+00:00 — Codex — EXP-031 numerical-tie audit passed

- Diagnosed every old20k route discrepancy. Name: zero changed candidate IDs, 16 ranks reordered among near-equal float32 scores at ranks67–84. Address: one rank100 candidate exchange for S1-428740130; both old/new targets are known negatives, plus 57 changed shared-pair ranks (ranks20–97). Maximum same-pair score delta is 2.384e-7; score span at changed ranks is below9e-8. No known-positive candidate was exchanged.
- Extended the parity verifier to keep `exact_id_rank_parity: false` visible and pass only this tightly bounded `NUMERIC_TIE_PASS` case after all 256 archive hashes and query counts verify; wider score/rank drift, a second boundary swap or a positive swap fails. Report: `outputs/analysis/P5-LEARNING-PARITY-001/report.json`. This does not retroactively turn the original strict failure into exact parity. The change and EXP-031 handoff risk are documented in DECISIONS.md. EXP-032 remains unlaunched pending fresh guard.
- Archiving the idle 3.44 GB audit database caused one existing full feature parity integration test to fail because its optional fixture was absent. The test now explicitly skips only when that archived fixture is absent and gives its restore path; its historical passing result remains recorded. Current suite: 85 passed, 1 skipped. This is a disk-reserve trade-off for final submission packaging, not a new feature parity result.

## 2026-09-25T04:42:00+00:00 — Codex — EXP-032 feature worker launched

- With EXP-031 exit0/terminated and all 256 archives verified under documented `NUMERIC_TIE_PASS`, launched EXP-032 / P5-FEATURE-200K-001 on r8i.2xlarge `i-08a2580f13957ad44`, immutable code commit `3381e311a4bff91cb750bf4166bf555c04ae440b`. Six-hour independent OS shutdown, automatic termination, SHA256 checkpointed feature parts and $4.50 ceiling. Outputs and scores remain pending; this job materializes features but does not train a model.
- Fresh cost guard estimated gross month-to-date plus all active worker maximum commitments and this proposed job at $55.4269198446 versus $120 soft cap. Actual credits unverified. Mac SUB-001 completed India shards0–7; France128 gzip files were SHA256-prefilled to the validator S3 prefix and India0–7 prefill began, with original shard bytes unchanged.

## 2026-09-25T04:45:13+00:00 — Codex — bounded upper India SUB-001 worker launched

- Fresh cost guard accepted P5-SUB001-INDIA-HIGH-001: projected worst-case project $69.93 below $120 soft cap, estimated new worker maximum $14.34 under $15 ceiling. Started on-demand r8i.4xlarge `i-01462cbfe56b611fb`, code commit `145d132`, exact frozen model/features/routes/threshold/input and only untouched India shards36–63. Automatic 12-hour shutdown/termination and per-shard S3 SHA256 checkpoints remain in force. This worker does not use labels or open Fold4.
- A separate local `aml-india-boundary` monitor observes lower India shard8–35 completion and upper shard36, verifies boundary object checksums and EC2 identities, then terminates only lower worker. It had not reached the boundary at 04:52 UTC.

## 2026-09-25T04:53:27+00:00 — Codex — split SUB-001 completion path prepared

- Implemented partial lower India and completed upper India download with service SHA256, exact frozen smoke parity (41/46 S1 respectively), explicit lower termination evidence, and immutable per-shard receipts. Extended country assembler to merge Mac France/India0–7, stopped lower India8–35, complete upper India36–63 and complete US; require exact frozen test S1 ID/country set, segment query totals and 384 shard files. Five focused tests passed. No frozen model, feature, candidate or threshold code changed. The full Mac run stays active as a fallback.
- Prefilled all Mac India0–7 and France shard files to final validator S3 prefix with checksum receipts. At this point US had 17/64 cloud shard pairs and lower India 5/28 required pairs. No final full-test TSV or official validator PASS exists.

## 2026-09-25T05:13:32+00:00 — Codex — disk reserve restored with verified remote archive

- Archive `artifacts/processed/prep-v001` and `artifacts/retrieval/P4-INDEX-PROBE-001` to private project S3 with AES256. Exact 2,327,511,818-byte archive was read back completely from S3 and matched local SHA256 `4a4f61752e0de0786c2992d3b877f9c5946d06047da6a5c4b690652d2e9db301`; local gzip/tar integrity listing found 54 entries. Receipt and restore command are in `artifacts/cloud/phase5/backups/prep-and-index-20260925.json`. Only then removed those reproducible local directories and the temporary tar, raising disk free to 17.56 GiB. Original competition TSVs, frozen SUB-001 bundle and live inference outputs remain untouched.
- Upper India cloud shard36 passed exact membership against all 12,693 frozen India shard36 S1 IDs, with all matches contained in scored candidates; shard37 candidate and matching output matched its prior frozen smoke row exactly. The SageMaker `ml.r5.2xlarge for processing job usage` service quota is now visibly 2. No full TSV or validator result exists yet.

## 2026-09-25T05:25:15+00:00 — Codex — EXP-032 completed and verified

- P5-FEATURE-200K-001 exited zero and EC2 `i-08a2580f13957ad44` terminated. Produced 39,440,694 owner-safe 51-feature candidate pairs for 200,000 fold1–3 S1; 854 S3 part objects matched the immutable receipt's exact key/size inventory, and summary/part receipt hashes plus sampled service SHA256/version IDs verified. Retrieval link recall 668,118/691,582 = 0.9660719915; complete positive-entity recall 0.9010811555; mean candidates/S1 197.20347. Feature computation 2,538.10 s; estimated EC2 compute $0.401096, final bill pending. No matcher F0.5 was measured by EXP-032. Fold4 CLOSED.

## 2026-09-25T05:27:58+00:00 — Codex — EXP-033 learning-curve worker launched

- Fresh cost guard passed: gross month-to-date estimate $3.09284 plus active worst-case worker commitments $63 and proposed new maximum $4.33408 totals $70.42692 below the $120 project soft cap. Started P5-LEARNING-FIT-001 on r8i.2xlarge `i-004e4dcc31fc9b130`, immutable code commit `fd56a5e`, six-hour OS shutdown/$4.50 planning ceiling, encrypted DeleteOnTermination EBS and per-model SHA256 checkpoints. Nine 20k/50k/100k fits on the same frozen 15k new S1 holdout are pending. No score claim before complete receipt and evaluation verification. Fold4 CLOSED.

## 2026-09-25T05:36:25+00:00 — Codex — EXP-033 full fixed-holdout curve verified

- P5-LEARNING-FIT-001 exited zero and EC2 terminated. Its 9 LightGBM model files and 9 fold-report JSON files all passed S3 service SHA256 and version-ID checks; the four summary files matched their immutable receipt. The pooled 15k macro entity F0.5 was 0.9349223520 at20k, 0.9363052750 at50k, 0.9371581365 at100k training S1. Each of three held folds improved from20k to100k. India 0.91210692→0.91438146; US 0.95005253→0.95226261. Precision and recall both improved; singleton F0.5 peaked at50k. Exact report: `artifacts/cloud/phase5/P5-LEARNING-FIT-001/results-metrics.json`; verification receipt: `artifacts/cloud/phase5/P5-LEARNING-FIT-001/verification.json`; plot: `docs/figures/learning_curve.png`.
- This is a fixed-threshold diagnostic on a new evaluation population; it is not directly comparable to the old 0.9318965293 OOF or any public leaderboard score. The rising curve triggers preparation of 250k/500k/1M owner-safe fits using the existing EXP-025 full retrieval, without launching duplicate retrieval. Fold4 remains closed. EXP-033 EC2 compute estimate $0.07818756, final bill pending.

## 2026-09-25T06:08:20+00:00 — Codex — urgent disjoint AWS SUB-001 acceleration

- User set a three-hour target for validated TSV and ZIP and forbade optional experiments. Live EC2 On-Demand standard quota is256 vCPUs; SageMaker ml.r5.2xlarge Processing quota is2. Latest eight-country-worker partition: India8–27 original lower,28–35 tail001,36–51 original upper,52–57 tail002,58–63 tail003; US0–47 original,48–55 tail001,56–63 tail002. France64 and India0–7 remain Mac-produced. New r8i.4xlarge on-demand workers launched from commits8912049/fe99571, each four-hour independent OS cutoff and $6.50 planning ceiling with SHA256-uploaded shards. Five new instance IDs are in their immutable ledger files. No model, feature, threshold, candidate route, or completed test shard changed.
- Detached boundary monitors replace the old India35 monitor and will terminate originals at verified shard27,51,47 after S3 SHA256 checkpoints. Mac fallback PID86583 was stopped after its complete France64 and India0–13 files were preserved; only France and India0–7 are selected for submission. Direct S3 checksum staging into the final validator prefix runs in detached screen `aml-stage-sub001`; it will emit a full384-object receipt only after all partition workers complete/terminate. Fresh last-worker cost guard: gross estimated$3.0928398446, active maximum commitments$89, new max$5.44544, project worst case$97.5382798446 below soft$120/hard$150. Actual bill pending.

## 2026-09-25T06:31:56+00:00 — Codex — portal matching-only override

- User reported the portal now accepts only `matching_results.tsv`. Stopped the waiting ZIP finalizer before any package was built; changed the required finalization path to SageMaker official default and strict validation, SHA256-verified matching-only download, and one TSV hash. Candidate pairs remain mandatory internal validator input. Existing AWS inference workers and S3 staging continue unchanged; no new fallback, duplicate or optional experiment was started.

## 2026-09-25T07:26:45+00:00 — Codex — validated matching-only SUB-001 ready

- SageMaker Processing job P5-SUB001-SM-VALIDATE-001 completed successfully. Both unchanged official validator modes passed; strict `--check-ids` had zero warnings. Final matching TSV has 1,732,544 data rows, 122,166 empty predictions, 1,610,378 non-empty; SHA256 `610478ebca0deb910a67d440edce299ab66ebc0ec55e285da8d0b7c20a24d6a3` matches the validator manifest and local full-file recomputation. Candidate file was consumed internally and passed official checks, SHA256 `89e95ad2f4cb0b0c4f082f9975c6549e8f073ceb71cbcb3ac1860247a590f4b6`. User will upload only `outputs/submissions/SUB-001/validated-v001/matching_results.tsv`; no ZIP was generated. Public leaderboard score/rank pending user upload.
