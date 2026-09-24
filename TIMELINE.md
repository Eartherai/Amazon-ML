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
