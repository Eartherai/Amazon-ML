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
