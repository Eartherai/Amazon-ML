# Runbook

Use the workspace .venv. All dataset reads use an explicit tab separator and preserve empty text. Commands below are updated as implementations become available.

```sh
source .venv/bin/activate
export PYTHONPATH="$PWD/code/business_entity_resolution"
python -m pytest code/business_entity_resolution/tests -q
```

Original supplied validator (when a real prediction set is available):

```sh
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

Never use original all-pairs Python dataframes. Avoid running concurrent heavy jobs on the 24 GiB Mac. Do not delete user files to free disk.

## Reproduce core research in one command

Use a new run directory (the command refuses to overwrite). Needs at least 18 GiB free before starting. This runs audit, labeled-pair diagnostics, fold creation, per-file memory measurements and the no-training exact baseline. It does **not** train a learned model or produce a final submission. Component commands were run and verified individually; this wrapper has not been rerun end-to-end to avoid duplicate artifacts.

```sh
PYTHONPATH=code/business_entity_resolution .venv/bin/python -m src.run_research \
  --data-dir student_resource/dataset \
  --run-dir artifacts/research/RUN-002 \
  --baseline-config configs/experiments/EXP-001.json
```

## Implemented commands from workspace root

```sh
export PYTHONPATH=code/business_entity_resolution
.venv/bin/python -m src.audit_data --data-dir student_resource/dataset --output-dir outputs/audit/NEW-AUDIT --database artifacts/new-audit.duckdb
.venv/bin/python -m src.audit_pairs --database artifacts/new-audit.duckdb --output-dir outputs/audit/NEW-PAIRS
.venv/bin/python -m src.build_validation --database artifacts/new-audit.duckdb --output-dir artifacts/validation/new-v1
.venv/bin/python -m src.audit_memory --data-dir student_resource/dataset --database artifacts/new-audit.duckdb --output outputs/audit/new-memory.json
.venv/bin/python -m src.exact_baseline --database artifacts/new-audit.duckdb --config configs/experiments/EXP-001.json --output-dir outputs/experiments/NEW-EXACT
```

Each fresh database is tied to the input file hashes in its corresponding audit JSON. Do not reuse a database for changed inputs. Audit attempts may resume only against the exact same files; source hashes must agree. The final baseline records the current Git commit; run from the repository after committing source changes.

Optional macOS-only generic transliteration diagnostic:

```sh
swiftc scripts/transliterate_probe.swift -o artifacts/transliterate_probe
.venv/bin/python -m src.audit_supplement --database artifacts/new-audit.duckdb --output outputs/audit/new-supplement.json --transliterator artifacts/transliterate_probe
```

Authoritative results already present: AUDIT-004, PAIR-002, supplement-001, memory-001, validation/v1, EXP-001. Compact JSON evidence is versioned under docs/audit_evidence/. Human-readable report is DATA_AUDIT.md. `scripts/render_audit_report.py` regenerates the report including the memory/suffix supplement.

## Cloud setup status

`colab version` reports 0.7.2; `colab usage` and `colab sessions` verified user login, 0 compute units and no active sessions. The AWS profile `amamzon_01_a1_0` has refreshed credentials and accesses account ending 6318. Cost Explorer reports gross billed estimates, but credit balance remains unverified. The capped full-training retrieval EC2 worker `P5-FULL-RETRIEVAL-001` is active; see `AWS_SPEND.md`. SageMaker `ml.r5.2xlarge` Processing quota is 2; no SageMaker job has launched.

## Phase 2 reproduction

Use fresh output directories for new audit/experiment runs. Canonical evidence: PROFILE-001, PREP-001, MORPH-002, TOKEN-001/run-002.

```sh
PYTHONPATH=code/business_entity_resolution .venv/bin/python -m src.analysis.profile_dataset --output-dir artifacts/data_profile/PROFILE-002 --markdown docs/DATA_PROFILE_002.md
PYTHONPATH=code/business_entity_resolution .venv/bin/python -m src.analysis.pair_morphology --database artifacts/audit.duckdb --output-dir artifacts/pair_analysis/MORPH-003
PYTHONPATH=code/business_entity_resolution .venv/bin/python -m src.analysis.transliteration_diagnostics --pairs artifacts/pair_analysis/MORPH-003 --binary artifacts/transliterate_probe
PYTHONPATH=code/business_entity_resolution .venv/bin/python -m src.blocking.token_candidates --database artifacts/audit.duckdb --config configs/blocking/TOKEN-001.json --output-dir outputs/candidates/TOKEN-001/run-003
.venv/bin/python scripts/render_preprocessing_report.py
PYTHONPATH=code/business_entity_resolution .venv/bin/python -m pytest code/business_entity_resolution/tests -q
```

The report renderer rebuilds presentation files from canonical PROFILE-001/MORPH-002/TOKEN-001 evidence. It does not rerun experimental computations. Swift transliteration binary is Mac-specific; verify cross-platform parity before moving preprocessing to Linux.

## Phase4 commands

- Freeze/samples (new outputs only): `.venv/bin/python scripts/phase4_prepare.py`
- Retrieval: `PYTHONPATH=code/business_entity_resolution OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m src.blocking.phase4_retrieval --sample A --output outputs/candidates/P4-A-001`
- Nested OOF: `PYTHONPATH=code/business_entity_resolution OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/run_with_openmp.py -m src.models.phase4_oof --sample A --routes outputs/candidates/P4-A-001 --output outputs/oof/P4-A-001`
- Error analysis: `.venv/bin/python scripts/phase4_error_analysis.py --run outputs/oof/P4-A-001`

Output directories refuse overwrite. Use new versioned IDs for reruns. Candidate config validates query hash and fit-owner folds. Phase4 retrieval uses full target search but disjoint fixed IDF fitting; see docs/OOF_VALIDATION.md. Do not reuse Phase3 IDF for OOF1–3.

## Phase5 cloud execution

Active profile: `amamzon_01_a1_0`. Benchmark command: `scripts/aws/launch_cpu_worker.sh`. Full run: `scripts/aws/launch_cpu_worker.sh --config configs/aws/P5-FULL-RETRIEVAL-001.json`. Requires committed source, uploaded input receipts and successful terminated benchmark. Each run uses a new immutable directory and S3 prefix.

Collect status: `AWS_PROFILE=amamzon_01_a1_0 .venv/bin/python scripts/aws/collect_run.py RUN_ID`. Emergency terminate only tagged project workers: `scripts/aws/terminate_workers.sh`. Full-run shards are immutable `.tar` files under its S3 `shards/` prefix; every upload is SHA256verified before proceeding. Work is bounded by24h scheduled OS shutdown, terminate behavior, encrypted delete-on-terminationEBS. Do not launch duplicate full jobs. See cloud ledger for exact instance ID and commit.

Poll AWS read-only status sequentially. Multiple simultaneous CLI invocations against the login profile briefly received HTTP 429 `CreateOAuth2Token Rate exceeded` on 2026-09-25; this was an authentication-service throttle, not evidence of a worker failure. Back off and retry one call after a short interval rather than issuing concurrent polls or relaunching a job.

Full label metrics intentionally excludeFold4; evaluator rejects locked labels. `results/retrieval-metrics-unlocked.json` will hold exact unlocked-fold metrics after completion. Full candidate coverage counts use allS1without labels. Fullquery outputs do not constitute OOF model predictions or final submission files.

The classical 200k learning-size sequence is EXP-031 retrieval -> EXP-032 feature store -> EXP-033 fixed-holdout LightGBM. EXP-031 (`P5-LEARNING-200K-001`) is already running; do not launch another retrieval. After `collect_run.py P5-LEARNING-200K-001` shows exit 0, terminated, `results-COMPLETE.json` reports 200,000 S1 and 256 S3 route archives are present, the committed launcher has an explicit predecessor gate for EXP-032:

First verify every archived route hash and exact old-20k candidate ID/rank parity. The final EXP-031 `results/` prefix contains `COMPLETE.json` and one receipt per route archive. Downloading these own-project objects is read-only; the verifier keeps the historical P4 routes untouched:

```sh
AWS_SDK_UA_APP_ID=AWSSkill-SageMaker AWS_PROFILE=amamzon_01_a1_0 aws s3 cp \
  s3://aml2026-ber-08be19ac500747/amazon-ml-2026/phase5/runs/P5-LEARNING-200K-001/results/ \
  artifacts/cloud/phase5/P5-LEARNING-200K-001/parity-summary/ \
  --recursive --exclude '*' --include '*.json'
AWS_SDK_UA_APP_ID=AWSSkill-SageMaker AWS_PROFILE=amamzon_01_a1_0 aws s3 cp \
  s3://aml2026-ber-08be19ac500747/amazon-ml-2026/phase5/runs/P5-LEARNING-200K-001/shards/ \
  artifacts/cloud/phase5/P5-LEARNING-200K-001/parity-archives/ --recursive
.venv/bin/python scripts/aws/verify_sample_retrieval_parity.py \
  --summary artifacts/cloud/phase5/P5-LEARNING-200K-001/parity-summary \
  --archives artifacts/cloud/phase5/P5-LEARNING-200K-001/parity-archives \
  --output outputs/analysis/P5-LEARNING-PARITY-001/report.json
```

Require `parity: PASS`, both 2-million-pair old-sample routes with zero ID/rank mismatches, every archive SHA256 verified, and a score difference at most 1e-5. Then use the launch command below; the EXP-032 worker repeats the archive checksum check on its own downloaded copies.

```sh
AWS_SDK_UA_APP_ID=AWSSkill-SageMaker AWS_PROFILE=amamzon_01_a1_0 .venv/bin/python scripts/aws/launch_cpu_worker.py --config configs/aws/P5-FEATURE-200K-001.json
```

EXP-032 consumes SHA256-verified `feature-200k-v001` inputs (200k fold1–3 labels, 10,320,219 full targets and owner folds), verifies all retrieval archive hashes, and uploads every feature part with SHA256. After `collect_run.py P5-FEATURE-200K-001` shows exit 0, terminated, complete part inventory and `results-COMPLETE.json`, launch EXP-033 with the analogous `--config configs/aws/P5-LEARNING-FIT-001.json`. Its worker rechecks every feature part receipt, fits 20k/50k/100k entities per outer fold against a fixed 15k new OOF set and uploads each model/metric checkpoint. Both workers have six-hour OS cutoffs, $4.50 planning ceilings and fresh cost guards; do not report scores before completion. Fold4 remains CLOSED. On this Mac, fresh LightGBM imports may require `DYLD_LIBRARY_PATH=/Users/earther/Library/Python/3.9/lib/python/site-packages/torch/lib`; the running SUB-001 process is unaffected.

After EXP-033 exits zero and its metrics are collected, fill `docs/LEARNING_CURVE.md` from the verified `results-metrics.json`, then generate the overall/India plot with `scripts/analysis/plot_learning_curve.py --metrics artifacts/cloud/phase5/P5-LEARNING-FIT-001/results-metrics.json --output docs/figures/learning_curve.png`. The plot command refuses an existing output; use a versioned path for reruns. Compare the same 15k entities at all sizes and note that the three thresholds came from the earlier 20k procedure.

## Frozen early-calibration SUB-001

To shorten the critical path without changing predictions, `configs/aws/P5-SUB001-US-001.json` defines a 12-hour/$15-cap on-demand US-only worker. `run_sub001.py --country US` processes the same frozen US queries, targets, char3 retrieval, features, model and threshold; default invocation and the detached Mac process are unchanged. Its 100-query US smoke outputs matched the prior frozen smoke output exactly in all 98 decompressed TSV files. The worker uploads each US shard with SHA256. Keep the Mac running for France and India. After both India and cloud US are complete, verify every shard and build a new complete 192-pair manifest from Mac France/India plus cloud US; do not treat a country-only COMPLETE.json as full inference. Then follow the existing 384-shard upload, SageMaker validation and package procedure below.

After `collect_run.py P5-SUB001-US-001` shows exit 0 and termination, run `scripts/submissions/download_sub001_country.py --run-id P5-SUB001-US-001 --output outputs/submissions/SUB-001/cloud-us-v001 --smoke outputs/submissions/SUB-001/smoke-100percountry/shards`. It verifies the S3 SHA256 checksum and byte length for the full US completion record and all 128 shard files, then requires exact candidate and matching output for every frozen 100-query US smoke ID and writes local per-shard receipts. Once `local-full-v005/inference/progress.json` lists complete France and India country progress, run `scripts/submissions/assemble_sub001_countries.py --mac outputs/submissions/SUB-001/local-full-v005/inference --us outputs/submissions/SUB-001/cloud-us-v001 --output outputs/submissions/SUB-001/hybrid-full-v001/inference`. The assembler rechecks all US SHA256 receipts, checks 1,732,544 unique S1 rows and hard-links the 384 gzip files. Use the hybrid directory with `upload_local_shards.py` if it finishes first. The Mac may continue US as an independent fallback; do not overwrite either result.

An additional India worker `P5-SUB001-INDIA-001` processes only deterministic India shards 8–63 (709,176 queries). The first eight India shards remain with the Mac, so no already-completed shard is restarted. Its `--first-shard 8 --last-shard 64` 100-query smoke run matched all 88 applicable prior frozen shard TSVs exactly after decompression. After it exits zero and terminates, use `scripts/submissions/download_sub001_country.py --run-id P5-SUB001-INDIA-001 --output outputs/submissions/SUB-001/cloud-india-v001 --smoke outputs/submissions/SUB-001/smoke-100percountry/shards` and the same downloader for US as above. The India download must match all 87 smoke IDs in its shard range. Once Mac India shard 7 is complete, run the assembler above with `--india outputs/submissions/SUB-001/cloud-india-v001` and a new output directory `outputs/submissions/SUB-001/hybrid-threeway-v001/inference`. It verifies all 384 files and exact country totals. The original Mac full run remains a fallback and is not overwritten.

User explicitly authorized a single early portal submission after the complete test output passes the unchanged official validator, without waiting for full training retrieval or Fold4. The frozen model is NUMERIC-V2 51-feature LightGBM, threshold 0.83, model SHA256 `d84957742e05f5cd790d7dfc8c14ca05d3b5a2dc941a5094b8874d623b35117b`; all final name/address char3 top100 union candidates are scored. The live detached Mac screen is `aml-sub001-v005`, Python PID86583, with output `outputs/submissions/SUB-001/local-full-v005/inference`. Do not restart completed shards or alter the model, thresholds, candidate routes or input bundle.

Check `progress.json`, `screen -ls`, process state and `df -h .` before action. On successful completion require `COMPLETE.json` with 1,732,544 processed queries and exactly 192 matching plus 192 candidate gzip shard files. Upload with:

```sh
AWS_PROFILE=amamzon_01_a1_0 .venv/bin/python scripts/submissions/upload_local_shards.py \
  --inference outputs/submissions/SUB-001/local-full-v005/inference \
  --bucket aml2026-ber-08be19ac500747 \
  --prefix amazon-ml-2026/phase5/sub001-mac-shards-v001 \
  --receipt artifacts/cloud/phase5/sub001-mac-shards-upload.json
```

The uploader resumes by SHA256 verification and writes `.complete.json` only after all 384 objects pass. The 64 GiB SageMaker validator is prepared as a contingency for the official `--check-ids` pass, which can require several GiB of Python sets. It has a 6-hour hard stop and $5 planning ceiling. After the upload receipt exists, inspect the dry-run plan with `AWS_SDK_UA_APP_ID=AWSSkill-SageMaker AWS_PROFILE=amamzon_01_a1_0 .venv/bin/python scripts/aws/launch_sagemaker_validator.py`; then launch with the same command plus `--launch` only if budget and role checks pass. It creates no endpoint or persistent compute. The separate EC2 validator is an alternative after the full-retrieval worker frees EC2 quota.

Download the completed validator result, verify its manifest and full TSV SHA256s, and inspect `official.log` and `official_check_ids.log`. Both official runs must exit 0 and say PASS; the `--check-ids` run must have zero warnings. A `READY.json` receipt must exist before packaging. Build the official ZIP with:

```sh
.venv/bin/python scripts/submissions/build_final_package.py \
  --validated-dir outputs/submissions/SUB-001/validated-v001 \
  --output outputs/submissions/SUB-001/amazon_ml_2026_submission.zip
```

The packager requires the verified matching and candidate hashes, includes both TSVs, runnable `code/business_entity_resolution/` with frozen artifacts and filled `Documentation_template.md`, then CRC-checks the ZIP and writes a SHA256 sidecar. Record matching/candidate/ZIP hashes, experiment and Git commit in `SUBMISSIONS.csv`. **The user will upload both files themselves:** provide the validated `matching_results.tsv` and final ZIP paths. Do not upload to the portal. Record observed public score and rank only if the user later provides them. See `docs/COMPETITION_RULES.md`.
