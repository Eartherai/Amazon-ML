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

`colab version` reports 0.7.2; `colab usage` and `colab sessions` verified user login, 0 compute units and no active sessions. Run these again before any cloud experiment. AWS authentication is expired; see docs/COMPUTE_SETUP.md. Account balances/quota/price remain unverified. No paid launch commands have been executed.

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

Full label metrics intentionally excludeFold4; evaluator rejects locked labels. `results/retrieval-metrics-unlocked.json` will hold exact unlocked-fold metrics after completion. Full candidate coverage counts use allS1without labels. Fullquery outputs do not constitute OOF model predictions or final submission files.
