#!/bin/bash
# Build a full submission from a stack parquet whose France rows differ: compact assembly (same args as CL-055 v5det-a),
# India/US robust name-collision override (as SUB014), upload packaging with official + strict validation. Usage: TAG STACK UPLOADNAME
set -euo pipefail; cd "$(dirname "$0")/../.."; A="$(pwd)"; TAG=$1; STACK=$2; UP=$3
mkdir -p outputs/experiments/CL-066
.venv/bin/python scripts/classical/assemble_compact.py --name $TAG --stack $STACK --dense outputs/experiments/CL-051/dense-test-all-scored-frk50.parquet --fr-route "$A/outputs/experiments/CL-012/France-rescue.parquet" --fr-add "$A/outputs/experiments/CL-012/France-rescue-VSAFE.parquet" --vsafe "$A/outputs/submissions/UPLOAD_FINAL_VSAFE/UPLOAD_FINAL_VSAFE_matching_results.tsv" --france new --cand-base 0.02 --cand-text 0.2 --thr 0.72 --add-thr 0.8 --dense-ce outputs/experiments/CL-051/cedense_test-frk50.parquet --dense-bundle outputs/experiments/CL-051/dense_test-frk50.parquet --dense-model-thr 0.72 --dump-cands outputs/experiments/CL-066/test_cands_$TAG.parquet > outputs/experiments/CL-066/assemble-$TAG.log 2>&1
.venv/bin/python scripts/classical/apply_override_submission.py --cands outputs/experiments/CL-066/test_cands_$TAG.parquet --base-dir outputs/submissions/$TAG --override outputs/experiments/CL-055/univ_apply_robust/override_test_indus_only.parquet --name $TAG-univ > outputs/experiments/CL-066/override-$TAG.log 2>&1
bash scripts/submissions/make_upload.sh outputs/submissions/$TAG-univ/matching_results.tsv $UP outputs/submissions/$TAG-univ/candidate_pairs.tsv > outputs/experiments/CL-066/upload-$UP.log 2>&1
cat outputs/submissions/UPLOAD_$UP/receipt.json
