#!/usr/bin/env bash
# Filled by launcher; no static credentials. Root boot script, finite on-demand worker.
set -Eeuo pipefail
exec > >(tee -a /var/log/aml-worker.log) 2>&1
START=$(date -u +%FT%TZ)
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /opt/aml
  printf '{"exit_code":%s,"start":"%s","stop":"%s"}\n' "$rc" "$START" "$(date -u +%FT%TZ)" > worker-status.json
  mkdir -p upload-status
  cp /var/log/aml-worker.log worker-status.json upload-status/
  if test -f scripts/aws/upload_verified.py; then
    python3 scripts/aws/upload_verified.py upload-status "$BUCKET" "$OUTPUT_PREFIX/status" /tmp/status-receipt.json
  fi
  shutdown -P now
}
# Independent OS watchdog covers failed bootstrap/job/upload; launch uses terminate-on-shutdown.
shutdown -P +"$SHUTDOWN_MINUTES"
trap finish EXIT
mkdir -p /opt/aml
cd /opt/aml
export AWS_DEFAULT_REGION=us-east-1 AWS_MAX_ATTEMPTS=2
aws s3 cp "s3://$BUCKET/$CODE_KEY" code.tar --only-show-errors
printf '%s  code.tar\n' "$CODE_SHA256" | sha256sum -c -
tar xf code.tar
aws s3 cp "s3://$BUCKET/$INPUT_PREFIX/" inputs/ --recursive --only-show-errors
python3 - <<'PY'
import hashlib,json
from pathlib import Path
for row in json.loads(Path('inputs/manifest.json').read_text())['files']:
 p=Path('inputs')/row['name']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert h.hexdigest()==row['sha256'] and p.stat().st_size==row['bytes'],row['name']
PY
dnf install -y python3.12 python3.12-pip libgomp
python3.12 -m venv .venv
.venv/bin/pip install --disable-pip-version-check -r configs/aws/requirements-index.txt
export PYTHONPATH=code/business_entity_resolution OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
.venv/bin/pip freeze > environment.txt
if [[ "$JOB_KIND" == sub001_inference || "$JOB_KIND" == sub001_validation || "$JOB_KIND" == sub001_country ]]; then
  if [[ "$JOB_KIND" == sub001_inference || "$JOB_KIND" == sub001_country ]]; then
    .venv/bin/pip install --disable-pip-version-check -r configs/aws/requirements-sub001.txt
    if [[ "$JOB_KIND" == sub001_country ]]; then
      test -n "$COUNTRY"
      .venv/bin/python -u scripts/submissions/run_sub002.py --inputs inputs --output results --threads 8 --country "$COUNTRY" --first-shard "$FIRST_SHARD" --last-shard "$LAST_SHARD" --upload-bucket "$BUCKET" --upload-prefix "$OUTPUT_PREFIX"
    else
      .venv/bin/python -u scripts/submissions/run_sub002.py --inputs inputs --output results --threads 8 --upload-bucket "$BUCKET" --upload-prefix "$OUTPUT_PREFIX"
    fi
  else
    mkdir -p results/shards
    aws s3 cp "s3://$BUCKET/$SHARD_PREFIX/" results/shards/ --recursive --only-show-errors
    .venv/bin/python - <<'PY'
import gzip
from pathlib import Path
files=list(Path('results/shards').glob('*.tsv.gz'))
if not files or len(files)%2:raise RuntimeError('Incomplete shard pair inventory')
for p in files:
 with gzip.open(p,'rb') as f:
  while f.read(8*1024*1024):pass
PY
    printf '{"source":"Mac SUB-001 full-test inference","shard_count":%s}\n' "$(find results/shards -name '*-matching.tsv.gz' | wc -l)" > results/COMPLETE.json
  fi
  if [[ "$JOB_KIND" == sub001_country ]]; then
    mkdir -p summary
    cp results/COMPLETE.json environment.txt summary/
    python3 scripts/aws/upload_verified.py summary "$BUCKET" "$OUTPUT_PREFIX/results" results-receipt.json
    mkdir -p upload-receipts
    cp results-receipt.json upload-receipts/
    python3 scripts/aws/upload_verified.py upload-receipts "$BUCKET" "$OUTPUT_PREFIX/receipts" /tmp/receipts.json
    exit 0
  fi
  mkdir -p student_resource/dataset/test
  for source in test_source1 test_source2 test_source3; do
    aws s3 cp "s3://$BUCKET/amazon-ml-2026/raw/test/${source}.tsv" "student_resource/dataset/test/${source}.tsv" --only-show-errors
  done
  .venv/bin/python -u scripts/submissions/merge_validate.py --shards results/shards --output output --test-dir student_resource/dataset/test
  mkdir -p summary
  gzip -c output/matching_results.tsv > summary/matching_results.tsv.gz
  gzip -c output/candidate_pairs.tsv > summary/candidate_pairs.tsv.gz
  cp output/validation.json output/official.log output/official_check_ids.log results/COMPLETE.json environment.txt summary/
  python3 scripts/aws/upload_verified.py summary "$BUCKET" "$OUTPUT_PREFIX/results" results-receipt.json
elif [[ "$JOB_KIND" == full_retrieval || "$JOB_KIND" == sample_retrieval ]]; then
  aws s3 cp "s3://$BUCKET/$INDEX_PREFIX/results/" indexes/ --recursive --only-show-errors
  aws s3 cp "s3://$BUCKET/$INDEX_PREFIX/receipts/results-receipt.json" index-receipt.json --only-show-errors
  python3 - <<'VERIFY'
import hashlib,json
from pathlib import Path
for row in json.loads(Path('index-receipt.json').read_text()):
 p=Path('indexes')/row['key'].split('/results/',1)[1]
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert p.stat().st_size==row['bytes'] and h.hexdigest()==row['sha256'],str(p)
VERIFY
  .venv/bin/python -u scripts/aws/retrieve_cached.py --indexes indexes --idf inputs --queries inputs/queries.parquet --output results --upload-bucket "$BUCKET" --upload-prefix "$OUTPUT_PREFIX/shards"
  if [[ "$JOB_KIND" == full_retrieval ]]; then
    aws s3 cp "s3://$BUCKET/amazon-ml-2026/phase5/inputs/retrieval-eval-v001/" evaluation-labels/ --recursive --only-show-errors
    python3 - <<'VERIFY_LABELS'
import hashlib,json
from pathlib import Path
for row in json.loads(Path('evaluation-labels/manifest.json').read_text())['files']:
 p=Path('evaluation-labels')/row['name']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['name']
VERIFY_LABELS
    .venv/bin/python -u scripts/aws/evaluate_retrieval.py --candidates results --labels evaluation-labels --output evaluation
  fi
  mkdir -p summary
  if [[ "$JOB_KIND" == full_retrieval ]]; then
    cp evaluation/metrics.json summary/retrieval-metrics-unlocked.json
  fi
  cp results/*.json results/query_coverage.parquet environment.txt summary/
  python3 scripts/aws/upload_verified.py summary "$BUCKET" "$OUTPUT_PREFIX/results" results-receipt.json
else
  .venv/bin/python -u scripts/aws/benchmark_indexes.py --input inputs --output results --queries-per-country 200
  cp environment.txt results/
  python3 scripts/aws/upload_verified.py results "$BUCKET" "$OUTPUT_PREFIX/results" results-receipt.json
fi
mkdir -p upload-receipts
cp results-receipt.json upload-receipts/
python3 scripts/aws/upload_verified.py upload-receipts "$BUCKET" "$OUTPUT_PREFIX/receipts" /tmp/receipts.json
