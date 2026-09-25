#!/usr/bin/env bash
# Two-file official validation of final submission TSVs against frozen SUB-002 candidate shards + additions.
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
  python3 scripts/aws/upload_verified.py upload-status "$BUCKET" "$OUTPUT_PREFIX/status" /tmp/status-receipt.json
  shutdown -P now
}
shutdown -P +"$SHUTDOWN_MINUTES"
trap finish EXIT
mkdir -p /opt/aml
cd /opt/aml
export AWS_DEFAULT_REGION=us-east-1 AWS_MAX_ATTEMPTS=4
aws s3 cp "s3://$BUCKET/$CODE_KEY" code.tar --only-show-errors
printf '%s  code.tar\n' "$CODE_SHA256" | sha256sum -c -
tar xf code.tar
aws s3 cp "s3://$BUCKET/$INPUT_PREFIX/" inputs/ --recursive --only-show-errors
python3 - <<'PY'
import hashlib,json
from pathlib import Path
for row in json.loads(Path('inputs/manifest.json').read_text())['files']:
 p=Path('inputs')/row['name'];h=hashlib.sha256(p.read_bytes()).hexdigest()
 assert h==row['sha256'] and p.stat().st_size==row['bytes'],row['name']
PY
mkdir -p shards test
for run in P5-SUB002-FR-001 P5-SUB002-IN0-001 P5-SUB002-IN1-001 P5-SUB002-IN2-001 P5-SUB002-IN3-001 P5-SUB002-US0-001 P5-SUB002-US1-001 P5-SUB002-US2-001; do
  aws s3 cp "s3://$BUCKET/amazon-ml-2026/phase5/runs/$run/shards/" shards/ --recursive --exclude "*" --include "*-candidates.tsv.gz" --only-show-errors
done
for s in test_source1 test_source2 test_source3; do aws s3 cp "s3://$BUCKET/amazon-ml-2026/raw/test/$s.tsv" test/$s.tsv --only-show-errors; done
python3 scripts/submissions/validate_two_file.py --shards shards --test-dir test --submissions inputs/submissions --validator student_resource/utils/validate_submission.py --strict scripts/submissions/strict_check.py --output results
python3 scripts/aws/upload_verified.py results "$BUCKET" "$OUTPUT_PREFIX/results" results-receipt.json
mkdir -p upload-receipts && cp results-receipt.json upload-receipts/
python3 scripts/aws/upload_verified.py upload-receipts "$BUCKET" "$OUTPUT_PREFIX/receipts" /tmp/receipts.json
