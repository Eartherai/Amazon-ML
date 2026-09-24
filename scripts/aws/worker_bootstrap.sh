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
shutdown -P +90
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
.venv/bin/python -u scripts/aws/benchmark_indexes.py --input inputs --output results --queries-per-country 200
python3 scripts/aws/upload_verified.py results "$BUCKET" "$OUTPUT_PREFIX/results" results-receipt.json
mkdir -p upload-receipts
cp results-receipt.json upload-receipts/
python3 scripts/aws/upload_verified.py upload-receipts "$BUCKET" "$OUTPUT_PREFIX/receipts" /tmp/receipts.json
