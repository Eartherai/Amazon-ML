#!/usr/bin/env bash
# EXP-038 entity-level set-decision OOF evaluation worker.
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
shutdown -P +"$SHUTDOWN_MINUTES"
trap finish EXIT
mkdir -p /opt/aml
cd /opt/aml
export AWS_DEFAULT_REGION=us-east-1 AWS_MAX_ATTEMPTS=2
aws s3 cp "s3://$BUCKET/$CODE_KEY" code.tar --only-show-errors
printf '%s  code.tar\n' "$CODE_SHA256" | sha256sum -c -
tar xf code.tar
aws s3 cp "s3://$BUCKET/$INPUT_PREFIX/" inputs/ --recursive --only-show-errors
aws s3 cp "s3://$BUCKET/$SHARD_PREFIX/results/" features/ --recursive --only-show-errors
python3 - <<'VERIFY'
import hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

manifest=json.loads(Path('inputs/manifest.json').read_text())
for row in manifest['files']:
 p=Path('inputs')/row['name']
 if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Input checksum mismatch: '+row['name'])
complete=json.loads(Path('features/COMPLETE.json').read_text())
if complete['queries']!=200000 or complete['fold4']!='CLOSED' or complete['feature_parts']<100:
 raise ValueError('Incomplete feature store')
parts=json.loads(Path('features/parts-receipt.json').read_text())
files=list(Path('features').glob('features-*.parquet'))
if len(parts)!=complete['feature_parts'] or len(files)!=complete['feature_parts']:
 raise ValueError('Feature part inventory mismatch')
for row in parts:
 p=Path('features')/row['key'].split('/results/',1)[1]
 if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Feature checksum mismatch: '+p.name)
print(json.dumps({'verified_feature_parts':len(files),'queries':complete['queries']}),flush=True)
VERIFY
dnf install -y python3.12 python3.12-pip libgomp
python3.12 -m venv .venv
.venv/bin/pip install --disable-pip-version-check -r configs/aws/requirements-index.txt
.venv/bin/pip install --disable-pip-version-check -r configs/aws/requirements-sub001.txt
export PYTHONPATH=code/business_entity_resolution OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
.venv/bin/pip freeze > environment.txt
mkdir -p models
aws s3 cp "s3://$BUCKET/amazon-ml-2026/phase5/runs/P5-LEARNING-FIT-001/results/checkpoint_receipts.json" models/checkpoint_receipts.json --only-show-errors
for held in 1 2 3; do
  aws s3 cp "s3://$BUCKET/amazon-ml-2026/phase5/runs/P5-LEARNING-FIT-001/results/model-size100000-held${held}.txt" "models/model-size100000-held${held}.txt" --only-show-errors
done
python3 - <<'VERIFY_MODELS'
import hashlib,json
from pathlib import Path
receipts={row['key'].split('/')[-1]:row for row in json.loads(Path('models/checkpoint_receipts.json').read_text())}
for held in (1,2,3):
 name=f'model-size100000-held{held}.txt';path=Path('models')/name
 if hashlib.sha256(path.read_bytes()).hexdigest()!=receipts[name]['sha256'] or path.stat().st_size!=receipts[name]['bytes']:
  raise ValueError('Model checksum mismatch: '+name)
VERIFY_MODELS
.venv/bin/python -u scripts/aws/evaluate_set_decision_200k.py --features features --inputs inputs --models models --output results --threads 8
mkdir -p summary
cp results/metrics.json environment.txt summary/
python3 scripts/aws/upload_verified.py summary "$BUCKET" "$OUTPUT_PREFIX/results" results-receipt.json
mkdir -p upload-receipts
cp results-receipt.json upload-receipts/
python3 scripts/aws/upload_verified.py upload-receipts "$BUCKET" "$OUTPUT_PREFIX/receipts" /tmp/receipts.json
