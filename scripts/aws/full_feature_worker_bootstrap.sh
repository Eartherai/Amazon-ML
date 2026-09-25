#!/usr/bin/env bash
# EXP-036 full unlocked classical feature worker; variables inserted by launcher.
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
aws s3 cp "s3://$BUCKET/$SHARD_PREFIX/results/" retrieval-summary/ --recursive --only-show-errors --exclude '*' --include '*.json'
aws s3 cp "s3://$BUCKET/$SHARD_PREFIX/shards/" retrieval-archives/ --recursive --only-show-errors
python3 - <<'VERIFY'
import hashlib,json,tarfile,shutil
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
complete=json.loads(Path('retrieval-summary/COMPLETE.json').read_text())
if complete['query_count']!=2206821 or complete['shards']!=64 or len(complete['routes'])!=4:
 raise ValueError('Incomplete retrieval summary')
expected={(field,ci,shard) for field in ('name','address') for ci in range(2) for shard in range(64)}
paths=list(Path('retrieval-archives').glob('*.tar'))
if len(paths)!=256:raise ValueError('Expected 256 route archives, got '+str(len(paths)))
Path('routes').mkdir()
seen=set()
for archive in sorted(paths):
 field,country,shard=archive.stem.split('-')
 key=(field,int(country[1:]),int(shard[1:]))
 if key not in expected or key in seen:raise ValueError('Unexpected route archive '+archive.name)
 seen.add(key)
 receipt=json.loads((Path('retrieval-summary')/(archive.stem+'-receipt.json')).read_text())
 if receipt['sha256']!=sha(archive) or receipt['bytes']!=archive.stat().st_size:
  raise ValueError('Route archive checksum mismatch '+archive.name)
 with tarfile.open(archive) as data:
  for member in data.getmembers():
   if not member.isfile() or '/' in member.name or not member.name.endswith('.parquet'):
    raise ValueError('Unsafe route archive member '+member.name)
   with data.extractfile(member) as source,(Path('routes')/member.name).open('xb') as dest:
    shutil.copyfileobj(source,dest)
if seen!=expected:raise ValueError('Missing route archives')
print(json.dumps({'verified_route_archives':len(seen),'query_count':complete['query_count']}),flush=True)
VERIFY
dnf install -y python3.12 python3.12-pip libgomp
python3.12 -m venv .venv
.venv/bin/pip install --disable-pip-version-check -r configs/aws/requirements-index.txt
.venv/bin/pip install --disable-pip-version-check -r configs/aws/requirements-sub001.txt
export PYTHONPATH=code/business_entity_resolution OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/pip freeze > environment.txt
.venv/bin/python -u scripts/aws/build_sample_features.py --routes routes --inputs inputs --output results --shards 64 --chunk-rows 200000 --expected-queries 1765649 --folds 0,1,2,3 --restrict-to-queries --experiment-id EXP-036 --upload-bucket "$BUCKET" --upload-prefix "$OUTPUT_PREFIX/results"
mkdir -p summary
cp results/metrics.json results/COMPLETE.json results/parts-receipt.json environment.txt summary/
python3 scripts/aws/upload_verified.py summary "$BUCKET" "$OUTPUT_PREFIX/results" results-receipt.json
mkdir -p upload-receipts
cp results-receipt.json upload-receipts/
python3 scripts/aws/upload_verified.py upload-receipts "$BUCKET" "$OUTPUT_PREFIX/receipts" /tmp/receipts.json
