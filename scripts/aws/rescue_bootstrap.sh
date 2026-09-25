#!/usr/bin/env bash
# CL-012 test-fitted retrieval rescue worker. JOB_KIND = rescue:<route>:<analyzer>:<k>
set -Eeuo pipefail
exec > >(tee -a /var/log/aml-worker.log) 2>&1
START=$(date -u +%FT%TZ)
finish() {
  rc=$?; trap - EXIT; set +e; cd /opt/aml
  printf '{"exit_code":%s,"start":"%s","stop":"%s"}\n' "$rc" "$START" "$(date -u +%FT%TZ)" > worker-status.json
  mkdir -p upload-status; cp /var/log/aml-worker.log worker-status.json upload-status/
  python3 scripts/aws/upload_verified.py upload-status "$BUCKET" "$OUTPUT_PREFIX/status" /tmp/status-receipt.json
  shutdown -P now
}
shutdown -P +"$SHUTDOWN_MINUTES"
trap finish EXIT
mkdir -p /opt/aml; cd /opt/aml
export AWS_DEFAULT_REGION=us-east-1 AWS_MAX_ATTEMPTS=4
aws s3 cp "s3://$BUCKET/$CODE_KEY" code.tar --only-show-errors
printf '%s  code.tar\n' "$CODE_SHA256" | sha256sum -c -
tar xf code.tar
aws s3 cp "s3://$BUCKET/$INPUT_PREFIX/" inputs/ --recursive --only-show-errors
mkdir -p test
for s in test_source1 test_source2 test_source3; do aws s3 cp "s3://$BUCKET/amazon-ml-2026/raw/test/$s.tsv" test/$s.tsv --only-show-errors & done; wait
dnf install -y python3.12 python3.12-pip >/dev/null
python3.12 -m venv .venv
.venv/bin/pip install --disable-pip-version-check -q polars==1.44.2 duckdb==1.5.5 lightgbm==4.7.0 scikit-learn==1.9.1 sparse_dot_topn==1.2.0 rapidfuzz==3.14.6 numpy==2.5.3 scipy==1.18.1 anyascii==0.3.2
IFS=: read -r _ ROUTE ANALYZER K <<< "$JOB_KIND"
mkdir -p results
.venv/bin/python -u scripts/classical/country_rescue_aws.py --country "$COUNTRY" --route "$ROUTE" --analyzer "$ANALYZER" --k "$K" \
  --test-dir test --top12 "inputs/top12-$COUNTRY.parquet" --idf inputs/test_s1_idf.json --model inputs/stage2-200k-top12-textonly.txt \
  --out "results/$COUNTRY-$ROUTE-$ANALYZER-k$K.parquet"
python3 scripts/aws/upload_verified.py results "$BUCKET" "$OUTPUT_PREFIX/results" results-receipt.json
