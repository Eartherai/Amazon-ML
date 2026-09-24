# S3 data layer: Phase 3

The user requested private S3 storage for original records and a reproducible multi-view feature foundation. This is storage preparation, not an EC2/GPU experiment.

## Security and immutability

- Dedicated project bucket in us-east-1; exact name recorded in `artifacts/cloud/phase3/s3_state.json`.
- All four Block Public Access controls enabled; bucket-owner-enforced ownership disables ACLs.
- AES256 SSE-S3 at rest; bucket policy denies non-TLS access.
- Versioning enabled; raw prefix denies deletion and object creation without conditional headers. Uploader uses `If-None-Match: *` for every data/manifest object.
- Every upload supplies SHA256 for S3 verification; HEAD verifies length, stored metadata SHA256, service checksum and version ID. Resume skips only a verified identical object. This does not trust multipart ETag as a hash.
- This is operational immutability against pipeline mistakes; an administrator can change policies. No irreversible compliance retention or Object Lock was enabled.
- No bucket grant to another team account or public principal. Future worker roles should be restricted to the exact project prefix. AWS CLI uses the existing authenticated session; code never loads/logs credential files.

## Layout

`amazon-ml-2026/raw/{train,test}/`: exact supplied TSV bytes, including train labels.

`amazon-ml-2026/processed/prep-v001/`: source directories of ZSTD Parquet shards and immutable manifests. Source/country strings remain data; files are not country-partitioned.

`amazon-ml-2026/manifests/raw-v001/MANIFEST_RAW.json`: original file sizes, SHA256, schema, audited row counts and modification metadata.

Future artifacts use versioned `retrieval/`, `indices/`, `models/`, `runs/`, `predictions/`, `submissions/`, `logs/` prefixes. S3 prefixes need no empty placeholder objects.

Only explicitly enumerated TSV, Parquet and JSON files are uploaded. No .git, environment, credentials, local cache dump or original ZIP duplication.

## Cost and observability

Verified AWS Price List API rates: Standard storage $0.023/GiB-month in us-east-1 and PUT-class requests $0.005/1,000. Raw input bytes are 2,520,573,701; script-calculated raw-only storage is about $0.054/month. Planned ≤10GiB storage plus ≤1,000 PUT requests is ≤$0.235 for a full month, before any future outbound transfer or optional logging. No compute instances. Storage persists until deliberately removed; actual bill/credit balance is not verified. [S3 pricing](https://aws.amazon.com/s3/pricing/).

Daily S3 storage metrics plus upload receipts/manifests provide the initial operational record. For broader team/cloud operation, enable encrypted server-access logging, appropriately scoped CloudTrail S3 data events and request metrics/alarms with an explicit cost budget; these additional billable services have not been provisioned in this small storage setup. Incomplete multipart uploads are aborted after one day. No expiration rule deletes competition data automatically.

## Reproduction

```sh
.venv/bin/python scripts/cloud/s3_data_layer.py raw
.venv/bin/python scripts/cloud/s3_data_layer.py processed
```

State and per-object receipts allow safe retries. Do not edit the raw manifest in place or reuse a processed version for changed transforms. Upload status lives in `artifacts/cloud/phase3/{raw,processed}_upload.json` only after completion; a running log alone is not proof the upload completed.

Primary references: [S3 security](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html), [conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html), [policy enforcement](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes-enforce.html).
