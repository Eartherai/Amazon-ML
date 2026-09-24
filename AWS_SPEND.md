# Compute ledger

Task spend: $0. No AWS/Colab/Kaggle runtimes launched. Existing team charges and balances are unknown.

| Date IST | Owner/profile | Instance | Experiment | Start | Stop | Hours | Estimated cost | Actual cost | Artifacts |
|---|---|---|---|---|---|---|---|---|---|

Current planning ceiling: approximately $200 reported by the user on the configured AWS account. Additional team accounts and future awards are optional, unverified capacity. Do not record credentials here.

2026-09-25 01:33 IST: read-only STS identity check succeeded in us-east-1 after user login. Zero resources launched; task spend remains $0. Credit balance was not queried or independently verified. Sanitized record: artifacts/aws_phase2_preflight.json.

## Phase 3 S3 storage plan (before upload)

Private S3 Standard in us-east-1, ≤10 GiB planned, ≤1,000 PUT-class requests. Verified storage rate $0.023/GiB-month and PUT $0.005/1,000 requests from AWS Price List API ([pricing](https://aws.amazon.com/s3/pricing/)). Script-calculated ceiling: $0.22999999999999998/month storage + $0.005 PUTs; raw-only storage $0.0540/month. No compute instances or GPU. Actual bill/credit balance unverified. Record completion/bytes after uploads.

### Phase 3 storage upload complete

2026-09-24T20:53:30.255143+00:00: raw 8 objects, 2,520,577,195 bytes. 2026-09-24T20:55:29.622341+00:00: processed 43 objects, 1,806,340,084 bytes. Every object verified with service SHA256, length, metadata SHA256 and version ID. Combined 4,326,917,279 bytes; calculated storage run-rate $0.0927/month before optional transfer/logging. Estimated successful PUT charges $0.000255, excluding setup/HEAD/retry requests. Actual billed cost and credits unknown. Compute spend $0; no EC2/GPU launched. Storage is now accruing charges, so total project spend should not be described as exactly zero.
