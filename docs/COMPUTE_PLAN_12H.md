# Compute plan — 12-hour first-principles sprint (started 2026-09-26 00:00 IST)

## Constraints discovered (live, not assumed)

- Account ending 6318, region us-east-1. Estimated gross spend before this sprint ~$80 (Cost Explorer lags ~24 h; only Sep 24 posted). Project planning cap $150, so ~$70 is available; every job below has a hard runtime cap.
- EC2: 256 on-demand standard vCPU; **0 on-demand G/P** (GPU only via SageMaker).
- SageMaker training GPU quotas (1 instance each): ml.g5.2xlarge, ml.g5.12xlarge, ml.g6.2xlarge, ml.g6.4xlarge, ml.g6e.xlarge, ml.g6e.12xlarge, ml.g4dn.* (T4, no bf16). No GPU quota in us-west-2 / us-east-2 / eu-west-1 / ap-south-1.
- **us-east-1 GPU capacity is scarce**: at 00:13 IST all five queued jobs showed "waiting for capacity"; queuing the same job on several instance types and stopping the losers once one starts works (pending jobs are not billed).
- IAM is least-privilege and is NOT modified in this sprint: the SageMaker neural role reads `raw/test/*` and `phase5/exp050-e5-large-v001/*`, writes `phase5/exp050-e5-large-v001/output/*`; the EC2 worker role reads `phase5/inputs/*`, `phase5/code/*`, `raw/test/*`, writes `phase5/runs/*`. Data is moved between these prefixes with **S3 server-side copy** from the operator account (no upload through the slow Mac uplink).
- Mac: 10 cores / 24 GB / slow uplink. Orchestration, small joins and validation only.
- Earlier EC2 rescue workers hung because the bootstrap's bare `wait` also waited on the `tee` process substitution; any new bootstrap must `wait` on explicit PIDs and upload a heartbeat.

## Workload placement

| Workload | Where | Instance | Cap |
|---|---|---|---|
| Cross-encoder e5-base (train folds 1-2, score fold 3 + 10.56M test band pairs) | SageMaker | ml.g5.12xlarge (1 GPU used) | 3 h |
| Cross-encoder mdeberta-v3-base (fold-3 comparison) | SageMaker | ml.g5.2xlarge | 2 h |
| Dense bi-encoder e5-small retrieval rescue (fold-3 miss recovery, then test) | SageMaker | ml.g6.4xlarge | 3 h |
| Stacking / ownership / decision models | Mac (cached matrices <1 GB) | — | — |
| Unseen-country simulation | Mac (cached matrices) | — | — |
| Any new >5 min / >8 GB job | EC2 r8i.* via launcher, heartbeat required | — | — |

## Staged data (server-side copies)

`phase5/exp050-e5-large-v001/ce-v001/`: `train/` (raw train TSVs + ground truth), `top12/` (200k held-fold OOF top-12 with base scores and folds), `test12/` (192 test top-12 score shards), `code-<job>/`.
