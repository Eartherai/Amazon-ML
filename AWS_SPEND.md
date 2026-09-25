# Compute ledger

Initial task spend was $0. EC2 workers and S3 storage have since been launched; current gross billing and credit balance must be read from the dated entries below.

| Date IST | Owner/profile | Instance | Experiment | Start | Stop | Hours | Estimated cost | Actual cost | Artifacts |
|---|---|---|---|---|---|---|---|---|---|

Current planning ceiling: approximately $200 reported by the user on the configured AWS account. Additional team accounts and future awards are optional, unverified capacity. Do not record credentials here.

2026-09-25 01:33 IST: read-only STS identity check succeeded in us-east-1 after user login. Zero resources launched; task spend remains $0. Credit balance was not queried or independently verified. Sanitized record: artifacts/aws_phase2_preflight.json.

## Phase 3 S3 storage plan (before upload)

Private S3 Standard in us-east-1, ≤10 GiB planned, ≤1,000 PUT-class requests. Verified storage rate $0.023/GiB-month and PUT $0.005/1,000 requests from AWS Price List API ([pricing](https://aws.amazon.com/s3/pricing/)). Script-calculated ceiling: $0.22999999999999998/month storage + $0.005 PUTs; raw-only storage $0.0540/month. No compute instances or GPU. Actual bill/credit balance unverified. Record completion/bytes after uploads.

### Phase 3 storage upload complete

2026-09-24T20:53:30.255143+00:00: raw 8 objects, 2,520,577,195 bytes. 2026-09-24T20:55:29.622341+00:00: processed 43 objects, 1,806,340,084 bytes. Every object verified with service SHA256, length, metadata SHA256 and version ID. Combined 4,326,917,279 bytes; calculated storage run-rate $0.0927/month before optional transfer/logging. Estimated successful PUT charges $0.000255, excluding setup/HEAD/retry requests. Actual billed cost and credits unknown. Compute spend $0; no EC2/GPU launched. Storage is now accruing charges, so total project spend should not be described as exactly zero.

## 2026-09-24T21:57:24.552608+00:00 — Phase4 checkpoint

Completed5k model/candidates/features/OOF/calibration/ablations/country-transfer backed up under existing private bucket checkpoints/phase4/A-001/. Two checksum-verified objects; 205,762,384bytes. Additional S3 storage estimate $0.004408/month plus approximately$0.000010 PUT cost; no paid compute, actual billed spend and credit balance not queried. Source commit ae5cfcb; manifest and receipts in artifacts/cloud/phase4/checkpoint-A-001.

## 2026-09-24T22:36:34.808653+00:00 — P5-INDEX-001 / EXP-024
Account ending6318, profile amamzon_01_a1_0, us-east-1. Launched i-01871abb0c9bb0e07 at2026-09-24T22:34:27Z, R8i.2xlarge on-demand. Stop/runtime pending. Compute quote$0.55568/hour;90min ceiling compute$0.83352, total planning ceiling$2 including temporary EBS/IP/requests. Actual job cost unknown. Inputs/results at S3 amazon-ml-2026/phase5/. Monthly budget API reports$100budget and$0.01actual before run; this is not a credit balance. User-reported$200remains unverified.

### 2026-09-24T22:47:14.680824+00:00 — benchmark completion
P5-INDEX-001 exit0; stop status22:43:07Z, termination observed22:45:43Z. Job wall521.283586seconds, compute estimate$0.080463 through job completion, excludes shutdown lag.184verified result/status/receipt objects6,078,184,306bytes. Exact AWS bill pending. Additional best20k backup2objects591,802,307bytes.

## 2026-09-24T22:52:42.142603+00:00 — P5-FULL-RETRIEVAL-001 / EXP-025 active
Profile amamzon_01_a1_0, account ending6318, us-east-1, R8i.2xlarge on-demand, instancei-0609d158c38e96160, start2026-09-24T22:47:45.616413+00:00. Full2,206,821S1retrieval + unlocked label evaluation. Query-time estimate17.76hours/$9.87compute; hard OS runtime cap24hours, total planning allowance$18 (not a guaranteed billing cap). Automatic terminate after verified uploads or failure. Stop/runtime/actualcost pending. S3prefixamazon-ml-2026/phase5/runs/P5-FULL-RETRIEVAL-001. Remaining credit unverified.

## 2026-09-24T23:22:32.013520+00:00 — SUB-001 cost controls and capacity
Estimated Cost Explorer gross month-to-date at prelaunch: $0.0099310264 (billing lag; not final). Active full-retrieval worker maximum planned commitment $18. Four attempted SUB-001 Spot launches were rejected for insufficient capacity before creation and did not start billable instances. Proposed SUB-001 job had 36h OS cutoff, max Spot price $0.30/h, $1 storage/transfer allowance, $11.80 computed maximum and $12 planning ceiling. Alerts at gross $25/$50/$75/$100/$125. Soft cap $120; hard planning cap $150; target reserve at least $50 from user-reported ~$200, balance not verified.

## 2026-09-25T01:25:00+00:00 — SageMaker validation contingency, no job yet
P5-SUB001-SM-VALIDATE-001 is prepared only. `ml.r5.2xlarge` Processing official price $0.605/hour, 6h maximum $3.63 compute plus $1 noncompute allowance = $4.63, under $5 ceiling. Dedicated role and validated API request exist; no billable processing job has launched. Launch guard rechecks gross billing, all active EC2 project ledgers, active processing jobs and completed 384-shard SHA256 receipt. Managed job stops on completion or at MaxRuntimeInSeconds=21600; no endpoint or persistent compute.

## 2026-09-25T02:36:00+00:00 — EXP-031 / P5-LEARNING-200K-001 prelaunch
The verified 200,000-query input includes the frozen 20,000 development entities plus deterministic fold1–3 extensions; no labels or Fold4 entities are present. Full-pool lexical index is reused from P5-INDEX-001. Projected query time is 1.61h by scaling the full-retrieval benchmark, excluding bootstrap and transfers; this is an estimate, not a measured run. Proposed r8i.2xlarge on-demand worker has an independent 6h OS shutdown and terminate-on-shutdown EBS cleanup. At $0.55568/h, maximum planned compute is $3.33408 plus $1.00 storage/network/request allowance, $4.33408 under the $4.50 experiment ceiling. Each of 256 country/route/query shards will be SHA256-verified in S3. No instance launched at this checkpoint; actual cost and credit balance unverified. Live quota checks showed 256 on-demand standard vCPUs, SageMaker ml.r5.2xlarge Processing quota 2, Training quota 1, and no active Processing job.

## 2026-09-25T03:06:19+00:00 — EXP-031 launched; EXP-032/033 prepared

P5-LEARNING-200K-001 launched at 02:39:53Z on r8i.2xlarge `i-014e9cdbb267db46d`, with six-hour independent OS shutdown and automatic EC2 termination. At 03:04:05Z it was running and had 71 output S3 objects totaling 74,096,640 bytes. EXP-025 full retrieval also remained running with 76 objects totaling 890,306,560 bytes at 03:04:41Z. The cloud code/input and shard uploads use SHA256 verification. Neither job has a final cost yet. Latest Cost Explorer gross month-to-date snapshot before EXP-031 was about $3.09 with billing lag; do not interpret that as credit balance. The two active jobs reserve $18 + $4.33408 planning exposure. The validator's separate $5 ceiling remains unspent. EXP-032 and EXP-033 are prepared but unlaunched, each with a six-hour hard stop and $4.50 planning ceiling; they will be launched sequentially after predecessor completion and fresh cost-guard checks. User-reported credit remains unverified.

## 2026-09-25T03:55:10+00:00 — SUB-001 US-only acceleration launched

Profile `amamzon_01_a1_0`, account ending 6318, us-east-1, on-demand r8i.4xlarge `i-01b623904660bdf50`. Input is the SHA256-verified frozen SUB-001 bundle; code commit `7dd0efa`; per-US-shard SHA256 S3 checkpoints under `amazon-ml-2026/phase5/runs/P5-SUB001-US-001/`. Twelve-hour OS shutdown, terminate-on-shutdown EC2 and encrypted 100 GiB DeleteOnTermination EBS. Planned hourly rate $1.11136, worst-case 12-hour compute $13.33632 plus $1 noncompute allowance, total $14.33632 under the $15 experiment ceiling. Fresh prelaunch gross month-to-date estimate $3.0928398446, existing active-worker commitments $22.50, projected worst-case project total $39.9291598446 versus $120 soft cap; billing estimate and credit balance are unverified. Stop time and actual cost pending.
