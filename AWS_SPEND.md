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

## 2026-09-25T04:07:00+00:00 — India remainder partition, planned before launch

`P5-SUB001-INDIA-001` proposes on-demand r8i.4xlarge for only deterministic India shards 8–63: 709,176 of 809,986 India S1 queries. Mac is assigned shards 0–7 and remains the fallback. Its 100-query smoke parity has zero TSV content differences against the frozen prior run for all 88 applicable files. A 12-hour OS shutdown at planning rate $1.11136/hour caps compute at $13.33632; plus $1 storage/network/request allowance gives $14.33632, within the $15 experiment ceiling. Per-shard SHA256 S3 checkpoints, encrypted DeleteOnTermination EBS and automatic EC2 termination are configured. Expected benefit is several hours earlier first complete validated submission, with no score change; final runtime and cost are unmeasured. A fresh cost guard is required immediately before launch; user-reported credit balance remains unverified.

### 2026-09-25T04:07:36+00:00 — launch

Launched on EC2 `i-0a29e371cccca1b99` under profile `amamzon_01_a1_0`, account ending 6318, code commit `4a1872ddcf9549286df75d3009ca11fc30af3c9e`. Prelaunch gross bill estimate $3.0928398446, active worst-case commitments $37.50, proposed maximum $14.33632 and projected project worst case $54.9291598446 below the $120 soft cap. Credit balance unverified. Stop, measured runtime and actual billed cost pending.

## 2026-09-25T04:32:00+00:00 — audit artifact disk backup

The generated 3,444,322,304-byte `artifacts/audit.duckdb` was stored in the same private S3 project bucket under `amazon-ml-2026/phase5/backups/audit-20260925.duckdb`, SSE-AES256, then streamed back and SHA256-verified before removing the idle local copy. S3 Standard storage at the previously verified $0.023/GiB-month rate is approximately $0.074/month while retained; actual request and data-transfer charges are not yet billed. This is a reversible disk reserve action, not paid compute. The backup version, hash and restore command are in `artifacts/cloud/phase5/audit-backup-v001.json`.

## 2026-09-25T04:42:00+00:00 — EXP-032 feature materialization launched

P5-FEATURE-200K-001, on-demand r8i.2xlarge `i-08a2580f13957ad44`, account ending6318, immutable code commit `3381e311a4bff91cb750bf4166bf555c04ae440b`, six-hour OS shutdown/terminate-on-shutdown and encrypted DeleteOnTermination EBS. Estimated maximum compute $3.33408 plus $1 noncompute allowance, total $4.33408 under its $4.50 ceiling. Fresh gross month-to-date billing estimate plus active worst-case commitments and this job was $55.4269198446 under the $120 soft cap. EXP-031 compute estimate $1.0194973778438667, final bill and credit balance pending. EXP-032 stop/runtime/actual cost pending.

## 2026-09-25T04:44:00+00:00 — India upper-half acceleration plan, before launch

Observed first cloud India shards at about 7–8 minutes each made its 56-shard serial path the first-submission bottleneck. `P5-SUB001-INDIA-HIGH-001` proposes only untouched India shards36–63 (354,947 S1); current `P5-SUB001-INDIA-001` supplies shards8–35 (354,229 S1) and Mac supplies0–7 (100,810 S1). Once shard35 has both SHA256-uploaded files, terminate the lower EC2 before it uploads shard36, preserving completed lower outputs. Upper worker has a 12-hour OS cap, on-demand r8i.4xlarge planning rate $1.11136/h, $13.33632 maximum compute plus $1 allowance, $14.33632 under a $15 ceiling, encrypted DeleteOnTermination EBS and per-shard S3 checkpointing. The expected benefit is about three hours earlier first submission, subject to measured pace. A fresh cost guard is mandatory; credit balance is unverified.

## 2026-09-25T04:45:13+00:00 — India upper-half worker launched

P5-SUB001-INDIA-HIGH-001 was launched on on-demand r8i.4xlarge `i-01462cbfe56b611fb`, account ending6318, code commit `145d132`, 12-hour OS shutdown/terminate-on-shutdown and encrypted DeleteOnTermination EBS. Its fresh cost guard reported estimated gross month-to-date $3.0928398446 plus all active maximum commitments $52.50 and this job maximum $14.33632, project worst case $69.9291598446 under the $120 soft cap. Credit balance and final billed cost remain unverified. Lower India is scheduled for intentional termination at verified shard35 boundary. No duplicate completed shard was started by upper worker.

## 2026-09-25T05:13:32+00:00 — local disk archive storage

The 2,327,511,818-byte reproducible processed-data/index archive was uploaded to encrypted S3 and fully read back once for SHA256 verification before local cleanup. S3 storage and transfer/request costs are expected to be small but have not yet appeared in billing; receipt and restore command are `artifacts/cloud/phase5/backups/prep-and-index-20260925.json`. No EC2 or SageMaker job was started for this operation.

## 2026-09-25T05:27:58+00:00 — EXP-032 completion and EXP-033 launch

EXP-032 / P5-FEATURE-200K-001 exited zero and terminated at 05:25:15 UTC, 2,598.52 s wall, estimated EC2 compute $0.401096; actual billed amount pending. EXP-033 / P5-LEARNING-FIT-001 launched on on-demand r8i.2xlarge `i-004e4dcc31fc9b130`, account ending6318, code commit `fd56a5e`, six-hour independent OS cutoff and automatic terminate-on-shutdown. Maximum compute $3.33408 plus $1 allowance = $4.33408 under its $4.50 planning ceiling. Fresh guard's project worst case was $70.42692 below the $120 soft cap. Credit balance and final bill unverified.

## 2026-09-25T05:36:25+00:00 — EXP-033 completion

P5-LEARNING-FIT-001 exited zero and terminated after 506.54 seconds wall time; 427.73 seconds were inside the training script. EC2 compute estimate $0.07818756 at the planning rate, versus $4.50 ceiling. Final billed compute, storage and request costs remain unverified. All 18 model/fold-report checkpoints and summary receipts passed SHA256/version verification. No further paid job was launched for the 250k/500k/1M plan yet; the already-running EXP-025 full retrieval will supply its routes.

## 2026-09-25T06:08:20+00:00 — urgent SUB-001 tail workers

Five additional on-demand r8i.4xlarge workers were launched on legitimate account ending6318, region us-east-1. Runs/instance IDs: P5-SUB001-INDIA-TAIL-001/i-015dc5abe06575e2c; INDIA-TAIL-002/i-07fd2f957d20d488b; INDIA-TAIL-003/i-074cfb03b27241c48; US-TAIL-001/i-0ebaea8a0b4a878df; US-TAIL-002/i-047ca1fafa186eb50. Each is capped at240min, planning compute$1.11136/hour, max compute$4.44544 plus$1 allowance ($5.44544) under its$6.50 job ceiling, with automatic terminate-on-shutdown and encrypted delete-on-termination EBS. Code/input hashes, exact ranges, dates and S3 checkpoints are in `artifacts/cloud/phase5/<run>/ledger.json`. Latest fresh guard across active workers projects$97.5382798446 total including gross billing estimate versus$120 soft cap. Stop times and actual costs remain pending; monitor and terminate immediately at completion.

## 2026-09-25T09:30:14.423504+00:00 — P5-SUB002-FIT-001 / SUB-002 active
Profile `amamzon_01_a1_0`, account ending 6318, us-east-1, r8i.2xlarge on-demand `i-045185b5404fe3b7c`; two-hour OS cap and $2.50 planning ceiling. Reuses verified 200k/39.44m-pair EXP-032 store. EXP-033 fixed 15k OOF scale slope +0.00223578 from 20k to 100k supports final fit. Automatic terminate after verified result or failure. Exact cost pending. Existing full-retrieval worker remains active. Eight SUB-002 inference workers are prepared but not launched; each has disjoint shards and separate cost guard.

## 2026-09-25T09:43:47.179540+00:00 — SUB-002 inference and EXP-034 active
Eight disjoint on-demand r8i.4xlarge test workers `P5-SUB002-FR-001`, `IN0–IN3`, `US0–US2`, each four-hour OS stop/$5.50 planning ceiling, plus `P5-CONFLICT-OOF-001` r8i.2xlarge 90-minute/$2 planning ceiling. Last fresh cost guard: gross billed estimate $5.19235 (billing lag), active maximum commitments $62, proposed EXP-034 $1.83352, project worst case $69.02587, beneath user sprint $100 and existing $120/$150 guards. Training worker P5-SUB002-FIT-001 terminated exit0, estimated EC2 compute $0.0424; exact bill and credit balance pending. No duplicate retrieval or inference shards.

## 2026-09-25T10:01:00+00:00 — EXP-034 and EXP-035 completed

Both on-demand r8i.2xlarge jobs exited zero and terminated under 90-minute/$2 caps. EXP-034 140.85 seconds wall and $0.02174 estimated EC2 compute; EXP-035 141.86 seconds and $0.02190. Final billed cost, storage, and S3 requests remain unverified. Eight SUB-002 inference workers continue under their separate four-hour/$5.50 caps.

## 2026-09-25T10:08:00.955739+00:00 — EXP-038 set-decision OOF launched

On-demand r8i.2xlarge i-09c529fa680766637, 90-minute OS shutdown and automatic termination, planned $1.83352 maximum including $1 noncompute allowance. Fresh cost guard gross estimate $5.19235 plus $62 active maximum commitments plus new maximum = $69.02587, below the user sprint $100 cap. Only AWS profile amamzon_01_a1_0; final billed cost and credit balance pending.

EXP-038 exited zero at 2026-09-25T10:10:23Z; estimated compute $0.02392 at planning rate, final bill unverified. Termination state was shutting-down at last collection.

EXP-039 first summary-only failure exited1 and terminated after 264.83 seconds, estimated EC2 compute $0.04088. Recovery P5-HARDNEG-ENSEMBLE-002 exited0/terminated with an estimated compute cost recorded in its ledger; both used independent 90-minute/$2 caps. Final bill pending.

## 2026-09-25T10:25:32.685992+00:00 — EXP-044 current-model OOF extraction

On-demand r8i.2xlarge i-099f8bb45c6dc80ef, independent 90-minute shutdown/terminate-on-shutdown, maximum $1.83352 including noncompute allowance. Fresh cost guard gross $5.19235 plus active max $62 plus job max = $69.02587, under user $100 cap. Only profile amamzon_01_a1_0. Final bill pending.

## 2026-09-25T12:25:00+00:00 — bounded concurrent validation and neural training

Only authenticated profile `amamzon_01_a1_0` was used. Two SUB-002 `ml.r5.2xlarge` official validation jobs (`P5-SUB002-SM-VALIDATE-001`, `-002`) are in progress under their separate six-hour/$5 caps; their CloudWatch logs already show the base merge official default and strict PASS. India neural G5 job `P5-SUB002-NGPU-85c999e8-INDI-001` continues under its two-hour/$7 cap while the France+US job has completed. EXP-050 18k-query E5 inputs (420,347 owner-safe pairs; ten S3 objects with SHA256 receipts) were staged; its three-hour/$10 G5 job has **not** launched while the India G5 occupies capacity. Actual billed cost and credits remain unverified. Do not sum simultaneous maximum caps as actual spend.
