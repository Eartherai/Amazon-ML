# AWS capacity — Phase5

Verified 2026-09-25 IST, us-east-1. Active user-selected profile `amamzon_01_a1_0`, account ending6318. `default` resolves to the SAME account, not extra credits. Other team accounts are not configured/verified. Initial named session expired; user refreshed it and identity/quota checks succeeded.

| Resource | Verified capacity |
|---|---:|
| Standard EC2 on-demand | 8 vCPUs |
| Standard EC2 Spot | 8 vCPUs |
| EC2 G/VT on-demand and Spot | 0 |
| EC2 P on-demand and Spot | 0 |
| Queried SageMaker G5/G6/G6e/P4/P5/M5/R5 training types | 0 |
| Existing EC2 instances at inventory | 0 |

Requests submitted: Standard on-demand64vCPU, G/VT8vCPU, SageMaker ml.g5.xlarge training1. All returned PENDING, not usable capacity. No claim of additional balance. User reports approximately$200; API credit balance remains unverified.

R8i.2xlarge and R7i.2xlarge:8vCPU/65536MiB; C7i.2xlarge:8vCPU/16384MiB. Current instance offerings were queried, but offerings do not guarantee capacity at launch. R8i benchmark selected for enough RAM to cache both source-partitioned sparse indexes. Large256GiB+ machines currently exceed standard quota.

Price List API returned Linux shared R8i.2xlarge $0.55568/hour. P5-INDEX-001 caps worker lifetime at90min with $2 planning ceiling including temporary100GiB gp3/publicIPv4 and requests; no exact bill or remaining-credit claim. Recheck actual billing separately.

Created dedicated `aml2026-phase5-worker` instance profile: read project Phase5 inputs/code, write/read project run outputs; no embedded credentials or broad admin grant. Security group has no ingress. Encrypted delete-on-termination EBS, IMDSv2, terminate-on-OS-shutdown. Raw S3 protections remain unchanged.

Evidence: artifacts/cloud/phase5/capacity-20260924T222612Z/inventory.json, quota-requests.json, r8i-price.json, infrastructure.json. Capacity inspection is read-only except separately recorded quota requests and worker-role/security-group creation.

References: [current EC2 families](https://docs.aws.amazon.com/ec2/latest/instancetypes/instance-types.html), [shutdown behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html). Detailed quota evidence includes all regional returned quotas, not assumed GPU availability.

## 2026-09-25 IST update

The active full-training retrieval occupies the account's entire on-demand Standard EC2 quota (8 vCPUs). Spot Standard quota is separately 8 vCPUs, but four independent SUB-001 Spot requests across R8i, R6i, R7a and R7i 2xlarge pools failed with EC2 insufficient-capacity errors before any new instance was created. Spot price history did not imply available capacity. The Mac full inference is active. A separate on-demand R8i validator-only job is prepared for the moment the retrieval worker finishes and quota is free. It has a 6-hour independent OS cutoff and $5 planning ceiling; it must not be launched until all 384 Mac shard files have checksum-verified uploads.

Fresh quota checks remain 0 for SageMaker ml.m5.2xlarge training, ml.m5.2xlarge/r5.2xlarge processing and ml.g5.xlarge training; requested increases are CASE_OPENED, not approved. GPU EC2 remains 0. Estimated Cost Explorer gross was $0.0099310264 at the prelaunch check, with billing lag. The $120 soft/$150 hard project caps and $25/$50/$75/$100/$125 budget alerts are active; credit balance is unverified.
