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
