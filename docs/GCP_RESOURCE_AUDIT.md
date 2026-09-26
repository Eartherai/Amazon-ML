# Google Cloud resource audit (2026-09-26, read-only; nothing created, enabled or spent)

## Identity and project

- gcloud: `/opt/homebrew/share/google-cloud-sdk/bin/gcloud` (SDK 586.0.0; not on the agent shell PATH)
- Authenticated account: `ramkhamir6@gmail.com` (the only credentialed account)
- Active project: `asymmetric-ray-477810-a5` ("My First Project", number 734123949353, ACTIVE)
- Billing: linked to `billingAccounts/0169EC-10EA84-C24F9E` ("My Billing Account"), open, billingEnabled=true
- Enabled APIs relevant to compute: `aiplatform` (Vertex AI), `compute`, `storage`, `iam`, `logging`, `monitoring`, `notebooks`. Also enabled: `maps-backend`, `generativelanguage`, BigQuery APIs. **Maps and any business/location lookup must never be used with challenge data.** Cloud Quotas, Cloud Billing Catalog and Artifact Registry APIs are not enabled (not needed; nothing was enabled).

## GPU quota (live)

Compute Engine:
- **GPUS_ALL_REGIONS = 0** (project-wide cap). No Compute Engine GPU VM can start in any region until this is raised. Per-region quotas of 1 (T4, L4, V100, P100, P4, K80, spot variants) exist in us-central1, us-east1, us-east4, us-west1, us-west4, europe-west4, asia-south1, asia-southeast1, but the global cap blocks them.
- CPUS_ALL_REGIONS = 32 (100-200 per region).

Vertex AI custom training (the usable path):
- On-demand training GPU quota: **0** for every accelerator type.
- **Spot (preemptible) training quota: A100 x8** in us-central1, europe-west4 and asia-southeast1; T4 x1 (8 regions), V100 x1 (us-central1, us-west1, europe-west4), P100 x1, P4 x1.

Vertex AI serving (endpoints / batch prediction): on-demand L4 (2 in us-central1, 1 elsewhere), RTX PRO 6000 x2 (us-central1, us-east1, europe-west4, asia-southeast1), T4 x1, V100 x1, plus spot L4/T4/V100. Endpoints are persistent and are not used; batch prediction is possible but more cumbersome than custom jobs.

Quota figures are limits, not guaranteed capacity; spot A100 availability must be confirmed by a tiny job.

## Credit applicability

| credit | nominal | scope evidence | usable for our GPU work |
|---|---:|---|---|
| Trial credit for GenAI App Builder | INR 88,202.50 (exp 2026-11-12) | Name and community reports tie it to Vertex AI Agent Builder / Search / Conversation ("certain usage"); no official statement that it covers custom training, Compute Engine or GPUs | **NO (treated as unusable until the console credit detail lists Vertex AI Training SKUs)** |
| Google Developer Program premium credits | ~INR 3,795 | developers.google.com/profile/help/benefits: may be spent on any Google Cloud Platform (or Maps Platform) product; Vertex AI named; no GPU exclusion listed | **YES (inference from official wording; no SKU list)** |
| Free Trial | expired | | no |

Estimated usable ML credit: **~INR 3,795 (about USD 45)**. The definitive check for either credit is Cloud Console > Billing > Credits > (credit) > eligible services. That is not exposed by gcloud or the Billing API. A first tiny job followed by the billing report (24 h lag) also shows which credit absorbed it.

## Recommended compute path

Vertex AI Custom Job, `scheduling.strategy=SPOT`, 1x A100 (a2-highgpu-1g) in us-central1, prebuilt PyTorch GPU training container, our own code only. Data comes in from the existing S3 artifacts through short-lived presigned URLs fetched inside the job (the Mac uplink is ~90 KB/s, so bulk uploads are impractical). Outputs go to a GCS bucket in the same project and are pulled to the Mac. Every job gets an experiment ID, a max runtime (`timeout`), checkpoint and resumable outputs, and no endpoint.

Approximate list prices (not verified via the Billing Catalog API; confirm with the first tiny job): Vertex training a2-highgpu-1g about USD 4.4/h on demand, spot typically 60-70% lower (about USD 1.3-1.8/h); n1 + T4 spot about USD 0.3/h. About USD 45 of credit is therefore roughly 25-35 spot A100-hours.

Best use: resumable GPU scoring and training jobs that AWS quota or budget cannot absorb, such as wider France dense retrieval and CE rescoring, and held-fold bi-encoders for folds 1/2 (multi-fold confirmation of the dense rescue).

## Commands used (all read-only)

```
gcloud auth list; gcloud config list; gcloud projects list
gcloud billing projects describe asymmetric-ray-477810-a5; gcloud billing accounts list
gcloud services list --enabled --project asymmetric-ray-477810-a5
gcloud compute project-info describe --project asymmetric-ray-477810-a5 --format=json   (GPUS_ALL_REGIONS)
gcloud compute regions describe <region> --project asymmetric-ray-477810-a5 --format=json
GET https://serviceusage.googleapis.com/v1beta1/projects/asymmetric-ray-477810-a5/services/aiplatform.googleapis.com/consumerQuotaMetrics  (Vertex AI quotas; bearer token from gcloud auth print-access-token)
```

`gcloud alpha services quota list` was not used: it needs the alpha component installed, and the SDK installation was not modified.
