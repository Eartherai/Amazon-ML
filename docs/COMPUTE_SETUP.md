# Compute setup and handoff

Verified locally: Apple M5, arm64, 10 CPU cores, 24 GiB unified RAM, macOS 27.0. Initially ~24 GiB free disk. Local .venv uses Python 3.12 with pinned installed packages in code/business_entity_resolution/requirements.txt. No Torch/MPS benchmark performed yet.

## Colab

Installed official google-colab-cli **0.7.2** with `uv tool install google-colab-cli==0.7.2`; `colab version` and command help succeed. Version 0.7.1 referenced by online changelog was unavailable from PyPI; package index confirmed 0.7.2. Installation and authentication are verified. User completed sign-in; `colab usage` returned 0.00 compute units, 0.00/hr, 0 active assignments; `colab sessions` returned no active sessions. Remote hardware execution has not been tested.

To inspect the current balance in your terminal:

```sh
colab usage
```

Follow the Google authorization URL and paste the code back into that terminal, not into project files. The initial research prompt was cancelled; the user then completed a separate successful login. No cloud runtime is allocated by the usage check. Installed help says OAuth2 is default even though the bundled README says ADC; trust installed help and pinned version.

After account/budget checks, optional hardware smoke test:

```sh
colab run --gpu T4 --timeout 120 scripts/compute_smoke.py
colab sessions
```

This command would allocate a runtime and consume the account's allowance; it has NOT been run. `colab run` releases its runtime by default, but verify sessions after interruption. Do not pass `--keep` for unattended jobs. Explicitly retrieve checkpoints before shutdown for real jobs. A project archive/data transfer workflow must be added for a full training job; a lone script upload does not transfer the whole repository.

## AWS

AWS CLI 2.37.1 is installed. On 2026-09-25, after the user's completed setup, a read-only STS check succeeded in us-east-1. The previous expired-session check is historical. No instances were created and no credit balance was independently verified. The user reports approximately $200 on this configured account; other team accounts are not assumed available.

Authentication identity is recorded only in sanitized local metadata; credentials and complete identifiers are not logged. Quotas, eligible credit services and current prices must be checked only when a measured experiment justifies paid compute. No GPU benchmark is needed for this phase.

## Budget and schedule

Plan within the currently reported $200; additional credits count as $0 until confirmed. AWS_SPEND.md is the task ledger. Use a measured representative run before any paid full-scale job. Enforce per-job wall-time and cost ceilings; account budgets are alerts, not a guaranteed shutdown mechanism. Check persistent disks, snapshots, public IPs and object storage after stopping compute.

Use Kaggle only with legitimate per-person accounts and permitted quotas, private competition data, and verified current availability. No Kaggle installation/account access is needed for phase one; it is optional capacity, not a dependency of the plan.
