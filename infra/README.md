# Phase5 finite EC2 benchmark

Run from repository root. AWS CLI authentication uses user-selected profile `amamzon_01_a1_0`; no credentials are stored in source or user data.

1. scripts/aws/check_capacity.sh
2. .venv/bin/python scripts/aws/check_costs.py
3. .venv/bin/python scripts/aws/prepare_index_inputs.py (new immutable input directory only)
4. Upload with scripts/aws/upload_verified.py; preserve receipt.
5. .venv/bin/python scripts/aws/provision_worker.py (one-time; refuses existing role)
6. Commit exact source, then scripts/aws/launch_cpu_worker.sh.
7. Inspect artifacts/cloud/phase5/P5-INDEX-001/ledger.json and EC2 console output/S3 run prefix.
8. Emergency: scripts/aws/terminate_workers.sh (only tagged Phase5 workers).

This is the first index benchmark execution layer, not yet a general GPU/SageMaker launcher or full OOF pipeline. It builds persistent country/source shards from all training targets, preserves frozen IDF, and checks sampled candidate parity without labels. Both successful and failed jobs invoke OS shutdown; a scheduled90-minute shutdown is set before bootstrap, and the launch requests terminate behavior. Upload errors are failures, not successful checkpoints. A failure before code download may leave only EC2 console logs; inspect those before claiming completion.

Full training retrieval may generate Fold4 candidates without labels, but Fold4 recall/oracle/accuracy remain withheld until the architecture freeze. Full label-based benchmark before freeze means folds0–3 only. Do not silently violate this gate to report a full-data number.
