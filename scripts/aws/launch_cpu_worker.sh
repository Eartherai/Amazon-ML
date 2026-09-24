#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export AWS_PROFILE="${AWS_PROFILE:-amamzon_01_a1_0}"
exec .venv/bin/python scripts/aws/launch_cpu_worker.py "$@"
