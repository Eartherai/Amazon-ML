#!/usr/bin/env bash
# Finish the frozen submission immediately after the verified 384-shard receipt.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export AWS_PROFILE=amamzon_01_a1_0
receipt=artifacts/cloud/phase5/sub001-mac-shards-upload.complete.json
validated=outputs/submissions/SUB-001/validated-v001

until test -f "$receipt"; do
  sleep 20
done
echo "$(date -u +%FT%TZ) all 384 verified shard files staged; launching official validator"
.venv/bin/python scripts/aws/launch_sagemaker_validator.py --launch

while true; do
  state=$(aws sagemaker describe-processing-job --region us-east-1 \
    --processing-job-name P5-SUB001-SM-VALIDATE-001 \
    --query ProcessingJobStatus --output text)
  case "$state" in
    Completed) break ;;
    Failed|Stopped) echo "Official validator ended $state" >&2; exit 1 ;;
    InProgress|Stopping) sleep 30 ;;
    *) echo "Unexpected validator state: $state" >&2; exit 1 ;;
  esac
done
echo "$(date -u +%FT%TZ) official SageMaker validator completed"
.venv/bin/python scripts/submissions/download_validated.py --matching-only --output "$validated"
# The AWS merge validates S1 alignment and match-to-candidate membership
# independently before both official validator modes run. Download verifies
# the raw matching TSV SHA256; no local inference or second compute pipeline runs.
shasum -a 256 "$validated/matching_results.tsv" \
  > outputs/submissions/SUB-001/final-upload-hashes.sha256
echo "$(date -u +%FT%TZ) FINAL_MATCHING_TSV_READY"
