#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export AWS_PROFILE=amamzon_01_a1_0
while true; do
  state=$(aws sagemaker describe-processing-job --region us-east-1 \
    --processing-job-name P5-SUB001-SM-VALIDATE-001 \
    --query ProcessingJobStatus --output text)
  echo "$(date -u +%FT%TZ) $state"
  case "$state" in
    Completed) break ;;
    Failed|Stopped) exit 1 ;;
    InProgress|Stopping) sleep 30 ;;
    *) exit 1 ;;
  esac
done
.venv/bin/python scripts/submissions/download_validated.py --matching-only \
  --output outputs/submissions/SUB-001/validated-v001
shasum -a 256 outputs/submissions/SUB-001/validated-v001/matching_results.tsv \
  > outputs/submissions/SUB-001/final-upload-hashes.sha256
echo FINAL_MATCHING_TSV_READY
