#!/usr/bin/env bash
# Emergency termination restricted to this project's tagged finite workers.
set -euo pipefail
profile="${AWS_PROFILE:-amamzon_01_a1_0}"
ids=$(aws --profile "$profile" --region us-east-1 ec2 describe-instances --filters Name=tag:Project,Values=aml2026-phase5 Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)
if [[ -n "$ids" && "$ids" != None ]]; then
  read -r -a workers <<< "$ids"
  aws --profile "$profile" --region us-east-1 ec2 terminate-instances --instance-ids "${workers[@]}"
fi
