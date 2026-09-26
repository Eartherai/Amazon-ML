#!/bin/bash
# Download a presigned S3 URL over IPv4 with retries: dl.sh OUT_PATH URL
set -euo pipefail
curl -4 -sS --retry 6 --retry-all-errors --connect-timeout 20 -o "$1" "$2"
ls -la "$1"
