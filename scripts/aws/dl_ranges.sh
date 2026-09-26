#!/bin/bash
# Parallel ranged download of a presigned S3 URL with per-chunk retries: dl_ranges.sh OUT SIZE_BYTES URL [CHUNK_MB]
set -uo pipefail
OUT="$1"; SIZE="$2"; URL="$3"; CH=$(( ${4:-4} * 1024 * 1024 )); T=$(mktemp -d)
n=$(( (SIZE + CH - 1) / CH ))
for ((i=0; i<n; i++)); do
  s=$(( i * CH )); e=$(( s + CH - 1 )); [ $e -ge $SIZE ] && e=$(( SIZE - 1 ))
  ( for a in 1 2 3 4 5 6 7 8; do
      curl -4 -sS --connect-timeout 15 --max-time 180 -r "$s-$e" -o "$T/p$i" "$URL" && [ "$(stat -f%z "$T/p$i")" -eq $(( e - s + 1 )) ] && exit 0
      sleep 2; done; echo "chunk $i failed"; exit 1 ) &
  while [ "$(jobs -rp | wc -l)" -ge 8 ]; do sleep 0.3; done
done
wait
for ((i=0; i<n; i++)); do cat "$T/p$i"; done > "$OUT"; rm -rf "$T"
ls -la "$OUT"; [ "$(stat -f%z "$OUT")" -eq "$SIZE" ] && echo SIZE_OK || echo SIZE_MISMATCH
