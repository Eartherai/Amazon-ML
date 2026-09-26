#!/bin/bash
# Copy a matching TSV to outputs/submissions/UPLOAD_<NAME>/, verify the copy's SHA256, and run the
# official validator (default and --check-ids) plus the independent strict check. Usage: make_upload.sh SRC NAME
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC="$1"; NAME="$2"; CSRC="${3:-}"; D="outputs/submissions/UPLOAD_${NAME}"; F="$D/UPLOAD_${NAME}_matching_results.tsv"
mkdir "$D"; cp "$SRC" "$F"
a=$(shasum -a 256 "$SRC" | cut -d' ' -f1); b=$(shasum -a 256 "$F" | cut -d' ' -f1)
[ "$a" = "$b" ] || { echo "COPY HASH MISMATCH"; exit 1; }
CARG=(); SARG=()
if [ -n "$CSRC" ]; then CF="$D/UPLOAD_${NAME}_candidate_pairs.tsv"; cp "$CSRC" "$CF"
  [ "$(shasum -a 256 "$CSRC" | cut -d' ' -f1)" = "$(shasum -a 256 "$CF" | cut -d' ' -f1)" ] || { echo "CAND COPY HASH MISMATCH"; exit 1; }
  CARG=(--candidate "$CF"); SARG=(--candidate "$CF"); fi
T=student_resource/dataset/test
e1=0; .venv/bin/python student_resource/utils/validate_submission.py --matching "$F" "${CARG[@]}" --test-dir $T > "$D/official_default.log" 2>&1 || e1=$?
e2=0; .venv/bin/python student_resource/utils/validate_submission.py --matching "$F" "${CARG[@]}" --test-dir $T --check-ids > "$D/official_check_ids.log" 2>&1 || e2=$?
e3=0; .venv/bin/python scripts/submissions/strict_check.py --matching "$F" "${SARG[@]}" --test-dir $T > "$D/strict_check.json" 2>&1 || e3=$?
echo "{\"name\":\"$NAME\",\"file\":\"$F\",\"sha256\":\"$b\",\"official_default_exit\":$e1,\"official_check_ids_exit\":$e2,\"strict_exit\":$e3}" | tee "$D/receipt.json"
tail -2 "$D/official_default.log" "$D/official_check_ids.log"; grep -E '"pass"|errors' "$D/strict_check.json"
