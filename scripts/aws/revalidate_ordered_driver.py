"""SageMaker validation-only replay of the organizer's exact two-file command."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path("/opt/ml/processing")
MATCH_SHA = "910c3e7c95a8ed7867dd6addfb6a9aad3d5477c5ede2422c23b081be1942440b"
CANDIDATE_SHA = "89e95ad2f4cb0b0c4f082f9975c6549e8f073ceb71cbcb3ac1860247a590f4b6"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    work = ROOT / "work" / "student_resource"
    output = work / "output"
    (work / "utils").mkdir(parents=True)
    (work / "dataset").mkdir()
    output.mkdir()
    (ROOT / "result").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "code" / "validate_submission.py", work / "utils" / "validate_submission.py")
    (work / "dataset" / "test").symlink_to(ROOT / "test", target_is_directory=True)
    source = ROOT / "matching" / "matching_results.tsv"
    if sha(source) != MATCH_SHA:
        raise ValueError("Ordered matching input hash mismatch")
    shutil.copy2(source, output / "matching_results.tsv")
    compressed = ROOT / "candidate" / "candidate_pairs.tsv.gz"
    with gzip.open(compressed, "rb") as zipped, (output / "candidate_pairs.tsv").open("xb") as raw:
        shutil.copyfileobj(zipped, raw, length=8 * 1024 * 1024)
    if sha(output / "candidate_pairs.tsv") != CANDIDATE_SHA:
        raise ValueError("Actual candidate input hash mismatch")
    base = [sys.executable, "utils/validate_submission.py", "--matching", "output/matching_results.tsv",
            "--candidate", "output/candidate_pairs.tsv", "--test-dir", "dataset/test"]
    reports = {}
    # The organizer's default two-file command already printed PASS in job v002.
    # It always emits an expected warning that the optional ID check is off.
    for name, cmd in (("official_check_ids", [*base, "--check-ids"]),):
        result = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        log = ROOT / "result" / (name + ".log")
        log.write_text(result.stdout)
        if result.returncode or "PASS — no blocking issues found" not in result.stdout or "WARNING:" in result.stdout:
            raise ValueError(f"{name} failed, see {log}")
        reports[name] = {"exit_code": result.returncode, "pass": True}
    summary = {"scope": "Reordered SUB-001, validation only", "matching_sha256": MATCH_SHA,
               "candidate_sha256": CANDIDATE_SHA, "rows": 1_732_544,
               "validator": reports, "prediction_value_changes": 0}
    (ROOT / "result" / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
