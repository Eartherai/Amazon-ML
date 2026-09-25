"""Run the frozen SUB-001 merger and official validator in SageMaker Processing.

This script needs only the Python standard library. Processing inputs are mounted
under /opt/ml/processing; SageMaker uploads /opt/ml/processing/result at job end.
"""
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path("/opt/ml/processing")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    code = ROOT / "code"
    test = ROOT / "test"
    shards = ROOT / "shards"
    work = ROOT / "work"
    result = ROOT / "result"
    result.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((code / "manifest.json").read_text())
    for item in manifest["files"]:
        path = code / item["name"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Validation code checksum mismatch: {item['name']}")
    student = work / "student_resource"
    (student / "utils").mkdir(parents=True)
    (student / "dataset").mkdir()
    (student / "dataset" / "test").symlink_to(test, target_is_directory=True)
    shutil.copy2(code / "validate_submission.py", student / "utils" / "validate_submission.py")
    matching = sorted(shards.glob("*-matching.tsv.gz"))
    candidates = sorted(shards.glob("*-candidates.tsv.gz"))
    if len(matching) != 192 or len(candidates) != 192:
        raise ValueError(f"Expected 192 complete matching/candidate shards, got {len(matching)}/{len(candidates)}")
    if {x.name.removesuffix("-matching.tsv.gz") for x in matching} != {
        x.name.removesuffix("-candidates.tsv.gz") for x in candidates
    }:
        raise ValueError("Shard inventories disagree")
    for filename in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
        if not (test / filename).is_file():
            raise FileNotFoundError(filename)
    output = work / "output"
    command = [sys.executable, str(code / "merge_validate.py"), "--shards", str(shards),
               "--output", str(output), "--test-dir", str(test), "--expected", "1732544"]
    subprocess.run(command, cwd=work, check=True)
    validation = json.loads((output / "validation.json").read_text())
    if validation["rows"] != 1732544 or any(not item["pass"] for item in validation["validation"].values()):
        raise ValueError("Official validation did not pass")
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        source = output / name
        with source.open("rb") as original, gzip.open(result / (name + ".gz"), "wb", compresslevel=4) as compressed:
            shutil.copyfileobj(original, compressed, length=8 * 1024**2)
    for name in ("validation.json", "official.log", "official_check_ids.log"):
        shutil.copy2(output / name, result / name)
    (result / "manifest.json").write_text(json.dumps({
        "scope": "Frozen SUB-001 complete test results; Fold4 CLOSED",
        "rows": validation["rows"],
        "matching_sha256": validation["matching_sha256"],
        "candidate_sha256": validation["candidate_sha256"],
        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in result.iterdir() if p.is_file()},
    }, indent=2) + "\n")
    print(json.dumps({"official_pass": True, "rows": validation["rows"], "matching_sha256": validation["matching_sha256"], "candidate_sha256": validation["candidate_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
