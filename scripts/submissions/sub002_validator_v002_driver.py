"""Merge SUB-002 shards, apply OOF-selected rules, and validate exact outputs."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path("/opt/ml/processing")
SCOPE = "SUB-002 200k with OOF-selected source completion and ownership; Fold4 CLOSED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_official(student: Path, output: Path, strict: bool) -> dict:
    name = "official_check_ids" if strict else "official"
    cmd = ["python3", "utils/validate_submission.py", "--matching", "output/matching_results.tsv",
           "--candidate", "output/candidate_pairs.tsv", "--test-dir", "dataset/test"]
    if strict:
        cmd.append("--check-ids")
    result = subprocess.run(cmd, cwd=student, capture_output=True, text=True)
    (output / (name + ".log")).write_text(result.stdout + "\n" + result.stderr)
    if result.returncode or "PASS" not in result.stdout:
        raise RuntimeError(f"Official validator failed in {name}")
    warnings = [line for line in result.stdout.splitlines() if line.startswith("WARNING:")]
    if strict and warnings:
        raise RuntimeError("Strict official validator emitted warnings")
    if not strict and (len(warnings) != 1 or "ID-existence check is OFF" not in warnings[0]):
        raise RuntimeError("Unexpected default official warning")
    return {"exit_code": 0, "pass": True}


def main() -> None:
    code = ROOT / "code"
    test = ROOT / "test"
    work = ROOT / "work"
    shards = work / "shards"
    scores = work / "scores"
    shards.mkdir(parents=True)
    scores.mkdir()
    result = ROOT / "result"
    result.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((code / "manifest.json").read_text())
    for item in manifest["files"]:
        path = code / item["name"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Code checksum mismatch: {item['name']}")

    for index in range(8):
        source_dir = ROOT / f"shards{index}"
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        for kind, target_dir in (("shards", shards), ("scores", scores)):
            folder = source_dir / kind
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            for source in folder.glob("*.tsv.gz"):
                suffixes = ("-matching.tsv.gz", "-candidates.tsv.gz") if kind == "shards" else ("-scores.tsv.gz",)
                if not source.name.endswith(suffixes):
                    raise ValueError("Unexpected shard: " + source.name)
                target = target_dir / source.name
                if target.exists():
                    raise ValueError("Duplicate shard: " + source.name)
                target.symlink_to(source)
    matching = sorted(shards.glob("*-matching.tsv.gz"))
    candidates = sorted(shards.glob("*-candidates.tsv.gz"))
    score_files = sorted(scores.glob("*-scores.tsv.gz"))
    if (len(matching), len(candidates), len(score_files)) != (192, 192, 192):
        raise ValueError("Incomplete matching/candidate/score shard inventory")
    bases = lambda files, suffix: {path.name.removesuffix(suffix) for path in files}
    if bases(matching, "-matching.tsv.gz") != bases(candidates, "-candidates.tsv.gz") or bases(matching, "-matching.tsv.gz") != bases(score_files, "-scores.tsv.gz"):
        raise ValueError("Shard stem inventories disagree")
    for name in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
        if not (test / name).is_file():
            raise FileNotFoundError(name)
    student = work / "student_resource"
    (student / "utils").mkdir(parents=True)
    (student / "dataset").mkdir()
    (student / "dataset" / "test").symlink_to(test, target_is_directory=True)
    shutil.copy2(code / "validate_submission.py", student / "utils" / "validate_submission.py")
    output = work / "output"
    subprocess.run([sys.executable, str(code / "merge_validate.py"), "--shards", str(shards),
                    "--output", str(output), "--test-dir", str(test), "--expected", "1732544"],
                   cwd=work, check=True)
    original = json.loads((output / "validation.json").read_text())
    if original["rows"] != 1_732_544 or any(not v["pass"] for v in original["validation"].values()):
        raise ValueError("Vanilla merger validation did not pass")

    updated = output / "matching_results.postprocessed.tmp"
    subprocess.run([sys.executable, str(code / "postprocess_sub002.py"),
                    "--matching", str(output / "matching_results.tsv"), "--scores", str(scores),
                    "--output", str(updated), "--expected", "1732544"], cwd=work, check=True)
    post = json.loads((output / "postprocess_report.json").read_text())
    if post["rows"] != 1_732_544 or post["score_shards"] != 192:
        raise ValueError("Postprocess report incomplete")
    os.replace(updated, output / "matching_results.tsv")
    checks = {"official": run_official(student, output, False),
              "official_check_ids": run_official(student, output, True)}
    final = {**original, "matching_sha256": sha256(output / "matching_results.tsv"),
             "validation": checks, "postprocess": post, "scope": SCOPE,
             "vanilla_matching_sha256": original["matching_sha256"]}
    (output / "validation.json").write_text(json.dumps(final, indent=2) + "\n")
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        with (output / name).open("rb") as raw, gzip.open(result / (name + ".gz"), "wb", compresslevel=4) as zipped:
            shutil.copyfileobj(raw, zipped, length=8 * 1024**2)
    for name in ("validation.json", "official.log", "official_check_ids.log", "postprocess_report.json"):
        shutil.copy2(output / name, result / name)
    (result / "manifest.json").write_text(json.dumps({
        "scope": SCOPE, "rows": final["rows"], "matching_sha256": final["matching_sha256"],
        "candidate_sha256": final["candidate_sha256"], "files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in result.iterdir() if path.is_file()},
    }, indent=2) + "\n")
    print(json.dumps({"official_pass": True, "rows": final["rows"],
                      "matching_sha256": final["matching_sha256"], "postprocess": post}), flush=True)


if __name__ == "__main__":
    main()
