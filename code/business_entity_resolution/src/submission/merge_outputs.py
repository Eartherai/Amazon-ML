"""Merge every scored SUB-001 shard, then validate the complete outputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import subprocess
import sys
from pathlib import Path

import duckdb


HEADERS = {"matching": "source1_entity_id\tmatched_entity_ids\n",
           "candidates": "source1_entity_id\tcandidate_entity_ids\n"}


def rows(path: Path, header: str):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        if stream.readline() != header:
            raise ValueError(f"Wrong shard header: {path}")
        last = ""
        for line in stream:
            key, tab, _ = line.partition("\t")
            if not tab or (last and key <= last):
                raise ValueError(f"Unsorted or duplicate shard IDs: {path}")
            last = key
            yield line


def merge(kind: str, shard_dir: Path, output: Path) -> int:
    files = sorted(shard_dir.glob(f"*-{kind}.tsv.gz"))
    if len(files) != 192:
        raise ValueError(f"Expected 192 {kind} shards, found {len(files)}")
    count, last = 0, ""
    with output.open("x", encoding="utf-8", newline="") as target:
        target.write(HEADERS[kind])
        for line in heapq.merge(*(rows(path, HEADERS[kind]) for path in files)):
            key = line.split("\t", 1)[0]
            if last and key <= last:
                raise ValueError("Duplicate or unsorted Source-1 ID across shards")
            target.write(line)
            last, count = key, count + 1
    if count != 1_732_544:
        raise ValueError(f"Incomplete {kind} output: {count}")
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(matching: Path, candidates: Path, test_dir: Path) -> None:
    con = duckdb.connect(config={"threads": 2, "memory_limit": "1GB"})
    expected = con.execute(
        "SELECT entity_id FROM read_csv(?,delim='\\t',header=true,all_varchar=true,"
        "nullstr='__AUDIT_IMPOSSIBLE_NULL_SENTINEL__',strict_mode=true) ORDER BY entity_id",
        [str(test_dir / "test_source1.tsv")],
    )
    with matching.open(encoding="utf-8") as matches, candidates.open(encoding="utf-8") as blocks:
        if matches.readline() != HEADERS["matching"] or blocks.readline() != HEADERS["candidates"]:
            raise ValueError("Output header mismatch")
        for expected_id, match_row, candidate_row in zip((row[0] for row in expected.fetchall()), matches, blocks, strict=True):
            mid, _, mtext = match_row.rstrip("\n").partition("\t")
            cid, _, ctext = candidate_row.rstrip("\n").partition("\t")
            if mid != expected_id or cid != expected_id:
                raise ValueError(f"Source-1 alignment or coverage error near {expected_id}")
            chosen = mtext.split(",") if mtext else []
            offered = ctext.split(",") if ctext else []
            if len(chosen) != len(set(chosen)) or len(offered) != len(set(offered)):
                raise ValueError(f"Duplicate target ID for {mid}")
            if not set(chosen) <= set(offered):
                raise ValueError(f"Unscored prediction for {mid}")
            if any(not item.startswith(("S2-", "S3-")) for item in offered):
                raise ValueError(f"Invalid target prefix for {mid}")
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--official-validator", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    matching, candidates = args.output / "matching_results.tsv", args.output / "candidate_pairs.tsv"
    merge("matching", args.shards, matching)
    merge("candidates", args.shards, candidates)
    validate(matching, candidates, args.test_dir)
    checks = {}
    if args.official_validator:
        for label, extra in (("official", []), ("official_check_ids", ["--check-ids"])):
            command = [sys.executable, str(args.official_validator.resolve()), "--matching", str(matching.resolve()),
                       "--candidate", str(candidates.resolve()), "--test-dir", str(args.test_dir.resolve()), *extra]
            result = subprocess.run(command, text=True, capture_output=True)
            (args.output / f"{label}.log").write_text(result.stdout + result.stderr)
            warnings = [line for line in result.stdout.splitlines() if line.startswith("WARNING:")]
            if result.returncode or "PASS" not in result.stdout or (extra and warnings):
                raise RuntimeError(f"{label} failed; inspect {label}.log")
            checks[label] = {"exit_code": result.returncode, "pass": True}
    receipt = {"rows": 1_732_544, "matching_sha256": sha256(matching),
               "candidate_sha256": sha256(candidates), "validation": checks}
    (args.output / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
