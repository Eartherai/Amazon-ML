"""Verify all EXP-031 archives and old-20k route parity before feature launch.

This reads only unlabeled candidate routes. It compares exact candidate IDs and
ranks with the frozen P4-B-002 name/address routes for every old 20k query.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

import duckdb
import polars as pl


ROUTE_NAME = re.compile(r"^(name|address)-c(\d{3})-s(\d{3})$")
MEMBER_NAME = re.compile(r"^(name|address)-c\d{3}-s\d{3}-b\d{5}\.parquet$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_archive(archive: Path, summary: Path, destination: Path, old_ids: set[str]) -> None:
    receipt = json.loads((summary / f"{archive.stem}-receipt.json").read_text())
    if receipt["bytes"] != archive.stat().st_size or receipt["sha256"] != sha256(archive):
        raise ValueError(f"Archive checksum mismatch: {archive.name}")
    selected = []
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if not members:
            raise ValueError(f"Empty route archive: {archive.name}")
        for member in members:
            if not member.isfile() or not MEMBER_NAME.fullmatch(member.name) or not member.name.startswith(archive.stem + "-"):
                raise ValueError(f"Unexpected archive member: {archive.name}/{member.name}")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            frame = pl.read_parquet(io.BytesIO(stream.read())).select(
                "source1_entity_id", "target_id", "route_score", "route_rank"
            ).filter(pl.col("source1_entity_id").is_in(old_ids))
            if len(frame):
                selected.append(frame)
    if selected:
        pl.concat(selected).write_parquet(destination, compression="zstd")


def compare_route(field: str, reference: Path, selected_glob: str) -> dict:
    con = duckdb.connect(config={"threads": 2, "memory_limit": "3GB"})
    con.read_parquet(str(reference)).create_view("reference")
    con.read_parquet(selected_glob).create_view("actual")
    counts = con.execute("SELECT (SELECT count(*) FROM reference),(SELECT count(*) FROM actual),"
                         "(SELECT count(DISTINCT source1_entity_id) FROM actual)").fetchone()
    missing = con.execute("SELECT count(*) FROM reference r ANTI JOIN actual a ON "
                          "r.source1_entity_id=a.source1_entity_id AND r.target_id=a.target_id "
                          "AND r.route_rank=a.route_rank").fetchone()[0]
    extra = con.execute("SELECT count(*) FROM actual a ANTI JOIN reference r ON "
                        "r.source1_entity_id=a.source1_entity_id AND r.target_id=a.target_id "
                        "AND r.route_rank=a.route_rank").fetchone()[0]
    score_delta = con.execute("SELECT max(abs(r.route_score-a.route_score)) FROM reference r JOIN actual a ON "
                              "r.source1_entity_id=a.source1_entity_id AND r.target_id=a.target_id "
                              "AND r.route_rank=a.route_rank").fetchone()[0]
    con.close()
    result = {"field": field, "reference_pairs": counts[0], "actual_pairs": counts[1],
              "actual_queries": counts[2], "missing_id_or_rank": missing,
              "extra_id_or_rank": extra, "max_abs_score_difference": score_delta}
    if counts[0] != counts[1] or counts[2] != 20_000 or missing or extra or score_delta is None or score_delta > 1e-5:
        raise ValueError(f"Route parity failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--old-ids", type=Path, default=Path("artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/candidates/P4-B-002"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(".").free < 9 * 1024**3:
        raise RuntimeError("Insufficient disk reserve for parity scratch")
    complete = json.loads((args.summary / "COMPLETE.json").read_text())
    if complete.get("query_count") != 200_000 or complete.get("shards") != 64 or len(complete.get("routes", [])) != 4:
        raise ValueError("Incomplete EXP-031 retrieval summary")
    old_ids = set(pl.read_parquet(args.old_ids, columns=["entity_id"])["entity_id"])
    if len(old_ids) != 20_000:
        raise ValueError("Old sample must contain exactly 20k unique IDs")
    expected = {(field, country, shard) for field in ("name", "address") for country in range(2) for shard in range(64)}
    archive_map = {}
    for archive in args.archives.glob("*.tar"):
        match = ROUTE_NAME.fullmatch(archive.stem)
        if not match:
            raise ValueError(f"Unexpected archive: {archive.name}")
        key = (match.group(1), int(match.group(2)), int(match.group(3)))
        if key in archive_map:
            raise ValueError(f"Duplicate route archive: {archive.name}")
        archive_map[key] = archive
    if set(archive_map) != expected:
        raise ValueError(f"Route archive inventory mismatch: {len(archive_map)} of {len(expected)}")
    with tempfile.TemporaryDirectory(prefix="aml-route-parity-") as temp:
        scratch = Path(temp)
        for field, country, shard in sorted(expected):
            archive = archive_map[(field, country, shard)]
            destination = scratch / f"{field}-c{country:03d}-s{shard:03d}.parquet"
            validate_archive(archive, args.summary, destination, old_ids)
        results = [compare_route(field, args.reference / f"{field}_char3.parquet",
                                 str(scratch / f"{field}-*.parquet"))
                   for field in ("name", "address")]
    report = {"experiment": "EXP-031", "scope": "All 256 SHA256-verified route archives; every old 20k S1 route compared with frozen P4-B-002",
              "query_count": complete["query_count"], "route_archives": len(archive_map),
              "summary_sha256": sha256(args.summary / "COMPLETE.json"), "old_ids_sha256": sha256(args.old_ids),
              "results": results, "parity": "PASS", "fold4_labels": "NOT READ"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
