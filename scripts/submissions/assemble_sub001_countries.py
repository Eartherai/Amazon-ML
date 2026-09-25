"""Assemble verified frozen Mac France/India and EC2 US SUB-001 shards.

The runner's 64 SHA256 query shards are independent by country. This utility
hard-links complete, immutable shard files and refuses missing or changed US
upload receipts, wrong query counts, duplicate S1 IDs, or incomplete coverage.
It does not change any candidate or prediction content.
"""
import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


COUNTS = {"France": 259452, "India": 809986, "US": 663106}
HEADER = {"candidates": "source1_entity_id\tcandidate_entity_ids\n",
          "matching": "source1_entity_id\tmatched_entity_ids\n"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def verify_us_receipts(root: Path) -> None:
    shards = root / "shards"
    for number in range(64):
        base = f"US-s{number:03d}"
        receipts = json.loads((shards / f"{base}-receipts.json").read_text())
        if len(receipts) != 2:
            raise ValueError(f"US shard receipt pair is incomplete: {base}")
        expected = {f"{base}-{kind}.tsv.gz" for kind in HEADER}
        names = {row["key"].rsplit("/", 1)[-1] for row in receipts}
        if names != expected:
            raise ValueError(f"US receipt names disagree: {base}")
        for row in receipts:
            path = shards / row["key"].rsplit("/", 1)[-1]
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError(f"US shard differs from verified upload: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mac", type=Path, required=True)
    parser.add_argument("--us", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mac = json.loads((args.mac / "progress.json").read_text())
    us = json.loads((args.us / "COMPLETE.json").read_text())
    mac_done = {row["country"]: row for row in mac["country_progress"]}
    if any(
        mac_done.get(country, {}).get("queries") != COUNTS[country]
        or mac_done[country].get("shards") != 64
        for country in ("France", "India")
    ):
        raise ValueError("Mac France/India have not both completed all 64 shards")
    us_done = us["country_progress"]
    if us["countries"] != ["US"] or len(us_done) != 1 or us_done[0]["queries"] != COUNTS["US"] or us_done[0]["shards"] != 64:
        raise ValueError("US country worker did not finish all 663106 queries")
    if mac["model_sha256"] != us["model_sha256"] or mac["config"] != us["config"] or mac["fold4"] != "CLOSED" or us["fold4"] != "CLOSED":
        raise ValueError("Frozen model or config disagrees between workers")
    if mac["query_total"] != us["query_total"] != sum(COUNTS.values()):
        raise ValueError("Query universe disagrees")
    verify_us_receipts(args.us)
    if args.output.exists():
        raise FileExistsError(args.output)
    (args.output / "shards").mkdir(parents=True)
    seen = set()
    counted = Counter()
    file_hashes = {}
    for country, root in (("France", args.mac), ("India", args.mac), ("US", args.us)):
        for number in range(64):
            base = f"{country}-s{number:03d}"
            paths = {kind: root / "shards" / f"{base}-{kind}.tsv.gz" for kind in HEADER}
            for kind, source in paths.items():
                if not source.is_file():
                    raise FileNotFoundError(source)
                file_hashes[source.name] = digest(source)
                os.link(source, args.output / "shards" / source.name)
            with gzip.open(paths["matching"], "rt", encoding="utf-8") as matching, gzip.open(paths["candidates"], "rt", encoding="utf-8") as candidates:
                if matching.readline() != HEADER["matching"] or candidates.readline() != HEADER["candidates"]:
                    raise ValueError(f"Wrong shard header: {base}")
                previous = ""
                while True:
                    match_row = matching.readline()
                    candidate_row = candidates.readline()
                    if not match_row and not candidate_row:
                        break
                    if not match_row or not candidate_row:
                        raise ValueError(f"Shard row counts disagree: {base}")
                    match_id, tab1, _ = match_row.partition("\t")
                    candidate_id, tab2, _ = candidate_row.partition("\t")
                    if not tab1 or not tab2 or match_id != candidate_id or (previous and match_id <= previous):
                        raise ValueError(f"Misaligned or unsorted shard: {base}")
                    if int.from_bytes(hashlib.sha256(match_id.encode()).digest()[:8], "big") % 64 != number:
                        raise ValueError(f"Wrong query shard: {match_id}")
                    if match_id in seen:
                        raise ValueError(f"Duplicate S1 ID: {match_id}")
                    seen.add(match_id)
                    counted[country] += 1
                    previous = match_id
    if dict(counted) != COUNTS or len(seen) != sum(COUNTS.values()) or len(file_hashes) != 384:
        raise ValueError(f"Incomplete assembled query coverage: {dict(counted)}")
    report = {**mac, "processed_query_count": len(seen), "countries": sorted(COUNTS),
              "country_progress": [mac_done["France"], mac_done["India"], us_done[0]],
              "assembly": {"mac": str(args.mac.resolve()), "us": str(args.us.resolve()),
                           "file_sha256": file_hashes, "country_query_counts": dict(counted)}}
    report.pop("active", None)
    (args.output / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "queries": len(seen), "shard_files": len(file_hashes)}))


if __name__ == "__main__":
    main()
