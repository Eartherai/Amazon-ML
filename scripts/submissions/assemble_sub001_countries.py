"""Assemble verified frozen Mac and EC2 SUB-001 country shards.

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
INDIA_CLOUD_COUNT = 709176
INDIA_FIRST_COUNT = 100810
INDIA_LOWER_COUNT = 354229
INDIA_UPPER_COUNT = 354947
HEADER = {"candidates": "source1_entity_id\tcandidate_entity_ids\n",
          "matching": "source1_entity_id\tmatched_entity_ids\n"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def verify_cloud_receipts(root: Path, country: str, first: int, last: int) -> None:
    shards = root / "shards"
    for number in range(first, last):
        base = f"{country}-s{number:03d}"
        receipts = json.loads((shards / f"{base}-receipts.json").read_text())
        if len(receipts) != 2:
            raise ValueError(f"Cloud shard receipt pair is incomplete: {base}")
        expected = {f"{base}-{kind}.tsv.gz" for kind in HEADER}
        names = {row["key"].rsplit("/", 1)[-1] for row in receipts}
        if names != expected:
            raise ValueError(f"Cloud receipt names disagree: {base}")
        for row in receipts:
            path = shards / row["key"].rsplit("/", 1)[-1]
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError(f"Cloud shard differs from verified upload: {path}")


def verify_ready(root: Path, filename: str, country: str, first: int, last: int,
                 expected_count: int, model_sha256: str) -> None:
    ready = json.loads((root / filename).read_text())
    if (ready.get("country") != country or ready.get("first_shard") != first
            or ready.get("last_shard") != last or ready.get("query_count") != expected_count
            or ready.get("pairs") != last - first or ready.get("model_sha256") != model_sha256):
        raise ValueError(f"Cloud segment readiness receipt disagrees: {root}")
    if filename == "SEGMENT_READY.json" and ready.get("partial_worker_terminated") is not True:
        raise ValueError("Lower India worker was not stopped at the boundary")
    verify_cloud_receipts(root, country, first, last)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mac", type=Path, required=True)
    parser.add_argument("--us", type=Path, required=True)
    parser.add_argument("--india", type=Path)
    parser.add_argument("--india-upper", type=Path,
                        help="Completed India 36-63 worker; --india then supplies stopped 8-35 segment")
    parser.add_argument("--query-universe", type=Path,
                        help="Frozen test queries.parquet; required for production assembly")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mac = json.loads((args.mac / "progress.json").read_text())
    us = json.loads((args.us / "COMPLETE.json").read_text())
    mac_done = {row["country"]: row for row in mac["country_progress"]}
    if mac_done.get("France", {}).get("queries") != COUNTS["France"] or mac_done["France"].get("shards") != 64:
        raise ValueError("Mac France has not completed all 64 shards")
    if args.india_upper and not args.india:
        raise ValueError("Split India assembly also requires --india lower segment")
    if args.india:
        first = mac_done.get("India", mac.get("active", {}))
        if first.get("country") != "India" or first.get("shards", 0) < 8 or first.get("queries", 0) < INDIA_FIRST_COUNT:
            # A fully completed Mac India country is also acceptable as prefix provenance.
            if mac_done.get("India", {}).get("queries") != COUNTS["India"]:
                raise ValueError("Mac India shards 0-7 have not completed")
        if args.india_upper:
            lower = json.loads((args.india / "SEGMENT_READY.json").read_text())
            upper = json.loads((args.india_upper / "COMPLETE.json").read_text())
            if (upper["countries"] != ["India"] or upper["processed_query_count"] != INDIA_UPPER_COUNT
                    or upper["country_progress"][0]["shards"] != 28
                    or upper["country_progress"][0]["queries"] != INDIA_UPPER_COUNT
                    or upper["query_total"] != sum(COUNTS.values())):
                raise ValueError("Cloud India 36-63 is incomplete")
            if (lower.get("run_id") != "P5-SUB001-INDIA-001"
                    or upper["model_sha256"] != mac["model_sha256"]
                    or upper["config"] != mac["config"] or upper["fold4"] != "CLOSED"):
                raise ValueError("Split India frozen model/config disagrees")
            verify_ready(args.india, "SEGMENT_READY.json", "India", 8, 36,
                         INDIA_LOWER_COUNT, mac["model_sha256"])
            verify_ready(args.india_upper, "DOWNLOAD_READY.json", "India", 36, 64,
                         INDIA_UPPER_COUNT, mac["model_sha256"])
            india_sources = {"Mac": "0-7", "EC2-lower": "8-35", "EC2-upper": "36-63"}
        else:
            india = json.loads((args.india / "COMPLETE.json").read_text())
            if (india["countries"] != ["India"] or len(india["country_progress"]) != 1
                    or india["country_progress"][0]["shards"] != 56
                    or india["country_progress"][0]["queries"] != INDIA_CLOUD_COUNT
                    or india["processed_query_count"] != INDIA_CLOUD_COUNT
                    or india["query_total"] != sum(COUNTS.values())):
                raise ValueError("Cloud India 8-63 is incomplete")
            if india["model_sha256"] != mac["model_sha256"] or india["config"] != mac["config"] or india["fold4"] != "CLOSED":
                raise ValueError("Cloud India frozen model/config disagrees")
            verify_cloud_receipts(args.india, "India", 8, 64)
            india_sources = {"Mac": "0-7", "EC2": "8-63"}
        india_progress = {"country": "India", "queries": COUNTS["India"], "shards": 64,
                          "sources": india_sources}
    else:
        if mac_done.get("India", {}).get("queries") != COUNTS["India"] or mac_done["India"].get("shards") != 64:
            raise ValueError("Mac India has not completed all 64 shards")
        india_progress = mac_done["India"]
    us_done = us["country_progress"]
    if us["countries"] != ["US"] or len(us_done) != 1 or us_done[0]["queries"] != COUNTS["US"] or us_done[0]["shards"] != 64:
        raise ValueError("US country worker did not finish all 663106 queries")
    if mac["model_sha256"] != us["model_sha256"] or mac["config"] != us["config"] or mac["fold4"] != "CLOSED" or us["fold4"] != "CLOSED":
        raise ValueError("Frozen model or config disagrees between workers")
    if mac["query_total"] != us["query_total"] or us["query_total"] != sum(COUNTS.values()):
        raise ValueError("Query universe disagrees")
    verify_cloud_receipts(args.us, "US", 0, 64)
    if sum(COUNTS.values()) == 1732544 and not args.query_universe:
        raise ValueError("Production assembly requires the frozen query universe")
    expected_ids = None
    if args.query_universe:
        import polars as pl
        frame = pl.read_parquet(args.query_universe, columns=["entity_id", "country"])
        if len(frame) != sum(COUNTS.values()) or frame["entity_id"].n_unique() != len(frame):
            raise ValueError("Frozen query universe count or uniqueness disagrees")
        expected_ids = {country: set(frame.filter(pl.col("country") == country)["entity_id"].to_list())
                        for country in COUNTS}
        if {country: len(ids) for country, ids in expected_ids.items()} != COUNTS:
            raise ValueError("Frozen country query counts disagree")
    if args.output.exists():
        raise FileExistsError(args.output)
    (args.output / "shards").mkdir(parents=True)
    seen = set()
    counted = Counter()
    segment_counted = Counter()
    seen_by_country = {country: set() for country in COUNTS}
    file_hashes = {}
    for country in COUNTS:
        for number in range(64):
            root = (args.us if country == "US" else
                    args.india_upper if country == "India" and args.india_upper and number >= 36 else
                    args.india if country == "India" and args.india and number >= 8 else args.mac)
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
                    seen_by_country[country].add(match_id)
                    counted[country] += 1
                    if country == "India":
                        segment_counted["0-7" if number < 8 else "8-35" if number < 36 else "36-63"] += 1
                    previous = match_id
    if dict(counted) != COUNTS or len(seen) != sum(COUNTS.values()) or len(file_hashes) != 384:
        raise ValueError(f"Incomplete assembled query coverage: {dict(counted)}")
    if args.india_upper and dict(segment_counted) != {"0-7": INDIA_FIRST_COUNT,
                                                       "8-35": INDIA_LOWER_COUNT,
                                                       "36-63": INDIA_UPPER_COUNT}:
        raise ValueError(f"India segment query counts disagree: {dict(segment_counted)}")
    if expected_ids is not None and seen_by_country != expected_ids:
        raise ValueError("Assembled S1 IDs differ from frozen test query universe")
    report = {**mac, "processed_query_count": len(seen), "countries": sorted(COUNTS),
              "country_progress": [mac_done["France"], india_progress, us_done[0]],
              "assembly": {"mac": str(args.mac.resolve()), "us": str(args.us.resolve()),
                           "india": str(args.india.resolve()) if args.india else None,
                           "india_upper": str(args.india_upper.resolve()) if args.india_upper else None,
                           "query_universe": str(args.query_universe.resolve()) if args.query_universe else None,
                           "file_sha256": file_hashes, "country_query_counts": dict(counted)}}
    report.pop("active", None)
    (args.output / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "queries": len(seen), "shard_files": len(file_hashes)}))


if __name__ == "__main__":
    main()
