"""Download and SHA256-verify a finished frozen SUB-001 country partition."""
import argparse
import base64
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path


BUCKET = "aml2026-ber-08be19ac500747"
MODEL_SHA256 = "d84957742e05f5cd790d7dfc8c14ca05d3b5a2dc941a5094b8874d623b35117b"
PARTITIONS = {"P5-SUB001-US-001": ("US", 0, 64, 663106, 100),
              "P5-SUB001-INDIA-001": ("India", 8, 64, 709176, 87),
              "P5-SUB001-INDIA-HIGH-001": ("India", 36, 64, 354947, 46)}
LOWER_SEGMENT = ("India", 8, 36, 354229, 41)


def aws(*args: str) -> dict:
    command = ["aws", "--profile", os.environ.get("AWS_PROFILE", "amamzon_01_a1_0"),
               "--region", "us-east-1", "--no-cli-pager", *args, "--output", "json"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def verified_get(key: str, path: Path) -> dict:
    metadata = aws("s3api", "get-object", "--bucket", BUCKET, "--key", key,
                   "--checksum-mode", "ENABLED", str(path))
    checksum = metadata.get("ChecksumSHA256")
    local = digest(path)
    if not checksum or base64.b64encode(bytes.fromhex(local)).decode() != checksum:
        raise ValueError(f"S3 download checksum mismatch: {key}")
    if metadata.get("ContentLength") != path.stat().st_size:
        raise ValueError(f"S3 download byte length mismatch: {key}")
    return {"key": key, "sha256": local, "bytes": path.stat().st_size,
            "version_id": metadata.get("VersionId")}


def smoke_parity(downloaded: Path, smoke: Path, country: str, first: int, last: int,
                 expected_rows: int | None = None) -> dict:
    """Require exact cloud/Mac candidate and matching rows on frozen 100-query smoke IDs."""
    counts = {}
    for kind in ("candidates", "matching"):
        checked = 0
        for number in range(first, last):
            name = f"{country}-s{number:03d}-{kind}.tsv.gz"
            reference = smoke / name
            if not reference.exists():
                continue
            with gzip.open(reference, "rt", encoding="utf-8") as source:
                header = source.readline()
                expected = dict(line.rstrip("\n").split("\t", 1) for line in source)
            found = {}
            with gzip.open(downloaded / name, "rt", encoding="utf-8") as source:
                if source.readline() != header:
                    raise ValueError(f"Cloud/smoke header mismatch: {name}")
                for line in source:
                    query = line.split("\t", 1)[0]
                    if query in expected:
                        found[query] = line.rstrip("\n").split("\t", 1)[1]
            if found != expected:
                raise ValueError(f"Cloud/smoke content mismatch: {name}")
            checked += len(expected)
        counts[kind] = checked
    target = expected_rows if expected_rows is not None else (100 if country == "US" else 87)
    if counts != {"candidates": target, "matching": target}:
        raise ValueError(f"Incomplete cloud/smoke parity coverage: {counts}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=PARTITIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--partial-lower", action="store_true",
                        help="Download only the intentionally stopped India 8-35 worker segment")
    parser.add_argument("--boundary-stop", type=Path, default=Path(
        "artifacts/cloud/phase5/P5-SUB001-INDIA-001/boundary-stop.json"))
    args = parser.parse_args()
    if args.partial_lower and args.run_id != "P5-SUB001-INDIA-001":
        raise ValueError("Only the stopped lower India worker can be a partial segment")
    country, first, last, expected, smoke_count = (LOWER_SEGMENT if args.partial_lower
                                                      else PARTITIONS[args.run_id])
    prefix = f"amazon-ml-2026/phase5/runs/{args.run_id}/"
    args.output.mkdir(parents=True, exist_ok=False)
    if args.partial_lower:
        boundary = json.loads(args.boundary_stop.read_text())
        if (boundary.get("termination_requested") is not True
                or boundary.get("lower_complete_pairs") != 28
                or boundary.get("upper_first_pair_verified") is not True
                or boundary.get("lower", {}).get("run_id") != args.run_id
                or boundary.get("upper", {}).get("run_id") != "P5-SUB001-INDIA-HIGH-001"):
            raise ValueError("Lower India boundary was not safely stopped")
        instance_id = boundary["lower"]["id"]
        state = aws("ec2", "describe-instances", "--instance-ids", instance_id)
        record = state["Reservations"][0]["Instances"][0]
        tags = {item["Key"]: item["Value"] for item in record.get("Tags", [])}
        if record["State"]["Name"] != "terminated" or tags.get("RunId") != args.run_id:
            raise ValueError("Lower India worker has not terminated with the expected identity")
        complete_receipt = None
    else:
        complete_receipt = verified_get(prefix + "results/COMPLETE.json", args.output / "COMPLETE.json")
        report = json.loads((args.output / "COMPLETE.json").read_text())
        if (report["query_total"] != 1732544 or report["processed_query_count"] != expected
                or report["countries"] != [country] or report["model_sha256"] != MODEL_SHA256
                or report["fold4"] != "CLOSED" or len(report["country_progress"]) != 1
                or report["country_progress"][0]["queries"] != expected
                or report["country_progress"][0]["shards"] != last - first):
            raise ValueError("Cloud job has not completed its frozen country partition")
    shard_dir = args.output / "shards"
    shard_dir.mkdir()
    for number in range(first, last):
        base = f"{country}-s{number:03d}"
        receipts = []
        for kind in ("candidates", "matching"):
            name = f"{base}-{kind}.tsv.gz"
            receipts.append(verified_get(prefix + "shards/" + name, shard_dir / name))
        (shard_dir / f"{base}-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(json.dumps({"verified_country_shard": number - first + 1, "total": last - first,
                          "country": country}), flush=True)
    parity = smoke_parity(shard_dir, args.smoke, country, first, last, smoke_count)
    ready = {"run_id": args.run_id, "country": country, "first_shard": first,
             "last_shard": last, "query_count": expected, "pairs": last - first,
             "model_sha256": MODEL_SHA256, "complete_receipt": complete_receipt,
             "smoke_parity_rows": parity, "partial_worker_terminated": args.partial_lower}
    if args.partial_lower:
        ready["boundary_stop"] = str(args.boundary_stop.resolve())
        ready["instance_id"] = instance_id
    (args.output / ("SEGMENT_READY.json" if args.partial_lower else "DOWNLOAD_READY.json")).write_text(
        json.dumps(ready, indent=2) + "\n")


if __name__ == "__main__":
    main()
